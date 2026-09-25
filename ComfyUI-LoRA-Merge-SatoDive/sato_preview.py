"""
In-node preview rendering. Uses only native ComfyUI sampling primitives so every
architecture ComfyUI supports works; the presets only tune defaults and conditioning.
"""

import logging
import math

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers
import nodes

from . import sato_arch as A

log = logging.getLogger("SatoDive.LoRAMerge")

SCHEDULERS = list(comfy.samplers.KSampler.SCHEDULERS) + ["flux2"]
SAMPLERS = list(comfy.samplers.KSampler.SAMPLERS)


# --------------------------------------------------------------------------------------
# Conditioning helpers
# --------------------------------------------------------------------------------------

def _encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def _zero_out(conditioning):
    c = []
    for t in conditioning:
        d = t[1].copy()
        for k in ("pooled_output", "conditioning_lyrics"):
            v = d.get(k, None)
            if v is not None:
                d[k] = torch.zeros_like(v)
        c.append([torch.zeros_like(t[0]), d])
    return c


def _scale_to_megapixels(image, mp=1.0, multiple=16):
    samples = image[:1].movedim(-1, 1)
    h, w = samples.shape[2], samples.shape[3]
    s = math.sqrt(mp * 1024 * 1024 / (w * h))
    nw = max(multiple, round(w * s / multiple) * multiple)
    nh = max(multiple, round(h * s / multiple) * multiple)
    return comfy.utils.common_upscale(samples, nw, nh, "lanczos", "disabled").movedim(1, -1)


def _node_output(res):
    """Normalise V1 tuples / V3 NodeOutput."""
    if hasattr(res, "args"):
        return res.args
    if isinstance(res, dict) and "result" in res:
        return res["result"]
    return res


def _qwen21_conditioning(clip, vae, prompt, negative, ref_image, resolution=1024):
    cls = nodes.NODE_CLASS_MAPPINGS.get("TextEncodeQwenImage21")
    images = {"image_1": ref_image} if ref_image is not None else {}
    if cls is not None:
        try:
            out = _node_output(cls.execute(clip=clip, prompt=prompt, negative_prompt=negative, vae=vae,
                                           resolution=resolution, images=images))
            return out[0], out[1], out[2]
        except Exception as e:
            log.warning("[SatoDive] native Qwen 2.1 encoder failed (%s), falling back to plain text encode", e)
    pos = _encode(clip, prompt)
    neg = _encode(clip, negative) if negative.strip() else _zero_out(pos)
    return pos, neg, None


def build_conditioning(pipe, clip, vae):
    preset = A.get_preset(pipe["arch"])
    prompt = pipe["prompt"]
    negative = pipe.get("negative", "") or ""
    ref = pipe.get("reference_image")
    edit = preset.get("edit")
    ref_latent_dict = None

    if edit == "qwen21":
        pos, neg, lat = _qwen21_conditioning(clip, vae, prompt, negative, ref)
        if ref is not None and lat is not None and pipe.get("match_reference_size", True):
            ref_latent_dict = lat
        return pos, neg, ref_latent_dict

    pos = _encode(clip, prompt)
    neg = _encode(clip, negative) if negative.strip() else _zero_out(pos)
    if ref is not None and edit == "reference_latents" and vae is not None:
        scaled = _scale_to_megapixels(ref, 1.0, 16)
        lat = vae.encode(scaled[:, :, :, :3])
        pos = node_helpers.conditioning_set_values(pos, {"reference_latents": [lat]}, append=True)
        neg = node_helpers.conditioning_set_values(neg, {"reference_latents": [lat]}, append=True)
        if pipe.get("match_reference_size", True):
            ref_latent_dict = {"size": (scaled.shape[2], scaled.shape[1])}
    return pos, neg, ref_latent_dict


# --------------------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------------------

def _flux2_sigmas(steps, width, height):
    seq_len = round(width * height / (16 * 16))
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if seq_len > 4300:
        mu = a2 * seq_len + b2
    else:
        m_200 = a2 * seq_len + b2
        m_10 = a1 * seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * steps + b
    t = torch.linspace(1, 0, steps + 1)
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1))


def apply_shift(model, shift):
    if not shift or shift <= 0:
        return model
    try:
        ms = model.get_model_object("model_sampling")
        new = ms.__class__(model.model.model_config)
        if hasattr(ms, "multiplier"):
            new.set_parameters(shift=shift, multiplier=ms.multiplier)
        else:
            new.set_parameters(shift=shift)
        m = model.clone()
        m.add_object_patch("model_sampling", new)
        return m
    except Exception as e:
        log.warning("[SatoDive] could not apply shift %.3f: %s", shift, e)
        return model


def empty_latent(model, width, height):
    fmt = model.get_model_object("latent_format")
    r = int(getattr(fmt, "spacial_downscale_ratio", 8) or 8)
    ch = int(getattr(fmt, "latent_channels", 4))
    lat = torch.zeros([1, ch, max(1, height // r), max(1, width // r)],
                      device=comfy.model_management.intermediate_device())
    return comfy.sample.fix_empty_latent_channels(model, lat)


def render(pipe, model, clip, seed_offset=0):
    """Render one preview image (1,H,W,3) with the given (already LoRA-patched) model + clip."""
    vae = pipe.get("vae")
    if clip is None or vae is None:
        raise RuntimeError("SatoDive preview needs CLIP and VAE connected to the Studio Setup node.")

    steps = int(pipe["steps"])
    cfg = float(pipe["cfg"])
    width, height = int(pipe["width"]), int(pipe["height"])
    seed = int(pipe["seed"]) + seed_offset

    pos, neg, ref_lat = build_conditioning(pipe, clip, vae)
    m = apply_shift(model, float(pipe.get("shift", 0.0)))

    if ref_lat is not None and "samples" in ref_lat:
        latent = comfy.sample.fix_empty_latent_channels(m, ref_lat["samples"])
    else:
        if ref_lat is not None and "size" in ref_lat:
            width, height = ref_lat["size"]
        latent = empty_latent(m, width, height)

    sched = pipe["scheduler"]
    if sched == "flux2":
        sigmas = _flux2_sigmas(steps, width, height)
    else:
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), sched, steps).cpu()

    noise = comfy.sample.prepare_noise(latent, seed, None)
    sampler = comfy.samplers.sampler_object(pipe["sampler"])
    callback = latent_preview.prepare_callback(m, len(sigmas) - 1)
    samples = comfy.sample.sample_custom(m, noise, cfg, sampler, sigmas, pos, neg, latent,
                                         noise_mask=None, callback=callback,
                                         disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]
    images = vae.decode(samples)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images[:1].float().clamp(0, 1).cpu()


# --------------------------------------------------------------------------------------
# Labelled comparison strips
# --------------------------------------------------------------------------------------

_TAG_COLORS = {
    "BASE": (148, 163, 184), "A": (34, 211, 238), "B": (244, 114, 182), "MERGE": (167, 139, 250),
}


def label_image(img, text, color_key=None):
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except Exception:
        return img
    arr = (img[0].numpy() * 255).astype("uint8")
    pil = Image.fromarray(arr)
    d = ImageDraw.Draw(pil, "RGBA")
    size = max(14, pil.width // 40)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    pad = size // 2
    try:
        tw, th = d.textbbox((0, 0), text, font=font)[2:]
    except Exception:
        tw, th = len(text) * size // 2, size
    col = _TAG_COLORS.get(color_key or text, (167, 139, 250))
    d.rounded_rectangle([pad, pad, pad + tw + pad * 2, pad + th + pad * 1.4], radius=pad,
                        fill=(15, 17, 26, 200), outline=col + (255,), width=max(2, size // 8))
    d.text((pad * 2, pad * 1.2), text, fill=col + (255,), font=font)
    out = torch.from_numpy(np.array(pil).astype("float32") / 255.0)[None]
    return out


def hstack(images):
    h = min(i.shape[1] for i in images)
    fixed = []
    for i in images:
        if i.shape[1] != h:
            w = round(i.shape[2] * h / i.shape[1])
            i = comfy.utils.common_upscale(i.movedim(-1, 1), w, h, "bilinear", "disabled").movedim(1, -1)
        fixed.append(i)
    gap = torch.full((1, h, max(4, h // 128), 3), 0.06)
    parts = []
    for n, i in enumerate(fixed):
        if n:
            parts.append(gap)
        parts.append(i)
    return torch.cat(parts, dim=2)


def blank(w=64, h=64):
    return torch.zeros((1, h, w, 3))
