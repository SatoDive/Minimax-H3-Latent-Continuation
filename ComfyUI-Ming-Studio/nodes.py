"""ComfyUI-Ming-Studio nodes.

Model loading is left to ComfyUI's native nodes (Load Diffusion Model, Load CLIP,
Load VAE). These nodes add the Ming-Image specific glue: resolution buckets,
structured design prompts, layer-decomposition conditioning, per-frame RGBA
layer decoding, an interactive layer studio, PSD export and website capture.
"""

import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers

from .ming_studio import buckets, design, psd_writer, website

CATEGORY = "Ming Studio"
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "ming_studio", "prompts")

ASPECTS = {
    "1:1 Square": (1, 1),
    "16:9 Desktop / Slide": (9, 16),
    "9:16 Mobile / Story": (16, 9),
    "4:3 Landscape": (3, 4),
    "3:4 Portrait": (4, 3),
    "3:2 Photo": (2, 3),
    "2:3 Poster": (3, 2),
    "4:5 Social post": (5, 4),
    "1:2 Landing page": (2, 1),
    "1:3 Long web page": (3, 1),
    "1:4 Full web page": (4, 1),
    "2:1 Banner": (1, 2),
    "4:1 Wide banner": (1, 4),
}


def _read_prompt(name):
    with open(os.path.join(PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read().strip()


def _resize(image, width, height, method="bilinear"):
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)


def _empty_latent(frames, height, width, batch=1):
    return torch.zeros([batch, 16, frames, height // 8, width // 8],
                       device=comfy.model_management.intermediate_device())


def _zero_negative(conditioning):
    """Mirror the reference pipeline: CFG negative = zeroed text features, same reference frames."""
    out = []
    for t in conditioning:
        d = t[1].copy()
        for k in ("pooled_output", "direct_context"):
            if isinstance(d.get(k), torch.Tensor):
                d[k] = torch.zeros_like(d[k])
        out.append([torch.zeros_like(t[0]), d])
    return out


def _ensure_rgba(img):
    if img.shape[-1] == 4:
        return img
    if img.shape[-1] == 3:
        return torch.cat([img, torch.ones_like(img[..., :1])], dim=-1)
    if img.shape[-1] == 1:
        return torch.cat([img.repeat(1, 1, 1, 3), torch.ones_like(img)], dim=-1)
    return img[..., :4]


def _to_pil(t):
    arr = (t.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")


def _pil_to_tensor(img):
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).unsqueeze(0)


# --------------------------------------------------------------------------- canvas

class MingCanvasSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "aspect": (list(ASPECTS.keys()) + ["Custom (snap to bucket)", "Custom (exact)"], {"default": "1:1 Square"}),
            "bucket": ([1024, 2048], {"default": 2048, "tooltip": "Ming native output buckets. 2048 is recommended for full layouts, 1024 is ~4x faster."}),
            "custom_width": ("INT", {"default": 1440, "min": 256, "max": 4096, "step": 16}),
            "custom_height": ("INT", {"default": 2560, "min": 256, "max": 4096, "step": 16}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Empty Ming-Image latent on the official resolution buckets (aspect 1:4 to 4:1)."

    def run(self, aspect, bucket, custom_width, custom_height, batch_size):
        if aspect == "Custom (exact)":
            w, h = custom_width // 16 * 16, custom_height // 16 * 16
        elif aspect == "Custom (snap to bucket)":
            h, w = buckets.closest_size(custom_height, custom_width, bucket)
        else:
            rh, rw = ASPECTS[aspect]
            h, w = buckets.closest_size(rh, rw, bucket)
        return ({"samples": _empty_latent(1, h, w, batch_size)}, w, h)


# --------------------------------------------------------------------------- prompts

class MingDesignPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "design_type": (list(design.TYPES.keys()), {"default": "Landing page (web)"}),
                "style": (list(design.STYLES.keys()), {"default": "Modern minimal"}),
                "brief": ("STRING", {"multiline": True, "default": "A landing page for an AI-powered note taking app called Nimbus."}),
                "brand_name": ("STRING", {"default": "Nimbus"}),
                "headline": ("STRING", {"default": "Think faster. Write less."}),
                "subheadline": ("STRING", {"default": "Your notes, organised by AI the moment you write them."}),
                "cta_text": ("STRING", {"default": "Start free"}),
                "items": ("STRING", {"multiline": True, "default": "Smart summaries\nInstant search\nTeam spaces", "tooltip": "One rendered text per line (feature cards, bullet points, menu items...)."}),
                "main_visual": ("STRING", {"multiline": True, "default": "A floating app window mockup showing a tidy note with AI highlights, soft glow behind it"}),
                "palette": ("STRING", {"default": "auto", "tooltip": "'auto' uses the style palette, or list hex colors: #0F172A #6366F1 ..."}),
                "width": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16}),
                "transparent_background": ("BOOLEAN", {"default": False, "tooltip": "Adds the official RGBA trigger phrase so the VAE writes real alpha."}),
            },
            "optional": {
                "custom_layers": ("STRING", {"multiline": True, "default": "", "tooltip": "Extra layers, one per line:  description | cx, cy, w, h | #hex #hex   (or one JSON layer object per line)."}),
                "extra_style": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Builds Ming's native Figma-style JSON caption (canvas_settings + layers) from simple fields."

    def run(self, design_type, style, brief, brand_name, headline, subheadline, cta_text, items, main_visual,
            palette, width, height, transparent_background, custom_layers="", extra_style=""):
        text = design.build(design_type, style, brief, brand_name, headline, subheadline, cta_text, items,
                            main_visual, "" if palette.strip().lower() == "auto" else palette, width, height,
                            transparent_background, custom_layers, extra_style)
        return (text,)


class MingTransparentTrigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "A glossy 3D chili pepper mascot with sunglasses"}),
            "trigger": (design.TRANSPARENT_TRIGGERS, {"default": design.TRANSPARENT_TRIGGERS[0]}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Appends one of the official phrases that make Ming-Image output a real alpha channel."

    def run(self, prompt, trigger):
        return (prompt.rstrip().rstrip(".") + ". " + trigger,)


class MingEnhancerPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "task": (["text-to-image rewriter", "layer-decompose rewriter"],),
            "request": ("STRING", {"multiline": True, "default": "A poster for a summer jazz festival in Lisbon, 12-14 July."}),
            "width": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Official Ming prompt-enhancer instructions. Feed into ComfyUI's native 'Generate Text' node "
                   "(e.g. Qwen3.8-27B) and clean the answer with 'Ming Prompt Cleanup'.")

    def run(self, task, request, width, height):
        if task == "text-to-image rewriter":
            sys_prompt = _read_prompt("t2i_rewriter_system_prompt.txt")
            prompt = "%s\n\nCanvas aspect_ratio: %s" % (request.strip(), design.aspect_string(width, height))
            return (prompt, sys_prompt)
        guided = _read_prompt("layer_rewriter_prompt.txt")
        return (guided.replace("{spec}", request.strip()), "")


class MingPromptCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True})},
                "optional": {"append_transparency": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "num_layers")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Strips <think> blocks / code fences from an LLM answer, re-formats JSON and reports the layer count."

    def run(self, text, append_transparency=False):
        t = design.clean_llm_output(text)
        if append_transparency:
            t += "\n" + design.TRANSPARENT_TRIGGERS[0]
        n = design.parse_num_layers(t, default=0)
        if n == 0:
            try:
                n = len(json.loads(t).get("layers", []))
            except (ValueError, AttributeError):
                n = 0
        return (t, n)


# --------------------------------------------------------------------------- conditioning

class MingLayerDecomposeEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "num_layers": ("INT", {"default": 5, "min": 1, "max": 16, "tooltip": "Used when no layer plan is given."}),
                "resolution": ([1024, 512], {"default": 1024, "tooltip": "Official working buckets for Design-Layer (512 is faster)."}),
                "layer_plan": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional: one line per layer, FRONT-most first, background last. The line count becomes the layer count."}),
            },
            "optional": {
                "prompt_override": ("STRING", {"forceInput": True, "tooltip": "Full prompt (e.g. from the enhancer). The layer count is parsed from it."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "image", "prompt", "num_layers", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Conditioning for Ming-Image-0.1-Design-Layer: the image is seen by the text encoder and appended as a clean "
                   "reference frame; the latent holds composite + N layer frames. Sample with CFG 2.0, 12 steps.")

    def run(self, clip, vae, image, num_layers, resolution, layer_plan, prompt_override=None):
        if prompt_override and prompt_override.strip():
            prompt = prompt_override.strip()
            n = design.parse_num_layers(prompt, default=num_layers)
        else:
            prompt, n = design.decompose_prompt(num_layers, layer_plan.splitlines())
        src = image[:1, :, :, :3]
        h, w = buckets.closest_size(src.shape[1], src.shape[2], resolution)
        ref = _resize(src, w, h)
        tokens = clip.tokenize(prompt, images=[ref])
        positive = clip.encode_from_tokens_scheduled(tokens)
        ref_latent = vae.encode(ref)
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = _zero_negative(positive)
        latent = {"samples": _empty_latent(n + 1, h, w)}
        return (positive, negative, latent, ref, prompt, n, w, h)


class MingEditEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": "Restyle this design as a dark-mode version with neon purple accents. Keep every text unchanged."}),
                "size": (["match image (snap /16)", "bucket 1024", "bucket 2048"], {"default": "bucket 1024"}),
            },
            "optional": {
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Ming-Image Design editing / multi-reference composition (same logic as the native edit encoder, plus sizing)."

    def run(self, clip, vae, prompt, size, image_1=None, image_2=None, image_3=None, image_4=None):
        images = [i[:1, :, :, :3] for i in (image_1, image_2, image_3, image_4) if i is not None]
        if images:
            H, W = images[0].shape[1:3]
            if size == "match image (snap /16)":
                h, w = max(16, H // 16 * 16), max(16, W // 16 * 16)
            else:
                h, w = buckets.closest_size(H, W, 2048 if "2048" in size else 1024)
            images = [_resize(i, w, h) for i in images]
        else:
            h, w = (1024, 1024) if "2048" not in size else (2048, 2048)
        tokens = clip.tokenize(prompt, images=images)
        positive = clip.encode_from_tokens_scheduled(tokens)
        if images:
            refs = [vae.encode(i) for i in images]
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": refs}, append=True)
        return (positive, _zero_negative(positive), {"samples": _empty_latent(1, h, w)}, w, h)


# --------------------------------------------------------------------------- decoding

class MingDecodeLayers:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT")
    RETURN_NAMES = ("composite", "layers", "layer_masks", "count")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Decodes each Ming latent frame on its own (composite, then layer 1 = front-most ... background) "
                   "into RGBA images. Works for Design (single frame) and Design-Layer outputs.")

    def run(self, samples, vae):
        lat = samples["samples"]
        if getattr(lat, "is_nested", False):
            lat = lat.unbind()[0]
        if lat.ndim == 4:
            lat = lat.unsqueeze(2)
        frames = []
        pbar = comfy.utils.ProgressBar(lat.shape[2])
        for f in range(lat.shape[2]):
            img = vae.decode(lat[:, :, f:f + 1])
            if img.ndim == 5:
                img = img.reshape(-1, *img.shape[-3:])
            frames.append(_ensure_rgba(img.float()))
            pbar.update(1)
        composite = frames[0]
        layers = torch.cat(frames[1:], dim=0) if len(frames) > 1 else frames[0]
        return (composite, layers, layers[..., 3].clone(), layers.shape[0])


# --------------------------------------------------------------------------- studio / export

def _default_state(n, names=None):
    names = names or []
    return {"layers": [{"index": i, "name": names[i] if i < len(names) and names[i] else ("Background" if i == n - 1 and n > 1 else "Layer %d" % (i + 1)),
                        "visible": True, "opacity": 1.0, "blend": "normal"} for i in range(n)]}


def _parse_state(state_json, n, names=None):
    try:
        st = json.loads(state_json) if state_json else {}
    except ValueError:
        st = {}
    layers = st.get("layers") if isinstance(st, dict) else None
    if not layers or len(layers) != n or sorted(int(l.get("index", -1)) for l in layers) != list(range(n)):
        return _default_state(n, names)
    return {"layers": layers}


def _composite_tensor(layers_rgba, state):
    """layers_rgba: [N,H,W,4] top-first; returns [1,H,W,4] normal-blended composite of visible layers."""
    n, H, W, _ = layers_rgba.shape
    acc = torch.zeros((H, W, 3))
    acc_a = torch.zeros((H, W, 1))
    for l in reversed(state["layers"]):
        if not l.get("visible", True):
            continue
        px = layers_rgba[int(l["index"])].float().cpu()
        a = px[..., 3:4] * float(l.get("opacity", 1.0))
        acc = px[..., :3] * a + acc * (1 - a)
        acc_a = a + acc_a * (1 - a)
    rgb = torch.where(acc_a > 1e-6, acc / acc_a.clamp(min=1e-6), torch.zeros_like(acc))
    return torch.cat([rgb, acc_a], dim=-1).unsqueeze(0)


def _names_from_text(text):
    return [n.strip() for n in (text or "").splitlines()]


def plan_names_from_prompt(prompt):
    names = []
    for line in (prompt or "").splitlines():
        line = line.strip()
        if line.lower().startswith("layer ") and ":" in line:
            names.append(line.split(":", 1)[1].strip()[:48])
    return names


class MingLayerStudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE", {"tooltip": "RGBA layer batch, front-most first (from Ming Decode Layers)."}),
                "studio_state": ("STRING", {"default": "", "multiline": False, "tooltip": "Managed by the studio UI."}),
            },
            "optional": {
                "composite": ("IMAGE",),
                "layer_names": ("STRING", {"forceInput": True, "tooltip": "One name per line, or a Ming decompose prompt (Layer 1: ...)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("composite", "visible_layers", "mask", "studio_state")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Interactive layer studio: preview, pick, hide, solo, rename, reorder, set opacity/blend, and export PSD/PNG from the node."

    def run(self, layers, studio_state, composite=None, layer_names=None):
        layers = _ensure_rgba(layers)
        n = layers.shape[0]
        names = plan_names_from_prompt(layer_names) or _names_from_text(layer_names)
        state = _parse_state(studio_state, n, names)

        temp_dir = folder_paths.get_temp_directory()
        sub = "ming_studio"
        os.makedirs(os.path.join(temp_dir, sub), exist_ok=True)
        token = "%08x" % random.getrandbits(32)
        files = []
        for i in range(n):
            fn = "layer_%s_%02d.png" % (token, i)
            _to_pil(layers[i]).save(os.path.join(temp_dir, sub, fn), compress_level=1)
            files.append({"filename": fn, "subfolder": sub, "type": "temp"})
        comp_file = None
        if composite is not None:
            fn = "composite_%s.png" % token
            _to_pil(_ensure_rgba(composite)[0]).save(os.path.join(temp_dir, sub, fn), compress_level=1)
            comp_file = {"filename": fn, "subfolder": sub, "type": "temp"}

        out = _composite_tensor(layers, state)
        visible = [int(l["index"]) for l in state["layers"] if l.get("visible", True)]
        vis_layers = layers[visible] if visible else layers[:1] * 0
        ui = {"ming_layers": files, "ming_state": [state], "ming_size": [layers.shape[2], layers.shape[1]]}
        if comp_file:
            ui["ming_composite"] = [comp_file]
        return {"ui": ui, "result": (out, vis_layers, out[..., 3], json.dumps(state))}


class MingSavePSD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE", {"tooltip": "RGBA layer batch, front-most first."}),
                "filename_prefix": ("STRING", {"default": "ming_studio/design"}),
                "crop_layers_to_content": ("BOOLEAN", {"default": True}),
                "also_save_pngs": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "composite": ("IMAGE", {"tooltip": "Optional flattened image, added as a hidden reference layer."}),
                "layer_names": ("STRING", {"forceInput": True}),
                "studio_state": ("STRING", {"forceInput": True, "tooltip": "From Ming Layer Studio: keeps order, names, visibility, opacity and blend modes."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("psd_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Writes a real layered Photoshop .psd (named, transparent layers) plus optional per-layer PNGs."

    def run(self, layers, filename_prefix, crop_layers_to_content, also_save_pngs, composite=None, layer_names=None, studio_state=None):
        layers = _ensure_rgba(layers)
        n, H, W, _ = layers.shape
        names = plan_names_from_prompt(layer_names) or _names_from_text(layer_names)
        state = _parse_state(studio_state or "", n, names)
        out_dir = folder_paths.get_output_directory()
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, W, H)
        base = "%s_%05d_" % (filename, counter)

        psd_layers = []
        for l in state["layers"]:
            psd_layers.append({"image": layers[int(l["index"])].cpu().numpy(), "name": l.get("name") or "Layer",
                               "visible": l.get("visible", True), "opacity": float(l.get("opacity", 1.0)),
                               "blend": l.get("blend", "normal")})
        merged = _composite_tensor(layers, state)[0].numpy()
        if composite is not None:
            psd_layers.append({"image": _ensure_rgba(composite)[0].cpu().numpy(), "name": "Original (reference)", "visible": False})
        psd_path = os.path.join(full_folder, base + ".psd")
        psd_writer.write_psd(psd_path, psd_layers, W, H, composite=merged, crop=crop_layers_to_content)

        previews = []
        prev_name = base + "preview.png"
        Image.fromarray((np.clip(merged, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(os.path.join(full_folder, prev_name))
        previews.append({"filename": prev_name, "subfolder": subfolder, "type": "output"})
        if also_save_pngs:
            for i, l in enumerate(state["layers"]):
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (l.get("name") or "layer"))[:40]
                _to_pil(layers[int(l["index"])]).save(os.path.join(full_folder, "%slayer_%02d_%s.png" % (base, i + 1, safe)))
        psd_info = {"filename": base + ".psd", "subfolder": subfolder, "type": "output"}
        return {"ui": {"images": previews, "ming_psd": [psd_info]}, "result": (psd_path,)}


# --------------------------------------------------------------------------- RGBA utils

class MingRGBAComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "background": (["checkerboard", "white", "black", "custom"], {"default": "white"}),
            "custom_color": ("STRING", {"default": "#1E1E2E"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgb", "alpha")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Flattens an RGBA image over a background and returns its alpha as a mask (for nodes that expect RGB)."

    def run(self, image, background, custom_color):
        image = _ensure_rgba(image)
        rgb, a = image[..., :3], image[..., 3:4]
        B, H, W, _ = image.shape
        if background == "checkerboard":
            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            chk = (((yy // 16) + (xx // 16)) % 2).float() * 0.12 + 0.82
            bg = chk[None, ..., None].expand(B, H, W, 3)
        else:
            hexc = {"white": "#FFFFFF", "black": "#000000"}.get(background, custom_color).lstrip("#")
            try:
                col = [int(hexc[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
            except ValueError:
                col = [1.0, 1.0, 1.0]
            bg = torch.tensor(col).view(1, 1, 1, 3).expand(B, H, W, 3)
        bg = bg.to(rgb.device, rgb.dtype)
        return (rgb * a + bg * (1 - a), a[..., 0])


class MingJoinAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "alpha": ("MASK",)},
                "optional": {"invert": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Builds an RGBA image from an RGB image and a mask (e.g. to send your own cut-outs to the studio / PSD)."

    def run(self, image, alpha, invert=False):
        a = alpha if alpha.ndim == 3 else alpha.unsqueeze(0)
        if invert:
            a = 1.0 - a
        if a.shape[1:] != image.shape[1:3]:
            a = torch.nn.functional.interpolate(a.unsqueeze(1), size=image.shape[1:3], mode="bilinear").squeeze(1)
        a = a.expand(image.shape[0], -1, -1) if a.shape[0] == 1 else a
        return (torch.cat([image[..., :3], a.unsqueeze(-1).to(image)], dim=-1),)


# --------------------------------------------------------------------------- website

def _palette_image(colors, width=1024, height=160):
    img = Image.new("RGB", (width, height), "#111111")
    if not colors:
        return img
    d = ImageDraw.Draw(img)
    sw = width / len(colors)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, c in enumerate(colors):
        d.rectangle([int(i * sw), 0, int((i + 1) * sw), height], fill=c)
        lum = website.luminance(c)
        d.text((int(i * sw) + 10, height - 30), c, fill="#000000" if lum > 0.5 else "#FFFFFF", font=font)
    return img


def _wireframe_image(info, width, height):
    """Fallback when no browser is available: a schematic render of the extracted page."""
    pal = info["palette"] or ["#FFFFFF", "#111827", "#2563EB"]
    bg, fg = pal[0], (pal[1] if len(pal) > 1 else "#111111")
    acc = pal[2] if len(pal) > 2 else "#2563EB"
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)

    def font(size):
        for f in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                continue
        return ImageFont.load_default()

    nav_h = 72
    d.text((40, 24), info["site_name"] or "Brand", fill=fg, font=font(26))
    x = width - 40
    for item in reversed(info["nav"][:6]):
        f = font(18)
        x -= int(d.textlength(item, font=f)) + 32
        d.text((x, 30), item, fill=fg, font=f)
    d.line([(0, nav_h), (width, nav_h)], fill=acc, width=1)
    y = nav_h + 90
    h1 = info["h1"] or info["title"]
    words, line, lines = h1.split(), "", []
    f = font(56)
    for w_ in words:
        t = (line + " " + w_).strip()
        if d.textlength(t, font=f) > width * 0.55:
            lines.append(line)
            line = w_
        else:
            line = t
    lines.append(line)
    for l in lines[:3]:
        d.text((60, y), l, fill=fg, font=f)
        y += 70
    if info["description"]:
        f = font(20)
        line, y = "", y + 10
        for w_ in info["description"][:180].split():
            t = (line + " " + w_).strip()
            if d.textlength(t, font=f) > width * 0.52:
                d.text((60, y), line, fill=fg, font=f)
                y += 28
                line = w_
            else:
                line = t
        d.text((60, y), line, fill=fg, font=f)
        y += 50
    bx = 60
    for b in info["buttons"][1:3] or info["buttons"][:1]:
        f = font(20)
        tw = int(d.textlength(b, font=f))
        d.rounded_rectangle([bx, y + 20, bx + tw + 48, y + 72], radius=26, fill=acc)
        d.text((bx + 24, y + 34), b, fill="#FFFFFF", font=f)
        bx += tw + 72
    d.rounded_rectangle([int(width * .62), nav_h + 70, width - 60, nav_h + 470], radius=24, outline=acc, width=3)
    y = max(y + 140, nav_h + 540)
    for s in info["sections"]:
        if y > height - 200 or not s["headings"] or s["headings"][0] == h1:
            continue
        d.text((60, y), s["headings"][0][:70], fill=fg, font=font(34))
        y += 60
        cards = s["headings"][1:4]
        if cards:
            cw = (width - 120 - 40 * (len(cards) - 1)) // len(cards)
            for i, c in enumerate(cards):
                x0 = 60 + i * (cw + 40)
                d.rounded_rectangle([x0, y, x0 + cw, y + 180], radius=18, outline=fg, width=2)
                d.text((x0 + 20, y + 20), c[:28], fill=fg, font=font(22))
            y += 220
        y += 40
    d.rectangle([0, height - 90, width, height], fill=fg)
    d.text((40, height - 60), "© %s" % (info["site_name"] or "Brand"), fill=bg, font=font(18))
    return img


class MingWebsiteDesign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com"}),
                "mode": (list(website.MODES.keys()), {"default": "faithful clone"}),
                "width": ("INT", {"default": 1408, "min": 256, "max": 4096, "step": 16, "tooltip": "Output canvas width (connect from Ming Canvas Size)."}),
                "height": ("INT", {"default": 2816, "min": 256, "max": 4096, "step": 16}),
                "take_screenshot": ("BOOLEAN", {"default": True, "tooltip": "Uses Playwright if installed, else a local Chrome/Edge/Chromium in headless mode."}),
                "screenshot_width": ("INT", {"default": 1440, "min": 320, "max": 3840, "step": 8}),
                "screenshot_height": ("INT", {"default": 2560, "min": 320, "max": 8192, "step": 8}),
                "decompose_layers": ("INT", {"default": 6, "min": 2, "max": 12}),
            },
            "optional": {
                "pasted_html": ("STRING", {"multiline": True, "default": "", "tooltip": "Paste page HTML here instead of fetching (for logged-in or blocked pages)."}),
                "extra_style": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("design_prompt", "decompose_prompt", "screenshot", "palette", "summary")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Fetches a website, extracts its structure, palette and fonts, and writes Ming prompts to recreate, restyle or layer it."

    def run(self, url, mode, width, height, take_screenshot, screenshot_width, screenshot_height, decompose_layers,
            pasted_html="", extra_style=""):
        notes = []
        try:
            info = website.analyze(url=url.strip(), html=pasted_html or "")
        except Exception as e:  # noqa: BLE001
            notes.append("Fetch failed (%s). Paste the HTML into 'pasted_html' to continue." % e)
            info = website.analyze(html="<title>%s</title>" % url)
        design_prompt = website.build_design_prompt(info, width, height, mode, extra_style)
        plan = website.build_decompose_plan(info, decompose_layers)
        decompose, _ = design.decompose_prompt(decompose_layers, plan)

        shot = None
        if take_screenshot and url.strip() and not pasted_html:
            try:
                shot = website.screenshot(url.strip(), screenshot_width, screenshot_height)
            except Exception as e:  # noqa: BLE001
                notes.append("Screenshot unavailable (%s); using a schematic wireframe instead. "
                             "Install Playwright (pip install playwright && playwright install chromium) or Chrome." % e)
        if shot is None:
            shot = _wireframe_image(info, screenshot_width, screenshot_height)

        summary = {
            "url": info["url"], "title": info["title"], "site": info["site_name"], "palette": info["palette"],
            "fonts": info["fonts"], "nav": info["nav"], "buttons": info["buttons"], "headings": info["headings"][:8],
            "dark": info["dark"], "notes": notes,
        }
        summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
        ui = {"ming_site": [summary]}
        return {"ui": ui, "result": (design_prompt, decompose, _pil_to_tensor(shot), _pil_to_tensor(_palette_image(info["palette"])), summary_text)}


NODE_CLASS_MAPPINGS = {
    "MingCanvasSize": MingCanvasSize,
    "MingDesignPrompt": MingDesignPrompt,
    "MingTransparentTrigger": MingTransparentTrigger,
    "MingEnhancerPrompts": MingEnhancerPrompts,
    "MingPromptCleanup": MingPromptCleanup,
    "MingLayerDecomposeEncode": MingLayerDecomposeEncode,
    "MingEditEncode": MingEditEncode,
    "MingDecodeLayers": MingDecodeLayers,
    "MingLayerStudio": MingLayerStudio,
    "MingSavePSD": MingSavePSD,
    "MingRGBAComposite": MingRGBAComposite,
    "MingJoinAlpha": MingJoinAlpha,
    "MingWebsiteDesign": MingWebsiteDesign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MingCanvasSize": "◆ Ming Canvas Size",
    "MingDesignPrompt": "◆ Ming Design Prompt Builder",
    "MingTransparentTrigger": "◆ Ming Transparent (RGBA) Trigger",
    "MingEnhancerPrompts": "◆ Ming Prompt Enhancer Instructions",
    "MingPromptCleanup": "◆ Ming Prompt Cleanup",
    "MingLayerDecomposeEncode": "◆ Ming Layer Decompose Encode",
    "MingEditEncode": "◆ Ming Edit / Reference Encode",
    "MingDecodeLayers": "◆ Ming Decode Layers (RGBA)",
    "MingLayerStudio": "◆ Ming Layer Studio",
    "MingSavePSD": "◆ Ming Save PSD (Photoshop)",
    "MingRGBAComposite": "◆ Ming RGBA → RGB + Alpha",
    "MingJoinAlpha": "◆ Ming RGB + Mask → RGBA",
    "MingWebsiteDesign": "◆ Ming Website → Design",
}
