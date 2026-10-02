"""Nodes: ① LoRA Studio Setup, ② LoRA Slot (one per LoRA), ③ LoRA Merge Studio (2-10 LoRAs)."""

import json
import logging
import os

import torch

import folder_paths
import nodes
from comfy_api.latest import io

from . import sato_arch as A
from . import sato_help as H
from . import sato_lora_core as core
from . import sato_preview as P

log = logging.getLogger("SatoDive.LoRAMerge")

CATEGORY = "SatoDive/LoRA Merge"
MAX_RES = nodes.MAX_RESOLUTION
LETTERS = P.LETTERS

PRECISIONS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
KEY_STYLES = ["comfy (diffusion_model.*)", "kohya (lora_unet_*)"]

SatoLoraType = io.Custom("SATO_LORA")
SatoPipeType = io.Custom("SATO_PIPE")


def _preview_ui(images, prefix):
    return nodes.PreviewImage().save_images(images, filename_prefix=prefix)["ui"]


def _resolve_group_arch(arch, model):
    if arch != A.AUTO:
        return arch
    try:
        cls = type(model.model).__name__
    except Exception:
        return A.AUTO
    for name, p in A.ARCH_PRESETS.items():
        if cls in p.get("model_classes", []):
            return name
    return A.AUTO


# ======================================================================================
# 1) Studio Setup
# ======================================================================================

class SatoDiveStudioSetup:
    DESCRIPTION = ("Connect your native UNET / CLIP / VAE loaders here and pick the architecture. "
                   "Changing the architecture fills in recommended sampling settings (you can still edit them).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Connect the native 'Load Diffusion Model' node here."}),
                "architecture": (A.ARCH_NAMES, {"default": "Krea2 (Turbo)",
                                  "tooltip": "Which model family you loaded. Changing it fills in the recommended steps / CFG / sampler below."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "default": "cinematic portrait of a woman in a neon-lit rainy street, 35mm photo, detailed skin"}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "",
                                        "tooltip": "Leave empty for distilled/turbo models (CFG 1): the negative is zeroed out."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed",
                                 "tooltip": "Fixed by default so tweaking the merge does not re-render the LoRA slot previews."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200, "tooltip": "Preview sampling steps (filled in from the architecture)."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "1.0 for turbo / distilled models. Higher only for base models."}),
                "sampler": (P.SAMPLERS, {"default": "euler"}),
                "scheduler": (P.SCHEDULERS, {"default": "simple"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": MAX_RES, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": MAX_RES, "step": 16}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01,
                                    "tooltip": "Sampling shift override. 0 = keep the model's native value."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "From the native Load CLIP node (set its type for the architecture)."}),
                "vae": ("VAE", {"tooltip": "From the native Load VAE node."}),
                "reference_image": ("IMAGE", {"tooltip": "Optional: edit / reference image for Qwen-Image 2.1, Klein and Krea2 previews."}),
                "match_reference_size": ("BOOLEAN", {"default": True,
                                         "tooltip": "Edit models: render the preview at the reference image's size (recommended)."}),
                "isolated_previews": ("BOOLEAN", {"default": True,
                                      "tooltip": "ON (recommended): the model is reloaded before every preview so each LoRA node "
                                                 "shows exactly its own LoRA - previews can never leak into each other. "
                                                 "Turn OFF for faster previews if you have lots of VRAM."}),
            },
        }

    RETURN_TYPES = ("SATO_PIPE", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("pipe", "model", "clip", "vae")
    FUNCTION = "setup"
    CATEGORY = CATEGORY

    def setup(self, model, architecture, prompt, negative, seed, steps, cfg, sampler, scheduler,
              width, height, shift, clip=None, vae=None, reference_image=None, match_reference_size=True,
              isolated_previews=True):
        preset = A.get_preset(architecture)
        ok, cls_name = A.arch_matches_model(preset, model)
        problems = A.check_compat(architecture, model, clip)
        if problems:
            # stop here with a clear message instead of a cryptic shape error deep inside sampling
            raise RuntimeError("LoRA Studio Setup - the loaded files don't match the chosen architecture:\n\n• "
                               + "\n• ".join(problems)
                               + "\n\n(Choose 'Auto-detect / Other' as architecture to skip this check.)")
        pipe = core.SatoPipe(
            model=model, clip=clip, vae=vae, arch=architecture,
            group_arch=_resolve_group_arch(architecture, model),
            prompt=prompt, negative=negative, seed=seed, steps=steps, cfg=cfg, sampler=sampler,
            scheduler=scheduler, width=width, height=height, shift=shift,
            reference_image=reference_image, match_reference_size=match_reference_size,
            isolate=isolated_previews,
        )
        ui = {"sato_setup": [{"arch": architecture, "model_class": cls_name, "warnings": [],
                              "has_clip": clip is not None, "has_vae": vae is not None,
                              "has_reference": reference_image is not None}]}
        return {"ui": ui, "result": (pipe, model, clip, vae)}


# ======================================================================================
# 2) LoRA Slot (classic node so it keeps the familiar LoRA file picker)
# ======================================================================================

def run_slot(pipe, lora_name, strength_model, strength_clip, preview, letter="A"):
    """LoRA Slot logic: load + analyse one LoRA, optionally render its preview."""
    mode = "off" if preview.lower().startswith("off") else ("both" if "base" in preview.lower() else "lora")
    path, sd, meta = core.load_lora_file(lora_name)
    info = dict(core.inspect_lora_file(lora_name, pipe["arch"]))
    model, clip = pipe["model"], pipe.get("clip")

    matched, unmatched = core.count_matches(sd, core.build_key_map(model, clip), core.build_shape_lookup(model, clip))
    info["matched"] = matched
    info["unmatched"] = unmatched
    info["slot"] = letter
    if matched == 0:
        log.warning("[SatoDive] LoRA %s: no keys matched the connected model - wrong architecture?", lora_name)

    m2, c2 = core.apply_lora(model, clip, sd, strength_model, strength_clip if clip is not None else 0.0)
    if clip is None:
        c2 = None
    info["patched"] = P.patched_count(model, m2)
    cp, c2p = getattr(clip, "patcher", None), getattr(c2, "patcher", None)
    info["patched_te"] = P.patched_count(cp, c2p) if (cp is not None and c2p is not None) else 0
    info["strength"] = [strength_model, strength_clip]

    bundle = core.SatoLoRA(lora_name, sd, meta, path=path, arch=pipe["arch"],
                           strength_model=strength_model, strength_clip=strength_clip, info=info)

    img = P.blank()
    ui = {}
    if mode != "off" and matched == 0:
        img = P.label_image(P.message_image("This LoRA does not fit the loaded model", "0 weights matched - it was made for another architecture"), "NO FIT", "BAD")
        ui = _preview_ui(img, "SatoDive_slot{}".format(letter))
    elif mode != "off":
        lora_img = P.render(pipe, m2, c2 if c2 is not None else clip)
        if mode == "both":
            base_img = P.render(pipe, model, clip)
            img = P.hstack([P.label_image(base_img, "BASE"), P.label_image(lora_img, letter)])
        else:
            img = lora_img
        ui = _preview_ui(img, "SatoDive_slot{}".format(letter))
        info["previewed"] = True
    ui["sato_info"] = [info]
    return {"ui": ui, "result": (bundle, m2, c2, img)}


class SatoDiveLoRASlot:
    DESCRIPTION = ("Load ONE LoRA. Shows its info (rank, trigger words, Civitai pictures) and can render a "
                   "preview with just this LoRA. Add one Slot per LoRA and plug them into the Merge Studio.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("SATO_PIPE", {"tooltip": "Connect the 'pipe' output of LoRA Studio Setup."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The LoRA file. Use 🔎 Browse to pick one with pictures."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01,
                                             "tooltip": "How strong this LoRA is on the image model. 1.0 = as trained. Also used by the merge when 'use_slot_strengths' is on."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01,
                                            "tooltip": "Strength on the text encoder (only matters if the LoRA trained it - the card tells you)."}),
                "preview": (H.PREVIEW_SLOT, {"default": H.PREVIEW_SLOT[0],
                                             "tooltip": "LoRA only: render with this LoRA. Before / after: base model next to the LoRA. "
                                                        "Off: only load + analyse (fast)."}),
            },
        }

    RETURN_TYPES = ("SATO_LORA", "MODEL", "CLIP", "IMAGE")
    RETURN_NAMES = ("lora", "model", "clip", "preview")
    OUTPUT_TOOLTIPS = ("Plug into the Merge Studio.", "Model with only this LoRA applied.",
                       "CLIP with only this LoRA applied.", "The preview image.")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, pipe, lora_name, strength_model, strength_clip, preview):
        return run_slot(pipe, lora_name, strength_model, strength_clip, preview, "LoRA")


# ======================================================================================
# 3) Merge Studio (V3 node -> native auto-growing LoRA inputs)
# ======================================================================================

def _parse_mix(text):
    try:
        d = json.loads(text or "{}")
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


class SatoDiveLoRAMergeStudioV2(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        T = H.TIPS
        return io.Schema(
            node_id="SatoDiveLoRAMergeStudioV2",
            display_name="③ LoRA Merge Studio - SatoDive",
            category=CATEGORY,
            description=("Mix 2 to 10 LoRAs into one. Connect LoRA Slots to lora_A, lora_B... (a new input appears every time "
                         "you connect one). Use the mixer and per-block sliders, preview, then save as a new LoRA."),
            search_aliases=["lora merge", "merge lora", "lora mixer", "combine lora", "satodive"],
            is_output_node=True,
            inputs=[
                SatoPipeType.Input("pipe", tooltip="Connect the 'pipe' output of LoRA Studio Setup."),
                io.Autogrow.Input("loras", template=io.Autogrow.TemplateNames(
                    SatoLoraType.Input("lora"), names=["lora_{}".format(c) for c in LETTERS], min=2),
                    tooltip="Connect LoRA Slot nodes. A new input appears each time you connect one (up to 10)."),
                io.Combo.Input("method", options=H.METHODS, default=H.METHODS[0],
                               tooltip="How the LoRAs are combined. The guide card below explains the choice you pick."),
                io.Float.Input("overall_strength", default=1.0, min=0.0, max=3.0, step=0.01, tooltip=T["overall_strength"]),
                io.Boolean.Input("use_slot_strengths", default=True, tooltip=T["use_slot_strengths"]),
                io.Int.Input("output_rank", default=0, min=0, max=1024, tooltip=T["output_rank"]),
                io.Float.Input("keep_ratio", default=0.5, min=0.01, max=1.0, step=0.01, tooltip=T["keep_ratio"]),
                io.Combo.Input("text_encoder", options=H.TE_MODES, default=H.TE_MODES[0], tooltip=T["text_encoder"]),
                io.Combo.Input("preview", options=H.PREVIEW_MERGE, default=H.PREVIEW_MERGE[0], tooltip=T["preview"]),
                io.Float.Input("preview_strength", default=1.0, min=-4.0, max=4.0, step=0.01, tooltip=T["preview_strength"]),
                io.Boolean.Input("save_lora", default=False, label_on="save when run", label_off="don't save", tooltip=T["save_lora"]),
                io.String.Input("filename", default="SatoDive_merge", tooltip=T["filename"]),
                io.Combo.Input("precision", options=list(PRECISIONS.keys()), default="fp16", tooltip=T["precision"], advanced=True),
                io.Combo.Input("key_style", options=KEY_STYLES, default=KEY_STYLES[0], tooltip=T["key_style"], advanced=True),
                io.Int.Input("dare_seed", default=0, min=0, max=0xffffffffffffffff, tooltip=T["dare_seed"], advanced=True,
                             control_after_generate=False),
                io.String.Input("mix_settings", default="{}", tooltip="Managed by the mixer / block panel."),
            ],
            outputs=[
                SatoLoraType.Output(display_name="merged_lora", tooltip="Plug into another Merge Studio to keep mixing."),
                io.Model.Output(display_name="model", tooltip="Model with the merged LoRA (at preview_strength)."),
                io.Clip.Output(display_name="clip", tooltip="CLIP with the merged LoRA (at preview_strength)."),
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="saved_path"),
            ],
        )

    @classmethod
    def execute(cls, pipe, loras, method, overall_strength, use_slot_strengths, output_rank, keep_ratio,
                text_encoder, preview, preview_strength, save_lora, filename, precision="fp16",
                key_style=KEY_STYLES[0], dare_seed=0, mix_settings="{}") -> io.NodeOutput:
        model, clip = pipe["model"], pipe.get("clip")
        preset = A.get_preset(pipe.get("group_arch", pipe["arch"]))

        connected = []
        for c in LETTERS:
            b = (loras or {}).get("lora_{}".format(c))
            if b is not None:
                connected.append((c, b))
        if len(connected) < 1:
            raise RuntimeError("Connect at least two LoRA Slot nodes to the Merge Studio.")

        mix = _parse_mix(mix_settings)
        gains = mix.get("_mix") or {}
        muted = set(mix.get("_mute") or [])
        solo = set(mix.get("_solo") or [])

        key_map = core.build_key_map(model, clip)
        shapes = core.build_shape_lookup(model, clip)
        deltas, stats, wfuns, ranks, used = [], [], [], [], []
        for letter, b in connected:
            d, st = core.extract_deltas(b.sd, key_map, shapes)
            sm = b.strength_model if use_slot_strengths else 1.0
            sc = b.strength_clip if use_slot_strengths else 1.0
            gain = float(gains.get(letter, 1.0))
            if letter in muted or (solo and letter not in solo):
                gain = 0.0
            blocks = {k: float(v) for k, v in (mix.get(letter) or {}).items()}
            wfuns.append(lambda g, blocks=blocks, sm=sm, sc=sc, gain=gain: gain * blocks.get(g, 1.0) * (sc if g == A.TE_GROUP else sm))
            r = [x.rank for v in d.values() for x in v if x.lowrank]
            deltas.append(d)
            stats.append(st)
            ranks.append(max(r) if r else 0)
            used.append({"letter": letter, "name": b.name, "gain": gain, "strength": [sm, sc], "matched": st["matched"]})

        te_mode = {H.TE_MODES[0]: "merge", H.TE_MODES[1]: "first only", H.TE_MODES[2]: "drop"}.get(text_encoder, "merge")
        merged, report = core.merge_many(deltas, preset, wfuns, method, rank=output_rank, density=keep_ratio,
                                         seed=dare_seed, te_mode=te_mode, merged_scale=overall_strength, ranks=ranks)
        if not merged:
            raise RuntimeError("Nothing to merge: no LoRA matched the connected model (or every LoRA is muted). "
                               "Check the architecture / UNET and the mixer.")

        dtype = PRECISIONS.get(precision, torch.float16)
        sd = core.deltas_to_state_dict(merged, key_style, dtype)
        out_name = os.path.splitext(os.path.basename(filename or "SatoDive_merge"))[0]
        recipe = {"loras": used, "method": method, "overall_strength": overall_strength,
                  "use_slot_strengths": use_slot_strengths, "output_rank": output_rank, "keep_ratio": keep_ratio,
                  "text_encoder": text_encoder, "mix_settings": mix, "arch": pipe["arch"], "output_name": out_name,
                  "dare_seed": dare_seed}
        triggers = list(dict.fromkeys(t for _, b in connected for t in ((b.info or {}).get("trigger_words") or [])))

        saved_path = ""
        if save_lora:
            saved_path = core.unique_lora_save_path(out_name)
            core.save_lora(sd, saved_path, core.build_metadata(pipe["arch"], recipe, report["ranks"], triggers))
            log.info("[SatoDive] saved merged LoRA -> %s", saved_path)

        merged_info = core.inspect_state_dict(sd, {"satodive_recipe": json.dumps(recipe)}, pipe["arch"])
        merged_info["trigger_words"] = triggers
        bundle = core.SatoLoRA("merge({})".format(" + ".join(os.path.basename(b.name) for _, b in connected)), sd, {},
                               path=saved_path or None, arch=pipe["arch"], info=merged_info, merged=True, recipe=recipe)

        m2, c2 = core.apply_lora(model, clip, sd, preview_strength,
                                               preview_strength if clip is not None else 0.0)
        if clip is None:
            c2 = None

        img = P.blank()
        ui = {}
        if not preview.lower().startswith("off"):
            merged_img = P.render(pipe, m2, c2 if c2 is not None else clip)
            if preview.lower().startswith("compare"):
                tiles = []
                for (letter, b), u in zip(connected, used):
                    mi, ci = core.apply_lora(model, clip, b.sd, u["strength"][0],
                                                           u["strength"][1] if clip is not None else 0.0)
                    tiles.append(P.label_image(P.render(pipe, mi, ci if ci is not None else clip), letter))
                tiles.append(P.label_image(merged_img, "MERGE"))
                img = P.grid(tiles, max_cols=3 if len(tiles) > 4 else 4)
            elif "base" in preview.lower():
                img = P.hstack([P.label_image(P.render(pipe, model, clip), "BASE"), P.label_image(merged_img, "MERGE")])
            else:
                img = merged_img
            ui = _preview_ui(img, "SatoDive_merge")

        report.update({
            "loras": [{"letter": u["letter"], "matched": u["matched"], "gain": u["gain"]} for u in used],
            "saved": os.path.relpath(saved_path, folder_paths.get_folder_paths("loras")[0]) if saved_path else "",
            "size_mb": merged_info.get("size_mb"),
            "patched": P.patched_count(model, m2),
        })
        ui["sato_report"] = [report]
        group_arch = pipe.get("group_arch", pipe["arch"])
        groups = core.groups_payload_many([core.delta_groups(d, preset) for d in deltas], group_arch)
        for g in groups:  # counts keyed by letter: robust even if the frontend merges ui lists
            g["by"] = {u["letter"]: g["counts"][i] for i, u in enumerate(used)}
        ui["sato_groups"] = groups
        ui["sato_letters"] = [u["letter"] for u in used]
        return io.NodeOutput(bundle, m2, c2, img, saved_path, ui=ui)


NODE_CLASS_MAPPINGS = {
    "SatoDiveLoRAStudioSetup": SatoDiveStudioSetup,
    "SatoDiveLoRASlot": SatoDiveLoRASlot,
    "SatoDiveLoRAMergeStudioV2": SatoDiveLoRAMergeStudioV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SatoDiveLoRAStudioSetup": "① LoRA Studio Setup - SatoDive",
    "SatoDiveLoRASlot": "② LoRA Slot - SatoDive",
    "SatoDiveLoRAMergeStudioV2": "③ LoRA Merge Studio - SatoDive",
}
