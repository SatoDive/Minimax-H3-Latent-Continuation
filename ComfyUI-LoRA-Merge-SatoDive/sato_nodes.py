import json
import logging
import os

import torch

import comfy.model_management
import comfy.sd
import folder_paths
import nodes

from . import sato_arch as A
from . import sato_lora_core as core
from . import sato_preview as P

log = logging.getLogger("SatoDive.LoRAMerge")

CATEGORY = "SatoDive/LoRA Merge"
MAX_RES = nodes.MAX_RESOLUTION

PRECISIONS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
KEY_STYLES = ["comfy (diffusion_model.*)", "kohya (lora_unet_*)"]


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
                "model": ("MODEL", {"tooltip": "From the native Load Diffusion Model (UNETLoader) node."}),
                "architecture": (A.ARCH_NAMES, {"default": "Krea2 (Turbo)"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "default": "cinematic portrait of a woman in a neon-lit rainy street, 35mm photo, detailed skin"}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "",
                                        "tooltip": "Leave empty for distilled/turbo models (CFG 1): the negative is zeroed out."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed",
                                 "tooltip": "Fixed by default so tweaking the merge does not re-render the A / B previews."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
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
                "match_reference_size": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("SATO_PIPE", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("pipe", "model", "clip", "vae")
    FUNCTION = "setup"
    CATEGORY = CATEGORY

    def setup(self, model, architecture, prompt, negative, seed, steps, cfg, sampler, scheduler,
              width, height, shift, clip=None, vae=None, reference_image=None, match_reference_size=True):
        preset = A.get_preset(architecture)
        ok, cls_name = A.arch_matches_model(preset, model)
        warnings = []
        if not ok:
            warnings.append("Model class '{}' does not match '{}' (expects {}). Check your UNET loader."
                            .format(cls_name, architecture, "/".join(preset["model_classes"])))
            log.warning("[SatoDive] %s", warnings[-1])
        pipe = core.SatoPipe(
            model=model, clip=clip, vae=vae, arch=architecture,
            group_arch=_resolve_group_arch(architecture, model),
            prompt=prompt, negative=negative, seed=seed, steps=steps, cfg=cfg, sampler=sampler,
            scheduler=scheduler, width=width, height=height, shift=shift,
            reference_image=reference_image, match_reference_size=match_reference_size,
        )
        ui = {"sato_setup": [{"arch": architecture, "model_class": cls_name, "warnings": warnings,
                              "has_clip": clip is not None, "has_vae": vae is not None,
                              "has_reference": reference_image is not None}]}
        return {"ui": ui, "result": (pipe, model, clip, vae)}


# ======================================================================================
# 2) LoRA Slot A / B
# ======================================================================================

PREVIEW_SLOT = ["LoRA", "Base | LoRA", "off"]


class SatoDiveLoRASlotA:
    SLOT = "A"
    DESCRIPTION = "Load a LoRA, inspect it, and optionally render a preview with the models from the Studio Setup."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("SATO_PIPE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                "preview": (PREVIEW_SLOT, {"default": "LoRA",
                                           "tooltip": "Render a preview inside this node. 'off' only loads + analyses the LoRA."}),
            },
        }

    RETURN_TYPES = ("SATO_LORA", "MODEL", "CLIP", "IMAGE")
    RETURN_NAMES = ("lora", "model", "clip", "preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, pipe, lora_name, strength_model, strength_clip, preview):
        path, sd, meta = core.load_lora_file(lora_name)
        info = dict(core.inspect_lora_file(lora_name, pipe["arch"]))
        model, clip = pipe["model"], pipe.get("clip")

        matched, unmatched = core.count_matches(sd, core.build_key_map(model, clip), core.build_shape_lookup(model, clip))
        info["matched"] = matched
        info["unmatched"] = unmatched
        info["slot"] = self.SLOT
        if matched == 0:
            log.warning("[SatoDive] LoRA %s: no keys matched the connected model — wrong architecture?", lora_name)

        m2, c2 = comfy.sd.load_lora_for_models(model, clip, sd, strength_model, strength_clip if clip is not None else 0.0)
        if clip is None:
            c2 = None

        bundle = core.SatoLoRA(lora_name, sd, meta, path=path, arch=pipe["arch"],
                               strength_model=strength_model, strength_clip=strength_clip, info=info)

        img = P.blank()
        ui = {}
        if preview != "off":
            lora_img = P.render(pipe, m2, c2 if c2 is not None else clip)
            if preview == "Base | LoRA":
                base_img = P.render(pipe, model, clip)
                img = P.hstack([P.label_image(base_img, "BASE"), P.label_image(lora_img, self.SLOT)])
            else:
                img = lora_img
            ui = _preview_ui(img, "SatoDive_slot{}".format(self.SLOT))
        ui["sato_info"] = [info]
        return {"ui": ui, "result": (bundle, m2, c2, img)}


class SatoDiveLoRASlotB(SatoDiveLoRASlotA):
    SLOT = "B"


# ======================================================================================
# 3) Merge Studio
# ======================================================================================

PREVIEW_MERGE = ["Merged", "A | B | Merged", "Base | Merged", "off"]


class SatoDiveLoRAMergeStudio:
    DESCRIPTION = ("Blend LoRA A and LoRA B with global + per-block weights detected from both LoRAs, "
                   "preview the result and save it as a new LoRA.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("SATO_PIPE",),
                "lora_a": ("SATO_LORA",),
                "lora_b": ("SATO_LORA",),
                "method": (core.MERGE_METHODS, {"default": core.MERGE_METHODS[0],
                           "tooltip": "add: exact weighted sum (rank A+B). svd: weighted sum compressed to 'rank'. "
                                      "ties / dare: conflict-aware merges where both LoRAs touch the same weight."}),
                "weight_a": ("FLOAT", {"default": 1.0, "min": -3.0, "max": 3.0, "step": 0.01}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": -3.0, "max": 3.0, "step": 0.01}),
                "use_slot_strengths": ("BOOLEAN", {"default": True,
                                       "tooltip": "Multiply by the strengths set on the A / B slot nodes (WYSIWYG with their previews)."}),
                "rank": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1,
                                 "tooltip": "Output rank for svd / ties / dare. 0 = auto."}),
                "density": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                                      "tooltip": "ties / dare: fraction of each delta that is kept."}),
                "text_encoder": (core.TE_MODES, {"default": "merge"}),
                "merged_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01, "advanced": True,
                                           "tooltip": "Baked into the saved LoRA."}),
                "preview": (PREVIEW_MERGE, {"default": "Merged"}),
                "preview_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                "save_lora": ("BOOLEAN", {"default": False, "label_on": "save on run", "label_off": "don't save"}),
                "filename": ("STRING", {"default": "SatoDive_merge"}),
                "precision": (list(PRECISIONS.keys()), {"default": "fp16", "advanced": True}),
                "key_style": (KEY_STYLES, {"default": KEY_STYLES[0], "advanced": True,
                              "tooltip": "comfy: native ComfyUI keys. kohya: lora_unet_* keys (wider tool compatibility)."}),
                "dare_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "advanced": True,
                                      "tooltip": "Seed for the DARE random drop (kept fixed so results are reproducible)."}),
                "block_weights": ("STRING", {"default": "{}", "multiline": False,
                                             "tooltip": "Managed by the slider panel."}),
            },
        }

    RETURN_TYPES = ("SATO_LORA", "MODEL", "CLIP", "IMAGE", "STRING")
    RETURN_NAMES = ("merged_lora", "model", "clip", "preview", "saved_path")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, pipe, lora_a, lora_b, method, weight_a, weight_b, use_slot_strengths, rank, density,
            text_encoder, merged_scale, preview, preview_strength, save_lora, filename, precision,
            key_style, dare_seed, block_weights):
        model, clip = pipe["model"], pipe.get("clip")
        preset = A.get_preset(pipe.get("group_arch", pipe["arch"]))

        try:
            bw = json.loads(block_weights or "{}")
        except Exception:
            bw = {}
        blk_a = {k: float(v) for k, v in (bw.get("a") or {}).items()}
        blk_b = {k: float(v) for k, v in (bw.get("b") or {}).items()}

        key_map = core.build_key_map(model, clip)
        shapes = core.build_shape_lookup(model, clip)
        da, sa = core.extract_deltas(lora_a.sd, key_map, shapes)
        db, sb = core.extract_deltas(lora_b.sd, key_map, shapes)

        sm_a = lora_a.strength_model if use_slot_strengths else 1.0
        sc_a = lora_a.strength_clip if use_slot_strengths else 1.0
        sm_b = lora_b.strength_model if use_slot_strengths else 1.0
        sc_b = lora_b.strength_clip if use_slot_strengths else 1.0

        def wfun(blk, sm, sc):
            return lambda g: float(blk.get(g, 1.0)) * (sc if g == A.TE_GROUP else sm)

        ranks_a = [d.rank for v in da.values() for d in v if d.lowrank]
        ranks_b = [d.rank for v in db.values() for d in v if d.lowrank]
        merged, report = core.merge(
            da, db, preset, weight_a, weight_b, wfun(blk_a, sm_a, sc_a), wfun(blk_b, sm_b, sc_b), method,
            rank=rank, density=density, seed=dare_seed, te_mode=text_encoder, merged_scale=merged_scale,
            rank_a=max(ranks_a) if ranks_a else 0, rank_b=max(ranks_b) if ranks_b else 0,
        )
        if not merged:
            raise RuntimeError("SatoDive merge produced nothing: neither LoRA matched the connected model "
                               "(A matched {}, B matched {}). Check the architecture / UNET.".format(sa["matched"], sb["matched"]))

        dtype = PRECISIONS.get(precision, torch.float16)
        sd = core.deltas_to_state_dict(merged, key_style, dtype)

        out_name = os.path.splitext(os.path.basename(filename or "SatoDive_merge"))[0]
        recipe = {
            "a": lora_a.name, "b": lora_b.name, "method": method, "weight_a": weight_a, "weight_b": weight_b,
            "slot_strengths": [sm_a, sc_a, sm_b, sc_b], "rank": rank, "density": density,
            "text_encoder": text_encoder, "merged_scale": merged_scale, "block_weights": bw,
            "arch": pipe["arch"], "output_name": out_name, "dare_seed": dare_seed,
        }
        info_a, info_b = lora_a.info or {}, lora_b.info or {}
        triggers = list(dict.fromkeys((info_a.get("trigger_words") or []) + (info_b.get("trigger_words") or [])))

        saved_path = ""
        if save_lora:
            saved_path = core.unique_lora_save_path(out_name)
            core.save_lora(sd, saved_path, core.build_metadata(pipe["arch"], recipe, report["ranks"], triggers))
            log.info("[SatoDive] saved merged LoRA -> %s", saved_path)

        merged_info = core.inspect_state_dict(sd, {"satodive_recipe": json.dumps(recipe)}, pipe["arch"])
        merged_info["trigger_words"] = triggers
        bundle = core.SatoLoRA("merge({} + {})".format(lora_a.name, lora_b.name), sd, {}, path=saved_path or None,
                               arch=pipe["arch"], info=merged_info, merged=True, recipe=recipe)

        m2, c2 = comfy.sd.load_lora_for_models(model, clip, sd, preview_strength,
                                               preview_strength if clip is not None else 0.0)
        if clip is None:
            c2 = None

        img = P.blank()
        ui = {}
        if preview != "off":
            merged_img = P.render(pipe, m2, c2 if c2 is not None else clip)
            if preview == "A | B | Merged":
                ma, ca = comfy.sd.load_lora_for_models(model, clip, lora_a.sd, sm_a, sc_a if clip is not None else 0.0)
                mb, cb = comfy.sd.load_lora_for_models(model, clip, lora_b.sd, sm_b, sc_b if clip is not None else 0.0)
                img_a = P.render(pipe, ma, ca if ca is not None else clip)
                img_b = P.render(pipe, mb, cb if cb is not None else clip)
                img = P.hstack([P.label_image(img_a, "A"), P.label_image(img_b, "B"), P.label_image(merged_img, "MERGE")])
            elif preview == "Base | Merged":
                base_img = P.render(pipe, model, clip)
                img = P.hstack([P.label_image(base_img, "BASE"), P.label_image(merged_img, "MERGE")])
            else:
                img = merged_img
            ui = _preview_ui(img, "SatoDive_merge")

        report.update({
            "a_matched": sa["matched"], "b_matched": sb["matched"],
            "a_skipped": len(sa["skipped"]), "b_skipped": len(sb["skipped"]),
            "saved": os.path.relpath(saved_path, folder_paths.get_folder_paths("loras")[0]) if saved_path else "",
            "size_mb": merged_info.get("size_mb"),
        })
        ui["sato_report"] = [report]
        ui["sato_groups"] = core.groups_payload(core.delta_groups(da, preset), core.delta_groups(db, preset),
                                                pipe.get("group_arch", pipe["arch"]))
        return {"ui": ui, "result": (bundle, m2, c2, img, saved_path)}


NODE_CLASS_MAPPINGS = {
    "SatoDiveLoRAStudioSetup": SatoDiveStudioSetup,
    "SatoDiveLoRASlotA": SatoDiveLoRASlotA,
    "SatoDiveLoRASlotB": SatoDiveLoRASlotB,
    "SatoDiveLoRAMergeStudio": SatoDiveLoRAMergeStudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SatoDiveLoRAStudioSetup": "① LoRA Studio Setup - SatoDive",
    "SatoDiveLoRASlotA": "② LoRA Slot A - SatoDive",
    "SatoDiveLoRASlotB": "③ LoRA Slot B - SatoDive",
    "SatoDiveLoRAMergeStudio": "④ LoRA Merge Studio - SatoDive",
}
