"""v2 nodes: one LoRA Slot node (add as many as you like) and a Merge Studio for 2-10 LoRAs."""

import json
import logging
import os

import torch

import comfy.sd
import folder_paths
from comfy_api.latest import io

from . import sato_arch as A
from . import sato_help as H
from . import sato_lora_core as core
from . import sato_preview as P
from .sato_nodes import CATEGORY, PRECISIONS, KEY_STYLES, _preview_ui, run_slot

log = logging.getLogger("SatoDive.LoRAMerge")

LETTERS = P.LETTERS
SatoLoraType = io.Custom("SATO_LORA")
SatoPipeType = io.Custom("SATO_PIPE")


# ======================================================================================
# LoRA Slot (classic node so it keeps the familiar LoRA file picker)
# ======================================================================================

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
# Merge Studio (V3 node -> native auto-growing LoRA inputs)
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
            display_name="④ LoRA Merge Studio - SatoDive",
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
        # v1 style keys so the info card can show "merge of A + B"
        recipe["a"], recipe["b"] = used[0]["name"], used[1]["name"] if len(used) > 1 else ""
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

        m2, c2 = comfy.sd.load_lora_for_models(model, clip, sd, preview_strength,
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
                    mi, ci = comfy.sd.load_lora_for_models(model, clip, b.sd, u["strength"][0],
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
    "SatoDiveLoRASlot": SatoDiveLoRASlot,
    "SatoDiveLoRAMergeStudioV2": SatoDiveLoRAMergeStudioV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SatoDiveLoRASlot": "② LoRA Slot - SatoDive",
    "SatoDiveLoRAMergeStudioV2": "④ LoRA Merge Studio - SatoDive",
}
