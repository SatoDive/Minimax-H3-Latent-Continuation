"""
Architecture presets for the SatoDive LoRA Merge pack.

Each preset describes:
  * recommended native loader files / CLIPLoader type (shown in the UI as a hint)
  * sampling defaults for the in-node previews
  * how transformer blocks are named, so the Merge Studio can build per-block sliders
  * aliases that translate diffusers/kohya LoRA block names into ComfyUI native names
"""

import re

AUTO = "Auto-detect / Other"

# Settings come from the official ComfyUI workflow templates (comfyui-workflow-templates 0.11.74)
# unless marked as a starting point.
# NOTE: the order of "containers" matters: longer / more specific names first.
ARCH_PRESETS = {
    "Krea2 (Turbo)": {
        "key": "krea2", "family": "krea2",
        "model_classes": ["Krea2"],
        "clip_type": "krea2",
        "unet_hint": "krea2_turbo_fp8_scaled.safetensors  (models/diffusion_models)",
        "te_hint": "qwen3vl_4b_fp8_scaled.safetensors  (CLIPLoader type: krea2)",
        "vae_hint": "qwen_image_vae.safetensors",
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "width": 1024, "height": 1024, "shift": 0.0,
        "edit": "reference_latents",
        "notes": "Official Krea-2 Turbo settings: 8 steps, CFG 1, euler / simple. Leave the negative prompt empty. A reference image works with style / edit LoRAs.",
        "containers": [
            ("txtfusion.layerwise_blocks", "Text-Fusion L"),
            ("txtfusion.refiner_blocks", "Text-Fusion R"),
            ("blocks", "Block"),
        ],
        "aliases": [
            ("text_fusion.layerwise_blocks", "txtfusion.layerwise_blocks"),
            ("text_fusion.refiner_blocks", "txtfusion.refiner_blocks"),
            ("transformer_blocks", "blocks"),
        ],
    },
    "Krea2 Raw (Base)": {
        "key": "krea2_raw", "family": "krea2",
        "model_classes": ["Krea2"],
        "clip_type": "krea2",
        "unet_hint": "krea2 raw / base weights  (models/diffusion_models)",
        "te_hint": "qwen3vl_4b_fp8_scaled.safetensors  (CLIPLoader type: krea2)",
        "vae_hint": "qwen_image_vae.safetensors",
        "steps": 30, "cfg": 4.0, "sampler": "euler", "scheduler": "simple",
        "width": 1024, "height": 1024, "shift": 0.0,
        "edit": "reference_latents",
        "notes": "Undistilled raw weights need more steps and real CFG (no official template yet - 30 steps / CFG 4 is a starting point). A negative prompt helps here.",
        "containers": [
            ("txtfusion.layerwise_blocks", "Text-Fusion L"),
            ("txtfusion.refiner_blocks", "Text-Fusion R"),
            ("blocks", "Block"),
        ],
        "aliases": [
            ("text_fusion.layerwise_blocks", "txtfusion.layerwise_blocks"),
            ("text_fusion.refiner_blocks", "txtfusion.refiner_blocks"),
            ("transformer_blocks", "blocks"),
        ],
    },
    "Z-Image Turbo": {
        "key": "zimage", "family": "zimage",
        "model_classes": ["Lumina2", "ZImage"],
        "clip_type": "lumina2",
        "unet_hint": "z_image_turbo_bf16.safetensors  (models/diffusion_models)",
        "te_hint": "qwen_3_4b.safetensors  (CLIPLoader type: lumina2)",
        "vae_hint": "ae.safetensors  (Flux VAE)",
        "steps": 8, "cfg": 1.0, "sampler": "res_multistep", "scheduler": "simple",
        "width": 1024, "height": 1024, "shift": 3.0,
        "edit": None,
        "notes": "Official Z-Image Turbo settings: 8 steps, CFG 1, res_multistep / simple, shift 3. Leave the negative prompt empty.",
        "containers": [
            ("context_refiner", "Context Refiner"),
            ("noise_refiner", "Noise Refiner"),
            ("layers", "Layer"),
        ],
        "aliases": [],
    },
    "Z-Image Base": {
        "key": "zimage_base", "family": "zimage",
        "model_classes": ["Lumina2", "ZImage"],
        "clip_type": "lumina2",
        "unet_hint": "z_image_bf16.safetensors  (models/diffusion_models)",
        "te_hint": "qwen_3_4b.safetensors  (CLIPLoader type: lumina2)",
        "vae_hint": "ae.safetensors  (Flux VAE)",
        "steps": 25, "cfg": 4.0, "sampler": "res_multistep", "scheduler": "simple",
        "width": 1024, "height": 1024, "shift": 3.0,
        "edit": None,
        "notes": "Official Z-Image Base settings: 25 steps, CFG 4, res_multistep / simple, shift 3. Negative prompt supported.",
        "containers": [
            ("context_refiner", "Context Refiner"),
            ("noise_refiner", "Noise Refiner"),
            ("layers", "Layer"),
        ],
        "aliases": [],
    },
    "Flux.2 Klein 9B": {
        "key": "klein9b", "family": "klein",
        "model_classes": ["Flux2"],
        "clip_type": "flux2",
        "unet_hint": "flux-2-klein-9b-fp8.safetensors  (models/diffusion_models)",
        "te_hint": "qwen_3_8b_fp8mixed.safetensors  (CLIPLoader type: flux2)",
        "vae_hint": "full_encoder_small_decoder.safetensors  or  flux2-vae.safetensors",
        "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "flux2",
        "width": 1024, "height": 1024, "shift": 0.0,
        "edit": "reference_latents",
        "notes": "Official Klein 9B (distilled) settings: 4 steps, CFG 1, Flux2 schedule. A reference image switches the preview to edit mode.",
        "containers": [
            ("double_blocks", "Double"),
            ("single_blocks", "Single"),
        ],
        "aliases": [
            ("single_transformer_blocks", "single_blocks"),
            ("transformer_blocks", "double_blocks"),
        ],
    },
    "Flux.2 Klein 9B Base": {
        "key": "klein9b_base", "family": "klein",
        "model_classes": ["Flux2"],
        "clip_type": "flux2",
        "unet_hint": "flux-2-klein-base-9b-fp8.safetensors  (models/diffusion_models)",
        "te_hint": "qwen_3_8b_fp8mixed.safetensors  (CLIPLoader type: flux2)",
        "vae_hint": "full_encoder_small_decoder.safetensors  or  flux2-vae.safetensors",
        "steps": 20, "cfg": 5.0, "sampler": "euler", "scheduler": "flux2",
        "width": 1024, "height": 1024, "shift": 0.0,
        "edit": "reference_latents",
        "notes": "Official Klein 9B Base settings: 20 steps, CFG 5, Flux2 schedule. Most Klein LoRAs are trained on this base.",
        "containers": [
            ("double_blocks", "Double"),
            ("single_blocks", "Single"),
        ],
        "aliases": [
            ("single_transformer_blocks", "single_blocks"),
            ("transformer_blocks", "double_blocks"),
        ],
    },
    "Qwen-Image 2.1 (Edit)": {
        "key": "qwen21", "family": "qwen21",
        "model_classes": ["QwenImage21", "QwenImage"],
        "clip_type": "qwen_image",
        "unet_hint": "qwen_image_2.1_int8_convrot.safetensors  (or fp8 / bf16)",
        "te_hint": "qwen3vl_8b_int8_convrot.safetensors  (CLIPLoader type: qwen_image)",
        "vae_hint": "qwen_image_2.1_vae_bf16.safetensors",
        "steps": 25, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "width": 1024, "height": 1024, "shift": 0.0,
        "edit": "qwen21",
        "notes": "Official Qwen-Image 2.1 settings: 25 steps, CFG 1, euler / simple. It does text-to-image and editing: connect a reference image to preview edits.",
        "containers": [
            ("transformer_blocks", "Block"),
        ],
        "aliases": [],
    },
    AUTO: {
        "key": "auto", "family": "auto",
        "model_classes": [],
        "clip_type": "(match your model)",
        "unet_hint": "any model supported by ComfyUI",
        "te_hint": "matching text encoder",
        "vae_hint": "matching VAE",
        "steps": 20, "cfg": 4.0, "sampler": "euler", "scheduler": "simple",
        "width": 1024, "height": 1024, "shift": 0.0,
        "edit": "reference_latents",
        "notes": "Generic mode for any other model. Block groups are detected automatically from the LoRA keys.",
        "containers": [],
        "aliases": [],
    },
}

ARCH_NAMES = list(ARCH_PRESETS.keys())

# Generic container names used when no preset matches.
GENERIC_CONTAINERS = [
    ("txtfusion.layerwise_blocks", "Text-Fusion L"),
    ("txtfusion.refiner_blocks", "Text-Fusion R"),
    ("single_transformer_blocks", "Single"),
    ("single_blocks", "Single"),
    ("double_blocks", "Double"),
    ("transformer_blocks", "Block"),
    ("context_refiner", "Context Refiner"),
    ("noise_refiner", "Noise Refiner"),
    ("joint_blocks", "Joint"),
    ("blocks", "Block"),
    ("layers", "Layer"),
]

TE_GROUP = "text_encoder"
OTHER_GROUP = "other"

TE_PREFIXES = (
    "lora_te", "text_encoder", "text_encoders.", "te_", "te.", "lora_clip", "clip_l.", "clip_g.",
    "t5xxl.", "qwen", "gemma", "llama", "mistral",
)

UNET_PREFIXES = (
    "model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "unet.",
    "lora_unet__", "lora_unet_", "lycoris_", "lora_transformer_", "model.",
)
UNDERSCORE_PREFIXES = ("lora_unet__", "lora_unet_", "lycoris_", "lora_transformer_")

ADAPTER_MARKERS = (
    ".lora_up.", ".lora_down.", ".lora_mid.", ".lora_A.", ".lora_B.", ".lora_A", ".lora_B",
    "_lora.up.", "_lora.down.", ".lora.up.", ".lora.down.", ".lora_linear_layer.",
    ".alpha", ".dora_scale", ".hada_", ".lokr_", ".oft_", ".oft_blocks", ".boft_",
    ".a1.weight", ".a2.weight", ".b1.weight", ".b2.weight",
    ".diff_b", ".diff", ".w_norm", ".b_norm", ".set_weight", ".reshape_weight",
)


def get_preset(name):
    return ARCH_PRESETS.get(name, ARCH_PRESETS[AUTO])


def preset_by_key(key):
    for name, p in ARCH_PRESETS.items():
        if p["key"] == key:
            return name, p
    return AUTO, ARCH_PRESETS[AUTO]


def public_presets():
    """JSON friendly view for the frontend."""
    out = {}
    for name, p in ARCH_PRESETS.items():
        out[name] = {k: v for k, v in p.items() if k not in ("containers", "aliases")}
        out[name]["containers"] = [c for c, _ in p["containers"]]
    return out


# --------------------------------------------------------------------------------------
# Key helpers
# --------------------------------------------------------------------------------------

def module_path(raw_key):
    """Strip adapter suffixes: 'transformer.blocks.0.attn.wq.lora_A.weight' -> 'transformer.blocks.0.attn.wq'."""
    idx = len(raw_key)
    for m in ADAPTER_MARKERS:
        i = raw_key.find(m)
        if i != -1 and i < idx:
            idx = i
    if idx == len(raw_key):
        if raw_key.endswith(".weight") or raw_key.endswith(".bias"):
            return raw_key.rsplit(".", 1)[0]
        return raw_key
    return raw_key[:idx]


def is_te_key(path):
    p = path
    if p.startswith("text_encoders.") or p.startswith("lora_te") or p.startswith("text_encoder"):
        return True
    for pre in UNET_PREFIXES:
        if p.startswith(pre):
            return False
    return p.startswith(TE_PREFIXES)


def _strip_unet_prefix(path):
    underscore = False
    changed = True
    while changed:
        changed = False
        for pre in UNET_PREFIXES:
            if path.startswith(pre):
                if pre in UNDERSCORE_PREFIXES:
                    underscore = True
                path = path[len(pre):]
                changed = True
                break
    return path, underscore


def _containers_for(preset):
    cont = list(preset.get("containers") or [])
    aliases = list(preset.get("aliases") or [])
    if not cont:
        cont = list(GENERIC_CONTAINERS)
    # Canonical containers + alias sources; longest first so that 'single_transformer_blocks'
    # wins against 'transformer_blocks'.
    entries = [(c, c) for c, _ in cont] + [(src, dst) for src, dst in aliases]
    entries.sort(key=lambda e: len(e[0]), reverse=True)
    return entries


def group_of_path(path, preset, te=None):
    """Canonical block group ('double_blocks.3', 'text_encoder', 'other') for a LoRA / model path."""
    if te is None:
        te = is_te_key(path)
    if te:
        return TE_GROUP
    p, underscore = _strip_unet_prefix(path)
    for src, dst in _containers_for(preset):
        if underscore:
            m = re.match(re.escape(src.replace(".", "_")) + r"_(\d+)(?:_|$)", p)
        else:
            m = re.match(re.escape(src) + r"\.(\d+)(?:\.|$)", p)
        if m:
            return "{}.{}".format(dst, int(m.group(1)))
    # generic fallback
    m = re.match(r"^([A-Za-z0-9_\.]*?(?:blocks|layers|refiner))[\._](\d+)(?:[\._]|$)", p)
    if m:
        return "{}.{}".format(m.group(1), int(m.group(2)))
    return OTHER_GROUP


def group_of_model_key(model_key, preset):
    """Group for a ComfyUI weight key ('diffusion_model.blocks.3.attn.wq.weight')."""
    if isinstance(model_key, tuple):
        model_key = model_key[0]
    if not model_key.startswith("diffusion_model."):
        return TE_GROUP
    path = model_key[len("diffusion_model."):]
    return group_of_path(path, preset, te=False)


def group_sort_key(group, preset):
    if group == TE_GROUP:
        return (10_000, 0, group)
    if group == OTHER_GROUP:
        return (9_000, 0, group)
    name, _, idx = group.rpartition(".")
    order = [c for c, _ in (preset.get("containers") or GENERIC_CONTAINERS)]
    try:
        o = order.index(name)
    except ValueError:
        o = 500
    try:
        i = int(idx)
    except ValueError:
        i = 0
    return (o, i, group)


def group_label(group, preset):
    if group == TE_GROUP:
        return ("Text Encoder", "TE")
    if group == OTHER_GROUP:
        return ("Embeddings / IO", "Other")
    name, _, idx = group.rpartition(".")
    labels = dict(preset.get("containers") or []) or dict(GENERIC_CONTAINERS)
    lab = labels.get(name) or dict(GENERIC_CONTAINERS).get(name) or name
    return (lab, "{} {}".format(lab, idx))


def detect_arch(keys):
    """Best-effort architecture guess from LoRA keys. Returns preset name."""
    ks = " ".join(list(keys)[:4000])
    if any(t in ks for t in ("double_blocks", "single_blocks", "single_transformer_blocks")):
        return "Flux.2 Klein 9B"
    if any(t in ks for t in ("txtfusion", "text_fusion", ".attn.wq", "_attn_wq", "attn.to_gate", "attn_to_gate")):
        return "Krea2 (Turbo)"
    if any(t in ks for t in ("noise_refiner", "context_refiner", "feed_forward.w1", "feed_forward_w1",
                             "attention.to_q", "attention_to_q", "attention.qkv", "attention_qkv")):
        return "Z-Image Turbo"
    if any(t in ks for t in ("img_mlp", "txt_mlp", "img_mod", "txt_mod")):
        return "Qwen-Image 2.1 (Edit)"
    return AUTO


def arch_matches_model(preset, model):
    """Returns (ok, model_class_name)."""
    try:
        cls = type(model.model).__name__
    except Exception:
        return True, "?"
    if not preset.get("model_classes"):
        return True, cls
    return cls in preset["model_classes"], cls


def family_of_base_model(base):
    """Map a Civitai / metadata base-model string to a preset family."""
    b = (base or "").lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
    if not b:
        return None
    if "krea" in b and "flux1" not in b:
        return "krea2"
    if "klein" in b:
        return "klein"
    if "zimage" in b:
        return "zimage"
    if "qwen" in b:
        return "qwen21"
    return "other"
