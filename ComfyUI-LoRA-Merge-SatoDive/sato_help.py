"""Plain-language help shown in tooltips and in the guide cards inside the nodes."""

# Friendly labels for the merge methods (what the user sees in the dropdown).
METHODS = [
    "Blend - keep everything (exact)",
    "Blend - smaller file (SVD)",
    "Smart blend - fix conflicts (TIES)",
    "Bold mix - stronger style (DARE-TIES)",
    "Soft mix - gentle blend (DARE)",
    "Strongest wins - max impact",
]

METHOD_HELP = {
    METHODS[0]: {
        "what": "Adds the LoRAs together exactly. Nothing is lost - the result looks like stacking the LoRAs in a normal workflow.",
        "pick": "Best default. Pick this when you want every LoRA at full power (e.g. a character + a style).",
        "cost": "File size = all LoRAs added together.",
    },
    METHODS[1]: {
        "what": "Same exact blend, then squeezed down to the 'output_rank' size.",
        "pick": "Pick this to get a smaller file to share or save VRAM. 'energy kept' above ~95% means it looks the same.",
        "cost": "Lower rank = smaller file but a little less detail.",
    },
    METHODS[2]: {
        "what": "Where LoRAs push the same weight in opposite directions, keeps the winning direction and drops the weak, conflicting bits.",
        "pick": "Pick this if the merged image looks muddy, washed out or like the LoRAs are 'fighting'. Gives a cleaner mix.",
        "cost": "Uses 'keep_ratio': lower = cleaner but weaker.",
    },
    METHODS[3]: {
        "what": "Randomly drops part of each LoRA, boosts what is left, then resolves conflicts.",
        "pick": "Want MORE style change or a punchier blend? Pick this. Try keep_ratio 0.3 - 0.6 and different dare_seed values.",
        "cost": "Each dare_seed gives a slightly different result.",
    },
    METHODS[4]: {
        "what": "Random thinning without conflict resolution - the LoRAs melt into each other.",
        "pick": "Pick this for subtle, painterly mixes of two styles.",
        "cost": "Each dare_seed gives a slightly different result.",
    },
    METHODS[5]: {
        "what": "For every weight, takes the change from whichever LoRA pushes it hardest.",
        "pick": "Pick this to keep the strongest features of each LoRA, e.g. a strong art style that should dominate.",
        "cost": "Can look harsh - lower overall_strength if it is too much.",
    },
}

TE_MODES = ["Blend (follow sliders)", "Use the first LoRA only", "Drop (image model only)"]

PREVIEW_MERGE = ["Merged result", "Compare: every LoRA + merged", "Before / after (base | merged)", "Off"]
PREVIEW_SLOT = ["LoRA only", "Before / after (base | LoRA)", "Off (just load)"]

TIPS = {
    "overall_strength": "How strong the merged LoRA is. This is baked into the saved file (1.0 = as mixed).",
    "use_slot_strengths": "ON: the strength you set on each LoRA Slot is used in the merge, so the merge matches what you saw in the slot previews.",
    "output_rank": "Detail / file-size of the result. 0 = automatic (recommended). Only used by the SVD, smart, bold, soft and strongest methods.",
    "keep_ratio": "Smart / Bold / Soft methods: how much of each LoRA is kept. 0.5 is a good start. Lower = cleaner but weaker.",
    "text_encoder": "Some LoRAs also change the text encoder (how prompts are understood). 'Blend' mixes it like the image part.",
    "preview": "What to render inside the node when you press Preview merge.",
    "preview_strength": "Strength used for the preview and for the MODEL / CLIP outputs only (not saved).",
    "save_lora": "Turn on (or press the Save button) to write the merged LoRA to models/loras/SatoDive/.",
    "filename": "Name of the saved file. A number is added so nothing is ever overwritten.",
    "precision": "fp16 is right for almost everyone. bf16 for some trainers, fp32 = double size.",
    "key_style": "comfy works in ComfyUI. kohya (lora_unet_*) is more compatible with other tools.",
    "dare_seed": "Random seed for the Bold / Soft methods. Change it to get a different variation.",
}

BLOCK_HELP = ("Rough guide: early blocks shape layout, pose and composition; late blocks shape textures, "
              "colors, lighting and fine style. Use 'Composition' on one LoRA and 'Style' on another to "
              "take the layout from one and the look from the other.")

HELP = {
    "methods": METHODS,
    "method_help": METHOD_HELP,
    "tips": TIPS,
    "block_help": BLOCK_HELP,
}
