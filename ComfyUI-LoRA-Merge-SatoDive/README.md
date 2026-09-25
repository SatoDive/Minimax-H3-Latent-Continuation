# LoRA Merge Studio - SatoDive

A ComfyUI node pack for inspecting, previewing and merging two LoRAs, then saving the result as a new LoRA. Everything happens inside ComfyUI.

![screenshot](docs/screenshot.png)

## The four nodes

| Node | What it does |
|---|---|
| **① LoRA Studio Setup - SatoDive** | Connect your **native** `Load Diffusion Model`, `Load CLIP` and `Load VAE` nodes here and pick the **architecture**. Picking one fills in the recommended steps, CFG, sampler, scheduler, size and shift, and you can still change any of them. A card lists the files and the CLIPLoader type to use, and warns you if the UNET you loaded is a different model class. There's an optional reference image input for edit models. |
| **② LoRA Slot A / ③ LoRA Slot B - SatoDive** | Load a LoRA and see its info card: rank, alpha, size, key format (kohya / diffusers / comfy), detected architecture, text-encoder modules and trigger words (click one to copy it). Preview is optional: set `preview` to `LoRA` or `Base \| LoRA`, then press **▶ Run this LoRA preview**. That queues only this node, and the image shows up inside it. |
| **④ LoRA Merge Studio - SatoDive** | Merge method, global A/B weights, and **per-block A/B sliders detected from both LoRAs**. It has presets (Flat, 50/50, A only, B only, Ramp A→B, Ramp B→A, Swap, TE off). **▶ Preview merge** renders `Merged`, `A \| B \| Merged` or `Base \| Merged` inside the node. **💾 Merge & Save LoRA** writes `models/loras/SatoDive/<name>_###.safetensors`. |

The Merge Studio also outputs the merged LoRA (`merged_lora`). You can feed that into another Merge Studio to merge 3 or more LoRAs. It also outputs the patched `MODEL`/`CLIP`, so you can keep going in a normal workflow.

## Architectures

| Preset | CLIPLoader type | Defaults |
|---|---|---|
| **Krea2 (Turbo)** | `krea2` (Qwen3-VL 4B) | 8 steps, CFG 1, euler/simple; reference image supported |
| **Krea2 Raw (Base)** | `krea2` | 30 steps, CFG 4, euler/simple |
| **Z-Image Turbo** | `lumina2` (Qwen3 4B) | 8 steps, CFG 1, res_multistep/simple, shift 3 |
| **Flux.2 Klein 9B** | `flux2` (Qwen3 8B) | 4 steps, CFG 1, Flux2 schedule; reference image = edit |
| **Qwen-Image 2.1 (Edit)** | `qwen_image` (Qwen3-VL 8B) | 30 steps, CFG 4; uses the native Qwen-Image 2.1 edit encoder with your reference image |
| **Auto-detect / Other** | whatever matches | generic; blocks detected automatically |

The UNET, text encoder and VAE always come from ComfyUI's own loaders, so you can swap any of them by hand. The presets only set the preview defaults, the hints, and the block layout used for the sliders. The Krea2 Raw and Qwen-Image 2.1 step/CFG values are starting points. Adjust them in the Setup node if your checkpoint prefers other settings.

## Merge methods

* **add (lossless concat)**: an exact `wA·A + wB·B`. The rank is A+B, and nothing is approximated.
* **svd (add + resize rank)**: the same sum, compressed to `rank` with an exact low-rank SVD. `0` = auto, which keeps the larger LoRA's rank on each weight. The report shows the % of energy kept.
* **ties / dare_ties / dare_linear / magnitude**: conflict-aware merges. They apply where both LoRAs touch the same weight, and the result is re-factorised to a LoRA. `density` controls how much is kept.

Other options:

* `use_slot_strengths` multiplies in the strengths set on the Slot nodes, so the merge matches what you saw in their previews.
* `text_encoder` can be `merge`, `A only`, `B only` or `drop`.
* `merged_scale` is baked into the saved file.
* `key_style` is `comfy` (native) or `kohya` (`lora_unet_*`, for other tools).

### How the merge works

Both LoRAs are resolved through **ComfyUI's own LoRA key maps**, so the merge happens in model-weight space. That means you can merge a kohya LoRA with a diffusers/PEFT LoRA for the same model, including LoRAs that address fused weights (qkv / gate_up) through slices. The saved LoRA loads with the normal `Load LoRA` node, and its metadata stores the recipe and the combined trigger words.

## Install

1. Unzip into `ComfyUI/custom_nodes/` so you get `ComfyUI/custom_nodes/ComfyUI-LoRA-Merge-SatoDive/`.
2. Restart ComfyUI and refresh the browser.
3. Load a workflow from `example_workflows/` (Krea2, Qwen-Image 2.1 Edit, Z-Image Turbo, Klein 9B), or add the nodes from **SatoDive/LoRA Merge**.

No extra Python dependencies. It needs a recent ComfyUI (with Krea2 / Qwen-Image 2.1 / Flux2 support).

## Tips

* Leave the Setup `seed` on **fixed**. Then when you tweak the merge sliders, ComfyUI re-uses the cached A/B previews and only re-renders the merge.
* To change all blocks in a group at once, drag the bold section slider. Double-click a value to reset it to 1.0.
* If a Slot card warns about the architecture, the LoRA was probably trained for another model.
