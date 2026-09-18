# Comfyui-Minimax-H3-Latent-Continuation 🎬SatoDive ✨

An enhanced continuation & latent-stitching extension for **ComfyUI** and **MiniMax-H3**, heavily modified from [nkxx188/ComfyUI-MiniMaxH3-Easy](https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy).

This suite enables **long, seamless multi-shot video continuations** with **100% visual context and environment consistency**—completely eliminating generation-to-generation quality degradation!

---
### 🎥 Click the image below to watch the tutorial :
[![Watch the tutorial](https://img.youtube.com/vi/KkugNjrpXAc/maxresdefault.jpg)](https://www.youtube.com/watch?v=KkugNjrpXAc)

## 💡 The Problem & The Solution

In typical multi-shot AI video workflows, continuing a scene requires exporting an MP4 and feeding it back into the model. This continuous **pixel decoding and re-encoding** causes:
- Blurry textures and generational compression loss.
- Darkened contrast, color drift, and muddy shadows.
- Severe character inconsistency and room layout drift.

### 🚀 The Solution: Native Latent Chaining
This extension operates directly inside **latent space**:
1. **Zero Quality Loss:** By saving and passing the raw uncompressed latent states directly between shots, colors and fine details stay razor-sharp across chained generations.
2. **Dual-Reference Pipeline:** 
   - **`Seed Video`**: Acts as a temporal transition guide using the tail of the previous clip for smooth motion continuity.
   - **`Seed Ref Video`**: Passes high-level macro context (e.g. an earlier wide master shot of the room layout) so changing camera angles doesn't break the environment.
3. **Auto Last-Frame Trim & Stitch:** Automatically locates the transition boundary frame and joins the previous and current clips into one seamless file.

---

## ✨ Key Additions Over Base

Built upon the great foundation of `MiniMaxH3-Easy`, this modified pack adds:
- 💾 **Latent Save / Load System:** Dedicated nodes to export and reload native video latents by label.
- 🔀 **Split Context Architecture:** Separate inputs for continuation guides vs. environmental context memory.
- ✂️ **Automatic Clip Concatenation:** Seamlessly stitches shots together without needing external editing software.
- ⚖️ **Subject Weight Modulation:** Allows lowering reference strength on recurring subjects so newly introduced characters get model focus.

---

## 📦 Installation

### Option 1: Git Clone (Recommended)

1. Open your terminal and navigate to your custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/

2. Clone this repository:

git clone https://github.com/SatoDive/Minimax-H3-Latent-Continuation.git

3. Restart ComfyUI.


🚀 Quick Start
Download the workflow .json provided in the /workflow directory (or from the video description).
Drag and drop the workflow into ComfyUI.
Scene 1: Connect character reference images, write your prompt, and save the latent file.
Scene 2 (Continuation): Load the saved latent into seed-latent, connect the previous clip to seed video, and queue your next shot!

💡 Tip: If newly saved latents don't appear in the dropdown list right away, press R on your keyboard to refresh ComfyUI.


🔮 Roadmap (Version 2 in Progress)
One-Shot Multi-Segment Pipeline: Feed in a single master prompt with multiple connected shots and let the node chain 6+ video clips automatically in a single queue.
Constant VRAM footprint across long cinematic sequences.

🙏 Credits & Acknowledgments
This project is a heavily modified fork built on top of ComfyUI-MiniMaxH3-Easy by @nkxx188.
Huge thanks to nkxx188 for creating the original compact MiniMax-H3 node suite, the unified multimedia inputs, and prompt reference system that made this continuation workflow possible!

📄 License
This repository inherits the license terms of the upstream project ([ComfyUI-MiniMaxH3-Easy](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fnkxx188%2FComfyUI-MiniMaxH3-Easy)).


   
