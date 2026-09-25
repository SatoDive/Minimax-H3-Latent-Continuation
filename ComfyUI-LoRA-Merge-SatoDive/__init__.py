"""
LoRA Merge Studio - SatoDive
A modern LoRA inspection / preview / merging node pack for ComfyUI.
Presets: Krea2, Krea2 Raw, Z-Image Turbo, Flux.2 Klein 9B, Qwen-Image 2.1 (+ auto).
"""

import logging

from .sato_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

try:
    from server import PromptServer
    from . import sato_routes

    sato_routes.register(PromptServer.instance)
except Exception as e:  # pragma: no cover - server not available (e.g. tests)
    logging.getLogger("SatoDive.LoRAMerge").warning("[SatoDive] UI routes not registered: %s", e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
