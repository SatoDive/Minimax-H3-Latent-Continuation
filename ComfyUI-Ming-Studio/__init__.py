"""ComfyUI-Ming-Studio: design generation, layer decomposition, layer studio and PSD export
for inclusionAI's Ming-Image 0.1 (Design + Design-Layer) using ComfyUI's native loaders."""

import os
import time
import zipfile

import numpy as np
from PIL import Image

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .ming_studio import psd_writer

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _register_routes():
    try:
        import folder_paths
        from aiohttp import web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:  # running outside the ComfyUI server
        return

    def _resolve(ref):
        base = folder_paths.get_directory_by_type(ref.get("type", "temp"))
        if base is None:
            raise ValueError("bad type")
        path = os.path.abspath(os.path.join(base, os.path.normpath(ref.get("subfolder", "")), os.path.basename(ref["filename"])))
        if os.path.commonpath([path, os.path.abspath(base)]) != os.path.abspath(base):
            raise ValueError("bad path")
        return path

    def _safe(name):
        return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()[:48] or "layer"

    @routes.post("/ming_studio/export")
    async def ming_export(request):
        try:
            body = await request.json()
            fmt = body.get("format", "psd")
            prefix = _safe(body.get("prefix") or "ming_design").replace(" ", "_").strip("_") or "ming_design"
            layers = []
            for l in body.get("layers", []):
                img = np.asarray(Image.open(_resolve(l["file"])).convert("RGBA"))
                layers.append({"image": img, "name": l.get("name") or "Layer", "visible": bool(l.get("visible", True)),
                               "opacity": float(l.get("opacity", 1.0)), "blend": l.get("blend", "normal")})
            if not layers:
                return web.json_response({"error": "no layers"}, status=400)
            if fmt in ("png", "zip"):
                layers_for_png = [l for l in layers if l["visible"]] or layers
            H, W = layers[0]["image"].shape[:2]
            out_dir = os.path.join(folder_paths.get_output_directory(), "ming_studio")
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            if fmt == "psd":
                fn = "%s_%s.psd" % (prefix, stamp)
                psd_writer.write_psd(os.path.join(out_dir, fn), layers, W, H, crop=bool(body.get("crop", True)))
            elif fmt == "png":
                fn = "%s_%s.png" % (prefix, stamp)
                Image.fromarray(psd_writer.composite_layers(layers_for_png, W, H), "RGBA").save(os.path.join(out_dir, fn))
            elif fmt == "zip":
                fn = "%s_%s_layers.zip" % (prefix, stamp)
                with zipfile.ZipFile(os.path.join(out_dir, fn), "w", zipfile.ZIP_DEFLATED) as z:
                    for i, l in enumerate(layers_for_png):
                        tmp = os.path.join(out_dir, "_tmp_%s_%d.png" % (stamp, i))
                        Image.fromarray(l["image"], "RGBA").save(tmp)
                        z.write(tmp, "%02d_%s.png" % (i + 1, _safe(l["name"])))
                        os.remove(tmp)
                    tmp = os.path.join(out_dir, "_tmp_%s_c.png" % stamp)
                    Image.fromarray(psd_writer.composite_layers(layers_for_png, W, H), "RGBA").save(tmp)
                    z.write(tmp, "00_composite.png")
                    os.remove(tmp)
            else:
                return web.json_response({"error": "unknown format"}, status=400)
            return web.json_response({"filename": fn, "subfolder": "ming_studio", "type": "output"})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)


def _install_example_inputs():
    """Copy the demo image used by the example workflows into ComfyUI/input (once)."""
    try:
        import shutil

        import folder_paths
        src_dir = os.path.join(os.path.dirname(__file__), "example_inputs")
        dst_dir = folder_paths.get_input_directory()
        for name in os.listdir(src_dir):
            dst = os.path.join(dst_dir, name)
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(src_dir, name), dst)
    except Exception:
        pass


_register_routes()
_install_example_inputs()
