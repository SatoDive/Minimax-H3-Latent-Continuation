"""HTTP endpoints used by the node UI (info cards, slider detection, Civitai, LoRA browser)."""

import asyncio
import logging
import os

import aiohttp
from aiohttp import web

import folder_paths

from . import sato_arch as A
from . import sato_civitai as C
from . import sato_lora_core as core
from . import sato_preview as P
from . import sato_help as H

log = logging.getLogger("SatoDive.LoRAMerge")
PREFIX = "/satodive/lora_merge"


def _safe_info(name, arch):
    if not name:
        return None
    try:
        return core.inspect_lora_file(name, arch or A.AUTO)
    except Exception as e:
        return {"name": name, "error": str(e)}


def _valid_lora(name):
    return bool(name) and name in set(folder_paths.get_filename_list("loras"))


def register(server):
    routes = server.routes
    run = lambda fn, *a: asyncio.get_running_loop().run_in_executor(None, fn, *a)

    @routes.get(PREFIX + "/presets")
    async def presets(request):
        return web.json_response({
            "presets": A.public_presets(), "auto": A.AUTO,
            "samplers": P.SAMPLERS, "schedulers": P.SCHEDULERS,
            "help": H.HELP,
        })

    @routes.get(PREFIX + "/info")
    async def info(request):
        q = request.rel_url.query
        res = await run(_safe_info, q.get("name", ""), q.get("arch", A.AUTO))
        return web.json_response(res or {"error": "no lora"})

    @routes.get(PREFIX + "/groups")
    async def groups(request):
        """Union of block groups for any number of LoRA files: ?names=a.safetensors|b.safetensors&arch=..."""
        q = request.rel_url.query
        arch = q.get("arch", A.AUTO)
        names = [n for n in q.get("names", "").split("|")]
        infos = [await run(_safe_info, n, arch) if n else None for n in names]
        group_arch = arch
        if arch == A.AUTO or arch not in A.ARCH_PRESETS:
            group_arch = next((i.get("arch_used") for i in infos if i and i.get("arch_used")), A.AUTO)
        counts = [(i or {}).get("groups") or {} for i in infos]
        return web.json_response({"arch": group_arch, "groups": core.groups_payload_many(counts, group_arch), "infos": infos})

    # ---- Civitai -------------------------------------------------------------------

    @routes.get(PREFIX + "/civitai")
    async def civitai(request):
        q = request.rel_url.query
        name = q.get("name", "")
        refresh = q.get("refresh", "0") == "1"
        offline = q.get("offline", "0") == "1"
        if not _valid_lora(name):
            return web.json_response({"error": "unknown LoRA"}, status=404)
        path = C.lora_path(name)
        out = {"name": name, "file": os.path.basename(path), "size_mb": round(os.path.getsize(path) / 1048576, 1),
               "user": C.get_notes(name), "local_preview": (await run(C.local_preview, name)) is not None}
        try:
            sha = await run(C.sha256_of, name, not offline)
        except Exception as e:
            return web.json_response(dict(out, error="hash failed: {}".format(e)))
        out["sha256"] = sha
        if not sha:
            return web.json_response(out)
        info = C.cached_civitai(sha)
        if info is None:
            side = await run(C._sidecar_info, name)
            if side:
                try:
                    info = C._simplify(side, side.get("model") if isinstance(side.get("model"), dict) and "name" in side["model"] else None)
                    info["source"] = "local sidecar file"
                except Exception:
                    info = None
        if not offline and C.needs_fetch(info, refresh):
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    info = await C.fetch_civitai(sha, session)
                C.store_civitai(sha, info)
            except Exception as e:
                out["error"] = "Civitai request failed: {}".format(e)
        out["civitai"] = C.refresh_thumbs(info) if info else info
        fam = A.family_of_base_model((info or {}).get("base_model"))
        out["family"] = fam
        return web.json_response(out)

    @routes.post(PREFIX + "/notes")
    async def notes(request):
        data = await request.json()
        name = data.get("name", "")
        if not _valid_lora(name):
            return web.json_response({"error": "unknown LoRA"}, status=404)
        return web.json_response({"user": C.set_notes(name, data)})

    @routes.post(PREFIX + "/civitai_key")
    async def civitai_key(request):
        data = await request.json()
        C.set_api_key(data.get("key", ""))
        return web.json_response({"ok": True, "has_key": bool(C.api_key())})

    @routes.get(PREFIX + "/settings")
    async def settings(request):
        return web.json_response({"has_key": bool(C.api_key())})

    # ---- LoRA browser --------------------------------------------------------------

    @routes.get(PREFIX + "/lora_list")
    async def lora_list(request):
        names = folder_paths.get_filename_list("loras")

        def build():
            out = []
            for n in names:
                try:
                    out.append(C.quick_summary(n))
                except Exception as e:
                    out.append({"name": n, "error": str(e)})
            return out

        return web.json_response({"loras": await run(build)})

    @routes.get(PREFIX + "/local_preview")
    async def local_preview(request):
        name = request.rel_url.query.get("name", "")
        if not _valid_lora(name):
            return web.Response(status=404)
        p = await run(C.local_preview, name)
        if not p:
            return web.Response(status=404)
        return web.FileResponse(p, headers={"Cache-Control": "max-age=600"})

    @routes.get(PREFIX + "/img")
    async def img(request):
        url = request.rel_url.query.get("u", "")
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                path, ctype = await C.fetch_media(url, session)
        except Exception as e:
            return web.Response(status=404, text=str(e))
        return web.FileResponse(path, headers={"Content-Type": ctype, "Cache-Control": "max-age=86400"})
