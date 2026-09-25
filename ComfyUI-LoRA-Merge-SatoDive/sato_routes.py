"""HTTP endpoints used by the node UI (info cards + live slider detection)."""

import asyncio
import logging

from aiohttp import web

from . import sato_arch as A
from . import sato_lora_core as core
from . import sato_preview as P

log = logging.getLogger("SatoDive.LoRAMerge")


def _safe_info(name, arch):
    if not name:
        return None
    try:
        return core.inspect_lora_file(name, arch or A.AUTO)
    except Exception as e:
        return {"name": name, "error": str(e)}


def register(server):
    routes = server.routes

    @routes.get("/satodive/lora_merge/presets")
    async def presets(request):
        return web.json_response({"presets": A.public_presets(), "auto": A.AUTO,
                                  "samplers": P.SAMPLERS, "schedulers": P.SCHEDULERS})

    @routes.get("/satodive/lora_merge/info")
    async def info(request):
        name = request.rel_url.query.get("name", "")
        arch = request.rel_url.query.get("arch", A.AUTO)
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, _safe_info, name, arch)
        return web.json_response(res or {"error": "no lora"})

    @routes.get("/satodive/lora_merge/groups")
    async def groups(request):
        q = request.rel_url.query
        arch = q.get("arch", A.AUTO)
        loop = asyncio.get_running_loop()
        ia = await loop.run_in_executor(None, _safe_info, q.get("a", ""), arch)
        ib = await loop.run_in_executor(None, _safe_info, q.get("b", ""), arch)
        group_arch = arch
        if arch == A.AUTO or arch not in A.ARCH_PRESETS:
            group_arch = (ia or {}).get("arch_used") or (ib or {}).get("arch_used") or A.AUTO
        ga = (ia or {}).get("groups") or {}
        gb = (ib or {}).get("groups") or {}
        return web.json_response({
            "arch": group_arch,
            "groups": core.groups_payload(ga, gb, group_arch),
            "a": ia, "b": ib,
        })
