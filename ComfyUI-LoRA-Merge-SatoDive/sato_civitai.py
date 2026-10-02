"""
LoRA info: file hash, Civitai lookup (by SHA256), local preview images, user notes.

Everything is cached in ComfyUI's user directory (user/satodive_lora_merge/), never next
to your model files:
  hashes.json        relpath -> {size, mtime, sha256}
  civitai/<sha>.json Civitai answer (or a "not found" marker)
  notes.json         relpath -> {name, strength_min, strength_max, notes}
"""

import hashlib
import json
import logging
import os
import threading
import time

import folder_paths

log = logging.getLogger("SatoDive.LoRAMerge")

API = "https://civitai.com/api/v1"
NOT_FOUND_RETRY = 24 * 3600  # don't ask Civitai again for unknown files for a day
IMAGE_EXT = (".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp", ".png", ".jpg", ".jpeg", ".webp")

_lock = threading.Lock()


def _root():
    d = os.path.join(folder_paths.get_user_directory(), "satodive_lora_merge")
    os.makedirs(os.path.join(d, "civitai"), exist_ok=True)
    return d


def _load(name, default):
    p = os.path.join(_root(), name)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(name, data):
    p = os.path.join(_root(), name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, p)


def lora_path(name):
    path = folder_paths.get_full_path("loras", name)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(name)
    return path


# --------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------

def sha256_of(name, compute=True):
    path = lora_path(name)
    st = os.stat(path)
    with _lock:
        hashes = _load("hashes.json", {})
        h = hashes.get(name)
        if h and h.get("size") == st.st_size and abs(h.get("mtime", 0) - st.st_mtime) < 1e-3:
            return h["sha256"]
    if not compute:
        return None
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            d.update(chunk)
    sha = d.hexdigest()
    with _lock:
        hashes = _load("hashes.json", {})
        hashes[name] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": sha}
        _save("hashes.json", hashes)
    return sha


# --------------------------------------------------------------------------------------
# Local sidecar files written by other tools (Civitai Helper, LoRA Manager, ...)
# --------------------------------------------------------------------------------------

def local_preview(name):
    path = lora_path(name)
    base = os.path.splitext(path)[0]
    for ext in IMAGE_EXT:
        if os.path.isfile(base + ext):
            return base + ext
    return None


def _sidecar_info(name):
    base = os.path.splitext(lora_path(name))[0]
    for ext in (".civitai.info", ".json", ".metadata.json"):
        p = base + ext
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict) and ("modelId" in d or "civitai" in d):
                    return d.get("civitai", d) if isinstance(d.get("civitai"), dict) else d
            except Exception:
                pass
    return None


# --------------------------------------------------------------------------------------
# Civitai
# --------------------------------------------------------------------------------------

def api_key():
    return os.environ.get("CIVITAI_API_KEY") or _load("settings.json", {}).get("civitai_api_key") or ""


def set_api_key(key):
    s = _load("settings.json", {})
    s["civitai_api_key"] = (key or "").strip()
    _save("settings.json", s)


def _thumb(url, width=450):
    if not url or "image.civitai.com" not in url:
        return url
    parts = url.split("/")
    # .../<uuid>/original=true/<file>  ->  .../<uuid>/width=450/<file>
    for i, p in enumerate(parts):
        if "=" in p and i == len(parts) - 2:
            parts[i] = "width={}".format(width)
            return "/".join(parts)
    return url


def _simplify(version, model=None):
    """Keep only what the UI needs."""
    images = []
    for im in version.get("images") or []:
        url = im.get("url")
        if not url:
            continue
        meta = im.get("meta") or {}
        images.append({
            "url": url,
            "thumb": _thumb(url),
            "large": _thumb(url, 1280) if im.get("type") != "video" else _thumb(url, 720),
            "type": im.get("type") or "image",
            "nsfw": int(im.get("nsfwLevel") or 1),
            "width": im.get("width"), "height": im.get("height"),
            "prompt": meta.get("prompt") or "",
            "negative": meta.get("negativePrompt") or "",
            "steps": meta.get("steps"), "cfg": meta.get("cfgScale"), "sampler": meta.get("sampler"),
            "seed": meta.get("seed"),
        })
    model = model or {}
    vm = version.get("model") or {}
    stats = model.get("stats") or version.get("stats") or {}
    return {
        "found": True,
        "model_id": version.get("modelId"),
        "version_id": version.get("id"),
        "model_name": model.get("name") or vm.get("name") or "",
        "version_name": version.get("name") or "",
        "type": model.get("type") or vm.get("type") or "LORA",
        "base_model": version.get("baseModel") or "",
        "trained_words": version.get("trainedWords") or [],
        "description": model.get("description") or "",
        "version_description": version.get("description") or "",
        "tags": (model.get("tags") or [])[:12],
        "creator": (model.get("creator") or {}).get("username", ""),
        "creator_image": (model.get("creator") or {}).get("image", ""),
        "downloads": stats.get("downloadCount"),
        "likes": stats.get("thumbsUpCount"),
        "nsfw": bool(vm.get("nsfw") or model.get("nsfw")),
        "url": "https://civitai.com/models/{}?modelVersionId={}".format(version.get("modelId"), version.get("id")),
        "images": images,
        "fetched": time.time(),
    }


async def _get_json(session, url):
    headers = {"User-Agent": "ComfyUI-SatoDive-LoRA-Merge/2.0"}
    key = api_key()
    if key:
        headers["Authorization"] = "Bearer " + key
    async with session.get(url, headers=headers, timeout=30) as r:
        if r.status == 404:
            return None
        r.raise_for_status()
        return await r.json(content_type=None)


async def fetch_civitai(sha, session):
    version = await _get_json(session, "{}/model-versions/by-hash/{}".format(API, sha))
    if version is None:
        return {"found": False, "fetched": time.time()}
    model = None
    if version.get("modelId"):
        try:
            model = await _get_json(session, "{}/models/{}".format(API, version["modelId"]))
        except Exception as e:  # model details are optional
            log.info("[SatoDive] civitai model details failed: %s", e)
    return _simplify(version, model)


def cached_civitai(sha):
    p = os.path.join(_root(), "civitai", sha + ".json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def store_civitai(sha, info):
    p = os.path.join(_root(), "civitai", sha + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)


def needs_fetch(info, refresh=False):
    if refresh or info is None:
        return True
    if not info.get("found") and time.time() - info.get("fetched", 0) > NOT_FOUND_RETRY:
        return True
    return False


# --------------------------------------------------------------------------------------
# User notes (editable fields in the info dialog)
# --------------------------------------------------------------------------------------

NOTE_FIELDS = ("name", "strength_min", "strength_max", "notes")


def get_notes(name):
    return _load("notes.json", {}).get(name, {})


def set_notes(name, data):
    with _lock:
        notes = _load("notes.json", {})
        cur = notes.get(name, {})
        for k in NOTE_FIELDS:
            if k in data:
                v = data[k]
                if k.startswith("strength"):
                    try:
                        v = None if v in ("", None) else float(v)
                    except (TypeError, ValueError):
                        v = None
                cur[k] = v
        notes[name] = {k: v for k, v in cur.items() if v not in ("", None)}
        _save("notes.json", notes)
        return notes[name]


def quick_summary(name):
    """Cheap per-file summary for the LoRA browser (no hashing, no network)."""
    sha = None
    try:
        sha = sha256_of(name, compute=False)
    except Exception:
        pass
    info = cached_civitai(sha) if sha else None
    if not (info and info.get("found")):
        side = None
        try:
            side = _sidecar_info(name)
        except Exception:
            pass
        if side:
            try:
                info = _simplify(side, side.get("model") if isinstance(side.get("model"), dict) and "name" in side.get("model") else None)
            except Exception:
                info = None
    thumb = None
    has_local = False
    try:
        has_local = local_preview(name) is not None
    except Exception:
        pass
    if info and info.get("found"):
        refresh_thumbs(info)
        imgs = sorted(info.get("images", []), key=lambda i: i.get("type") == "video")
        for im in imgs:
            if im.get("nsfw", 1) <= 1:
                thumb = {"url": im["thumb"], "type": im["type"]}
                break
        if thumb is None and imgs:
            im = imgs[0]
            thumb = {"url": im["thumb"], "type": im["type"], "nsfw": True}
    return {
        "name": name,
        "title": (info or {}).get("model_name") or "",
        "base_model": (info or {}).get("base_model") or "",
        "found": bool(info and info.get("found")),
        "checked": info is not None,
        "thumb": thumb,
        "local_preview": has_local,
        "user": get_notes(name),
    }


def refresh_thumbs(info):
    """Older cache entries: make sure every media item has small + large variants."""
    for im in (info or {}).get("images", []) or []:
        im["thumb"] = _thumb(im.get("url"))
        im["large"] = _thumb(im.get("url"), 1280 if im.get("type") != "video" else 720)
    return info


# --------------------------------------------------------------------------------------
# Image proxy cache (pictures load through ComfyUI: cached on disk, work offline later)
# --------------------------------------------------------------------------------------

ALLOWED_IMAGE_HOSTS = ("image.civitai.com",)
MAX_MEDIA_BYTES = 40 * 1024 * 1024


def media_cache_path(url):
    d = os.path.join(_root(), "media")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, hashlib.sha1(url.encode("utf-8")).hexdigest())


async def fetch_media(url, session):
    """Returns (path, content_type) - downloads once, then serves from disk."""
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if urlparse(url).scheme != "https" or host not in ALLOWED_IMAGE_HOSTS:
        raise ValueError("host not allowed")
    p = media_cache_path(url)
    if os.path.isfile(p) and os.path.isfile(p + ".type"):
        with open(p + ".type", "r") as f:
            return p, f.read().strip()
    async with session.get(url, headers={"User-Agent": "ComfyUI-SatoDive-LoRA-Merge/2.0"}, timeout=60) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "application/octet-stream").split(";")[0]
        if not (ctype.startswith("image/") or ctype.startswith("video/")):
            raise ValueError("not media: " + ctype)
        size = 0
        tmp = p + ".part"
        with open(tmp, "wb") as f:
            async for chunk in r.content.iter_chunked(256 * 1024):
                size += len(chunk)
                if size > MAX_MEDIA_BYTES:
                    raise ValueError("file too large")
                f.write(chunk)
    os.replace(tmp, p)
    with open(p + ".type", "w") as f:
        f.write(ctype)
    return p, ctype


# --------------------------------------------------------------------------------------
# Trigger words for the previews (used at execution time, so plain blocking HTTP)
# --------------------------------------------------------------------------------------

EXPLICIT_TRIGGER_KEYS = ("modelspec.trigger_phrase", "trigger_words", "ss_trigger_words", "activation text")


def _get_json_sync(url, timeout=10):
    import urllib.error
    import urllib.request
    headers = {"User-Agent": "ComfyUI-SatoDive-LoRA-Merge/2.0"}
    key = api_key()
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch_civitai_sync(sha):
    version = _get_json_sync("{}/model-versions/by-hash/{}".format(API, sha))
    if version is None:
        return {"found": False, "fetched": time.time()}
    model = None
    if version.get("modelId"):
        try:
            model = _get_json_sync("{}/models/{}".format(API, version["modelId"]))
        except Exception:
            pass
    return _simplify(version, model)


def trigger_words(name, metadata=None, allow_network=True, limit=3):
    """Words the LoRA was trained with: Civitai 'trained words' first (cached, fetched once),
    then explicit trigger fields in the file's metadata. Never the noisy caption tag counts."""
    words = []
    try:
        sha = sha256_of(name)
        info = cached_civitai(sha)
        if info is None:
            side = _sidecar_info(name)
            if side:
                info = _simplify(side, side.get("model") if isinstance(side.get("model"), dict) and "name" in side["model"] else None)
        if allow_network and needs_fetch(info):
            try:
                info = fetch_civitai_sync(sha)
                store_civitai(sha, info)
            except Exception as e:
                log.info("[SatoDive] Civitai lookup for trigger words failed (%s) - using file metadata", e)
        if info and info.get("found"):
            words = [w.strip() for w in info.get("trained_words") or [] if w and w.strip()]
    except Exception as e:
        log.info("[SatoDive] trigger word lookup failed for %s: %s", name, e)
    if not words and metadata:
        for k in EXPLICIT_TRIGGER_KEYS:
            v = metadata.get(k)
            if v:
                words += [w.strip() for w in str(v).replace(";", ",").split(",") if w.strip()]
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out[:limit]


def prompt_with_triggers(prompt, words):
    """Prepend trigger words that are not already in the prompt. Returns (new_prompt, added_words)."""
    low = (prompt or "").lower()
    added = [w for w in words if w.lower() not in low]
    if not added:
        return prompt, []
    return ", ".join(added) + (", " + prompt if prompt and prompt.strip() else ""), added
