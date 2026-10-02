"""
Core LoRA logic for the SatoDive LoRA Merge pack.

Merging happens in *model weight space*: both LoRAs are resolved through ComfyUI's own
LoRA key maps (comfy.lora.model_lora_keys_unet / _clip), so a kohya LoRA can be merged
with a diffusers / PEFT / comfy LoRA of the same model, including LoRAs that address
fused weights (qkv, gate_up) through slices.
"""

import json
import logging
import math
import os
import time
from collections import OrderedDict

import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
import comfy.weight_adapter as weight_adapter
import folder_paths

from . import sato_arch as A

log = logging.getLogger("SatoDive.LoRAMerge")

# --------------------------------------------------------------------------------------
# Bundles passed between nodes (custom type SATO_LORA / SATO_PIPE)
# --------------------------------------------------------------------------------------

class SatoLoRA:
    def __init__(self, name, sd, metadata=None, path=None, arch=A.AUTO, strength_model=1.0,
                 strength_clip=1.0, info=None, merged=False, recipe=None):
        self.name = name
        self.sd = sd
        self.metadata = metadata or {}
        self.path = path
        self.arch = arch
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        self.info = info or {}
        self.merged = merged
        self.recipe = recipe or {}

    def __repr__(self):
        return "SatoLoRA({}, {} keys)".format(self.name, len(self.sd))


class SatoPipe(dict):
    """model / clip / vae / arch / sampling settings (a dict so it is easy to extend)."""


# --------------------------------------------------------------------------------------
# File loading + caches
# --------------------------------------------------------------------------------------

_SD_CACHE = OrderedDict()
_SD_CACHE_MAX = 4
_INFO_CACHE = OrderedDict()


def lora_path(name):
    return folder_paths.get_full_path_or_raise("loras", name)


def load_lora_file(name):
    path = lora_path(name)
    mtime = os.path.getmtime(path)
    hit = _SD_CACHE.get(path)
    if hit is not None and hit[0] == mtime:
        _SD_CACHE.move_to_end(path)
        return path, hit[1], hit[2]
    sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    _SD_CACHE[path] = (mtime, sd, meta or {})
    while len(_SD_CACHE) > _SD_CACHE_MAX:
        _SD_CACHE.popitem(last=False)
    return path, sd, meta or {}


def _read_header(path):
    """keys, shapes, alphas, dtype, metadata without loading the full file (safetensors)."""
    if path.lower().endswith(".safetensors"):
        from safetensors import safe_open
        shapes, alphas, dtypes = {}, {}, set()
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            keys = list(f.keys())
            for k in keys:
                sl = f.get_slice(k)
                shapes[k] = tuple(sl.get_shape())
                try:
                    dtypes.add(str(sl.get_dtype()))
                except Exception:
                    pass
                if k.endswith(".alpha"):
                    try:
                        alphas[k] = float(f.get_tensor(k).float().reshape(-1)[0].item())
                    except Exception:
                        pass
        return keys, shapes, alphas, dtypes, meta
    sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    shapes = {k: tuple(v.shape) for k, v in sd.items()}
    alphas = {k: float(v.float().reshape(-1)[0].item()) for k, v in sd.items() if k.endswith(".alpha")}
    dtypes = {str(v.dtype) for v in sd.values()}
    return list(sd.keys()), shapes, alphas, dtypes, meta or {}


# --------------------------------------------------------------------------------------
# Analysis (used by the info cards + slider panel)
# --------------------------------------------------------------------------------------

_DOWN_MARKERS = (".lora_down.weight", ".lora_A.weight", ".lora.down.weight", "_lora.down.weight",
                 ".lora_linear_layer.down.weight", ".lora_A.default.weight", ".lora_A")
_UP_MARKERS = (".lora_up.weight", ".lora_B.weight", ".lora.up.weight", "_lora.up.weight",
               ".lora_linear_layer.up.weight", ".lora_B.default.weight", ".lora_B")


def _trigger_words(meta):
    words = []
    for k in ("modelspec.trigger_phrase", "trigger_words", "ss_trigger_words", "activation text", "satodive_trigger_words"):
        v = meta.get(k)
        if v:
            words += [w.strip() for w in str(v).replace(";", ",").split(",") if w.strip()]
    tf = meta.get("ss_tag_frequency")
    if tf:
        try:
            freq = {}
            for _, tags in json.loads(tf).items():
                for t, c in tags.items():
                    freq[t.strip()] = freq.get(t.strip(), 0) + int(c)
            words += [t for t, _ in sorted(freq.items(), key=lambda x: -x[1])[:6]]
        except Exception:
            pass
    seen, out = set(), []
    for w in words:
        if w and w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out[:10]


def analyze_keys(keys, shapes, alphas, meta, arch_name=A.AUTO, dtypes=None, size_bytes=None):
    detected = A.detect_arch(keys)
    use_arch = arch_name if arch_name in A.ARCH_PRESETS and arch_name != A.AUTO else detected
    preset = A.get_preset(use_arch)

    modules = {}
    types = set()
    for k in keys:
        path = A.module_path(k)
        m = modules.setdefault(path, {"rank": None, "alpha": None, "type": None})
        if k.endswith(".alpha"):
            m["alpha"] = alphas.get(k)
            continue
        if any(k.endswith(s) for s in _DOWN_MARKERS):
            m["type"] = m["type"] or "lora"
            shp = shapes.get(k)
            if shp:
                m["rank"] = int(shp[0])
        elif any(k.endswith(s) for s in _UP_MARKERS):
            m["type"] = m["type"] or "lora"
        elif ".hada_" in k:
            m["type"] = "loha"
        elif ".lokr_" in k:
            m["type"] = "lokr"
        elif k.endswith(".diff") or k.endswith(".diff_b"):
            m["type"] = m["type"] or "full-diff"
        elif k.endswith(".dora_scale"):
            types.add("dora")
        if m["type"]:
            types.add(m["type"])

    modules = {p: m for p, m in modules.items() if m["type"]}
    ranks = sorted({m["rank"] for m in modules.values() if m["rank"]})
    alpha_vals = sorted({round(m["alpha"], 4) for m in modules.values() if m["alpha"] is not None})

    groups = {}
    te_modules = 0
    for p, m in modules.items():
        g = A.group_of_path(p, preset)
        if g == A.TE_GROUP:
            te_modules += 1
        groups[g] = groups.get(g, 0) + 1

    fmt = "unknown"
    ks = keys[0] if keys else ""
    if any(k.startswith("lora_unet_") or k.startswith("lora_te") for k in keys):
        fmt = "kohya"
    elif any(k.startswith("diffusion_model.") for k in keys):
        fmt = "comfy"
    elif any(".lora_A." in k or ".lora_B." in k for k in keys):
        fmt = "diffusers / peft"
    elif any(k.startswith("lycoris_") for k in keys):
        fmt = "lycoris"
    elif ks:
        fmt = "custom"

    base = meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or meta.get("ss_sd_model_name") or ""
    recipe = meta.get("satodive_recipe")

    return {
        "keys": len(keys),
        "modules": len(modules),
        "te_modules": te_modules,
        "types": sorted(types),
        "ranks": ranks,
        "alphas": alpha_vals,
        "format": fmt,
        "dtypes": sorted(dtypes or []),
        "size_mb": round(size_bytes / (1024 * 1024), 1) if size_bytes else None,
        "detected_arch": detected,
        "arch_used": use_arch,
        "base_model_meta": base,
        "trigger_words": _trigger_words(meta),
        "title": meta.get("modelspec.title") or meta.get("ss_output_name") or "",
        "satodive_recipe": json.loads(recipe) if recipe else None,
        "groups": groups,
    }


def inspect_lora_file(name, arch_name=A.AUTO):
    path = lora_path(name)
    st = os.stat(path)
    ck = (path, st.st_mtime, arch_name)
    if ck in _INFO_CACHE:
        return _INFO_CACHE[ck]
    keys, shapes, alphas, dtypes, meta = _read_header(path)
    info = analyze_keys(keys, shapes, alphas, meta, arch_name, dtypes, st.st_size)
    info["name"] = name
    _INFO_CACHE[ck] = info
    while len(_INFO_CACHE) > 64:
        _INFO_CACHE.popitem(last=False)
    return info


def inspect_state_dict(sd, meta, arch_name=A.AUTO):
    shapes = {k: tuple(v.shape) for k, v in sd.items()}
    alphas = {k: float(v.float().reshape(-1)[0].item()) for k, v in sd.items() if k.endswith(".alpha")}
    size = sum(v.numel() * v.element_size() for v in sd.values())
    return analyze_keys(list(sd.keys()), shapes, alphas, meta or {}, arch_name, {str(v.dtype) for v in sd.values()}, size)


def groups_payload_many(counts_list, arch_name):
    """Ordered slider groups for any number of LoRAs: counts[i] = modules LoRA i has in the group."""
    preset = A.get_preset(arch_name)
    allg = set()
    for c in counts_list:
        allg |= set(c)
    out = []
    for g in sorted(allg, key=lambda g: A.group_sort_key(g, preset)):
        section, label = A.group_label(g, preset)
        counts = [int(c.get(g, 0)) for c in counts_list]
        out.append({"id": g, "section": section, "label": label, "counts": counts})
    return out


# --------------------------------------------------------------------------------------
# Resolving a LoRA against a model -> weight deltas
# --------------------------------------------------------------------------------------

class Delta:
    """delta = up @ down (low rank, 2D) or a dense tensor with the full weight shape."""
    __slots__ = ("up", "down", "dense", "shape")

    def __init__(self, shape, up=None, down=None, dense=None):
        self.shape = tuple(shape)
        self.up = up
        self.down = down
        self.dense = dense

    @property
    def lowrank(self):
        return self.up is not None

    @property
    def rank(self):
        return self.up.shape[1] if self.up is not None else 0

    def to_dense(self, device=None):
        if self.dense is not None:
            return self.dense.to(device) if device is not None else self.dense
        up = self.up.to(device) if device is not None else self.up
        down = self.down.to(device) if device is not None else self.down
        return (up @ down).reshape(self.shape)


def build_key_map(model, clip):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    return key_map


def build_shape_lookup(model, clip):
    shapes = {}
    if model is not None:
        for k, v in model.model.state_dict().items():
            try:
                shapes[k] = tuple(v.shape)
            except Exception:
                pass
    if clip is not None:
        for k, v in clip.cond_stage_model.state_dict().items():
            try:
                shapes[k] = tuple(v.shape)
            except Exception:
                pass
    return shapes


def _identity(a):
    return a


def _parse_target(target):
    """ComfyUI patch targets: 'key' or (key, offset, function?) with offset = (dim, start, size) or None."""
    if isinstance(target, tuple):
        key = target[0]
        offset = target[1] if len(target) > 1 else None
        func = target[2] if len(target) > 2 else None
        return key, offset, func
    return target, None, None


def _commutes_with_rows(func, U, D):
    """True if func(U @ D) == func(U) @ D (row permutations such as swap_scale_shift)."""
    try:
        probe_d = D[:, : min(D.shape[1], 64)]
        a = func(U @ probe_d)
        b = func(U) @ probe_d
        return a.shape == b.shape and torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    except Exception:
        return False


def extract_deltas(lora_sd, key_map, shapes):
    """Resolve a LoRA against the model. Returns ({weight_key: [Delta, ...]}, stats)."""
    sd = comfy.lora_convert.convert_lora(lora_sd)
    patches = comfy.lora.load_lora(sd, key_map, log_missing=False)
    out = {}
    stats = {"matched": 0, "skipped": [], "dora_ignored": 0}

    for target, patch in patches.items():
        key, offset, func = _parse_target(target)
        full_shape = shapes.get(key)
        if full_shape is None:
            stats["skipped"].append(str(key))
            continue
        dim, start, size = offset if offset is not None else (None, 0, 0)
        slice_shape = list(full_shape)
        if dim is not None:
            slice_shape[dim] = size

        delta = None
        # ---- plain LoRA: keep it low rank ------------------------------------------------
        if isinstance(patch, weight_adapter.LoRAAdapter):
            up, down, alpha, mid, dora, reshape = patch.weights
            if mid is None and reshape is None and len(full_shape) >= 2 and dim in (None, 0):
                r = down.shape[0]
                scale = (alpha / r) if alpha is not None else 1.0
                U = up.flatten(start_dim=1).float()
                D = down.flatten(start_dim=1).float() * scale
                ok = D.shape[1] == math.prod(full_shape[1:]) and U.shape[0] == slice_shape[0]
                if ok and func is not None:
                    if _commutes_with_rows(func, U, D):
                        U = func(U)
                    else:
                        ok = False
                if ok:
                    if dora is not None:
                        stats["dora_ignored"] += 1
                    if dim is not None:
                        Uf = torch.zeros((full_shape[0], U.shape[1]), dtype=torch.float32)
                        Uf[start:start + size] = U
                        U = Uf
                    delta = Delta(full_shape, up=U, down=D)

        # ---- anything else: materialise a dense delta ------------------------------------
        if delta is None:
            try:
                if isinstance(patch, weight_adapter.WeightAdapterBase):
                    if getattr(patch, "name", "") in ("oft", "boft"):
                        stats["skipped"].append("{} ({} is multiplicative)".format(key, patch.name))
                        continue
                    zero = torch.zeros(slice_shape, dtype=torch.float32)
                    dense = patch.calculate_weight(zero, key, 1.0, 1.0, None, func or _identity,
                                                   intermediate_dtype=torch.float32).float()
                elif isinstance(patch, tuple) and len(patch) == 2 and patch[0] == "diff":
                    dense = patch[1][0].float()
                    if func is not None:
                        dense = func(dense)
                else:
                    stats["skipped"].append("{} (unsupported patch)".format(key))
                    continue
            except Exception as e:
                stats["skipped"].append("{} ({})".format(key, e))
                continue
            if tuple(dense.shape) != tuple(slice_shape):
                if dense.numel() == math.prod(slice_shape):
                    dense = dense.reshape(slice_shape)
                else:
                    stats["skipped"].append("{} (shape mismatch)".format(key))
                    continue
            if dim is not None:
                full = torch.zeros(full_shape, dtype=torch.float32)
                full.narrow(dim, start, size).copy_(dense)
                dense = full
            delta = Delta(full_shape, dense=dense)

        out.setdefault(key, []).append(delta)
        stats["matched"] += 1

    return out, stats


def count_matches(lora_sd, key_map, shapes):
    """Cheap check used by the slot nodes: how many LoRA modules land on a real model weight."""
    patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora_sd), key_map, log_missing=False)
    matched = sum(1 for t in patches if _parse_target(t)[0] in shapes)
    return matched, len(patches) - matched


def apply_lora(model, clip, lora_sd, strength_model, strength_clip):
    """Like comfy.sd.load_lora_for_models, but quiet: one summary line instead of a warning per key."""
    key_map = build_key_map(model, clip)
    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora_sd), key_map, log_missing=False)
    m2 = c2 = None
    k = k1 = set()
    if model is not None:
        m2 = model.clone()
        k = set(m2.add_patches(loaded, strength_model))
    if clip is not None:
        c2 = clip.clone()
        k1 = set(c2.add_patches(loaded, strength_clip))
    skipped = sum(1 for x in loaded if x not in k and x not in k1)
    if skipped:
        log.info("[SatoDive] %d LoRA weights did not apply to this model", skipped)
    return m2 if model is not None else None, c2 if clip is not None else None


def delta_groups(deltas, preset):
    groups = {}
    for k in deltas:
        g = A.group_of_model_key(k, preset)
        groups[g] = groups.get(g, 0) + 1
    return groups


# --------------------------------------------------------------------------------------
# Merge math
# --------------------------------------------------------------------------------------

def lowrank_resize(U, D, r):
    """Exact truncated SVD of U @ D without materialising it (QR trick)."""
    k = U.shape[1]
    if k <= r:
        return U, D, 1.0
    Qu, Ru = torch.linalg.qr(U)
    Qd, Rd = torch.linalg.qr(D.T)
    Us, S, Vh = torch.linalg.svd(Ru @ Rd.T, full_matrices=False)
    r = max(1, min(r, S.shape[0]))
    total = float((S ** 2).sum())
    kept = float((S[:r] ** 2).sum()) / total if total > 0 else 1.0
    s = S[:r].sqrt()
    return (Qu @ Us[:, :r]) * s, s[:, None] * (Vh[:r] @ Qd.T), kept


def dense_to_lowrank(M, r):
    M2 = M.reshape(M.shape[0], -1)
    m, n = M2.shape
    r = max(1, min(r, m, n))
    if r >= min(m, n) * 0.5 or min(m, n) <= 256:
        U, S, Vh = torch.linalg.svd(M2, full_matrices=False)
        V = Vh.T
    else:
        U, S, V = torch.svd_lowrank(M2, q=min(r + 12, m, n), niter=4)
    total = float((M2 ** 2).sum())
    kept = float((S[:r] ** 2).sum()) / total if total > 0 else 1.0
    s = S[:r].sqrt()
    return U[:, :r] * s, (V[:, :r] * s).T, min(kept, 1.0)


def _ties(deltas, density, drop_random=False, gen=None, use_sign=True):
    """deltas: list of dense tensors (already weighted)."""
    trimmed = []
    for d in deltas:
        if drop_random:
            if density < 1.0:
                mask = torch.rand(d.shape, generator=gen, device="cpu").to(d.device) < density
                d = d * mask / max(density, 1e-6)
        elif density < 1.0:
            flat = d.abs().flatten()
            k = max(1, int(flat.numel() * density))
            thr = torch.topk(flat, k, sorted=False).values.min()
            d = d * (d.abs() >= thr)
        trimmed.append(d)
    stack = torch.stack(trimmed)
    if not use_sign:
        return stack.sum(0)
    sign = torch.sign(stack.sum(0))
    agree = (torch.sign(stack) == sign) & (stack != 0)
    num = (stack * agree).sum(0)
    cnt = agree.sum(0).clamp(min=1)
    return num / cnt


def _method_key(method):
    """Friendly UI label -> internal method name."""
    m = (method or "").lower()
    if "ties" in m and "dare" in m:
        return "dare_ties"
    if "bold" in m:
        return "dare_ties"
    if "dare" in m or "soft" in m:
        return "dare_linear"
    if "ties" in m or "smart" in m:
        return "ties"
    if "magnitude" in m or "strongest" in m:
        return "magnitude"
    if "svd" in m or "smaller" in m:
        return "svd"
    return "add"


def merge_many(deltas_list, preset, wfuns, method, rank=0, density=0.5, seed=0, te_mode="merge",
               merged_scale=1.0, device=None, ranks=None):
    """Merge any number of resolved LoRAs.

    deltas_list: [ {weight_key: [Delta, ...]}, ... ] one dict per LoRA
    wfuns:       [ group -> multiplier ] one per LoRA (global weight x per-block slider x slot strength)
    Returns ({weight_key: Delta}, report).
    """
    device = device or comfy.model_management.get_torch_device()
    mkey = _method_key(method)
    n = len(deltas_list)
    keys = list(dict.fromkeys(k for d in deltas_list for k in d.keys()))
    out = {}
    kept_vals = []
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    ranks = ranks or [0] * n
    auto_rank = max(ranks + [4])
    pbar = comfy.utils.ProgressBar(max(1, len(keys)))

    for key in keys:
        comfy.model_management.throw_exception_if_processing_interrupted()
        g = A.group_of_model_key(key, preset)
        ws = [float(f(g)) for f in wfuns]
        if g == A.TE_GROUP:
            if te_mode == "drop":
                ws = [0.0] * n
            elif te_mode == "first only":
                ws = [w if i == 0 else 0.0 for i, w in enumerate(ws)]
        terms = []  # (lora index, weight, [Delta])
        for i, d in enumerate(deltas_list):
            if ws[i] != 0 and key in d:
                terms.append((i, ws[i], d[key]))
        if not terms:
            pbar.update(1)
            continue
        shape = terms[0][2][0].shape
        all_lowrank = all(x.lowrank for _, _, ds in terms for x in ds) and len(shape) >= 2
        several = len(terms) > 1
        dense_method = mkey in ("ties", "dare_ties", "dare_linear", "magnitude") and several

        if all_lowrank and not dense_method:
            ups = [x.up.to(device) * w for _, w, ds in terms for x in ds]
            downs = [x.down.to(device) for _, _, ds in terms for x in ds]
            U = torch.cat(ups, dim=1) * merged_scale
            D = torch.cat(downs, dim=0)
            # auto rank is per weight: keep the capacity of the biggest LoRA at this weight
            # (a fused qkv hit by three rank-r slices really is rank 3r)
            key_rank = rank if rank > 0 else max(max(sum(x.rank for x in ds) for _, _, ds in terms), 1)
            if mkey == "svd" or (mkey != "add" and U.shape[1] > key_rank):
                U, D, kept = lowrank_resize(U, D, key_rank)
                kept_vals.append(kept)
            out[key] = Delta(shape, up=U.cpu(), down=D.cpu())
            pbar.update(1)
            continue

        # ---- dense path ----
        dense = [sum(x.to_dense(device) for x in ds) * w for _, w, ds in terms]
        if several:
            if mkey == "ties":
                M = _ties(dense, density)
            elif mkey == "dare_ties":
                M = _ties(dense, density, drop_random=True, gen=gen)
            elif mkey == "dare_linear":
                M = _ties(dense, density, drop_random=True, gen=gen, use_sign=False)
            elif mkey == "magnitude":
                stack = torch.stack(dense)
                idx = stack.abs().argmax(dim=0, keepdim=True)
                M = torch.gather(stack, 0, idx)[0]
            else:
                M = sum(dense)
        else:
            M = dense[0]
        M = M * merged_scale

        if M.ndim >= 2:
            rk = [sum(x.rank if x.lowrank else auto_rank for x in ds) for _, _, ds in terms]
            r = rank if rank > 0 else (sum(rk) if dense_method else max(rk))
            U, D, kept = dense_to_lowrank(M.float(), max(1, r))
            kept_vals.append(kept)
            out[key] = Delta(shape, up=U.cpu(), down=D.cpu())
        else:
            out[key] = Delta(shape, dense=M.cpu())
        pbar.update(1)

    out_ranks = sorted({d.rank for d in out.values() if d.lowrank})
    report = {
        "modules": len(out),
        "ranks": out_ranks,
        "energy_kept": round(100.0 * sum(kept_vals) / len(kept_vals), 2) if kept_vals else 100.0,
        "resized": len(kept_vals),
        "method": mkey,
    }
    return out, report


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------

def deltas_to_state_dict(deltas, key_style="comfy", dtype=torch.float16):
    sd = {}
    for key, d in deltas.items():
        is_bias = key.endswith(".bias")
        if key.endswith(".weight"):
            base = key[:-len(".weight")]
        elif is_bias:
            base = key[:-len(".bias")]
        else:
            base = key
        if base.startswith("diffusion_model."):
            prefix = "lora_unet_" + base[len("diffusion_model."):].replace(".", "_") if key_style.startswith("kohya") else base
        else:
            prefix = "text_encoders." + base
        if d.lowrank and not is_bias:
            r = d.up.shape[1]
            sd[prefix + ".lora_up.weight"] = d.up.contiguous().to(dtype)
            sd[prefix + ".lora_down.weight"] = d.down.contiguous().to(dtype)
            sd[prefix + ".alpha"] = torch.tensor(float(r), dtype=dtype)
        else:
            dense = d.to_dense()
            sd[prefix + (".diff_b" if is_bias else ".diff")] = dense.contiguous().to(dtype)
    return sd


def unique_lora_save_path(filename, subfolder="SatoDive"):
    root = folder_paths.get_folder_paths("loras")[0]
    folder = os.path.join(root, subfolder) if subfolder else root
    os.makedirs(folder, exist_ok=True)
    base = "".join(c for c in filename if c.isalnum() or c in "-_ .").strip() or "SatoDive_merge"
    if base.lower().endswith(".safetensors"):
        base = base[:-len(".safetensors")]
    i = 1
    while True:
        p = os.path.join(folder, "{}_{:03d}.safetensors".format(base, i))
        if not os.path.exists(p):
            return p
        i += 1


def save_lora(sd, path, metadata):
    meta = {str(k): (v if isinstance(v, str) else json.dumps(v)) for k, v in metadata.items()}
    comfy.utils.save_torch_file(sd, path, metadata=meta)
    return path


def build_metadata(arch_name, recipe, rank_list, trigger_words):
    return {
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(max(rank_list) if rank_list else 0),
        "ss_network_alpha": str(max(rank_list) if rank_list else 0),
        "ss_base_model_version": A.get_preset(arch_name)["key"],
        "modelspec.architecture": A.get_preset(arch_name)["key"] + "/lora",
        "modelspec.title": recipe.get("output_name", "SatoDive merge"),
        "modelspec.date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "satodive_trigger_words": ", ".join(trigger_words),
        "satodive_recipe": json.dumps(recipe),
        "satodive_tool": "SatoDive LoRA Merge Studio",
    }
