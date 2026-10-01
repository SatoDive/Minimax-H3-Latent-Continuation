"""Minimal, dependency-free layered PSD writer (8-bit RGB + transparency).

Writes real Photoshop layers: names (unicode), visibility, opacity, blend mode,
tight bounds (layers are cropped to their alpha content) and a merged composite,
all PackBits (RLE) compressed. Opens in Photoshop, Photopea, GIMP, Affinity and Krita.
"""

import struct

import numpy as np

BLEND_KEYS = {
    "normal": b"norm",
    "multiply": b"mul ",
    "screen": b"scrn",
    "overlay": b"over",
    "darken": b"dark",
    "lighten": b"lite",
    "color-dodge": b"div ",
    "color-burn": b"idiv",
    "soft-light": b"sLit",
    "hard-light": b"hLit",
    "difference": b"diff",
    "exclusion": b"smud",
    "hue": b"hue ",
    "saturation": b"sat ",
    "color": b"colr",
    "luminosity": b"lum ",
}


def _packbits_row(row):
    """PackBits-encode one row of uint8 values."""
    n = row.shape[0]
    if n == 0:
        return b""
    # run boundaries
    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [n])))
    out = bytearray()
    literal = bytearray()

    def flush_literal():
        i = 0
        while i < len(literal):
            chunk = literal[i:i + 128]
            out.append(len(chunk) - 1)
            out.extend(chunk)
            i += 128
        literal.clear()

    data = row.tobytes()
    for s, ln in zip(starts.tolist(), lengths.tolist()):
        if ln >= 3:
            flush_literal()
            v = data[s]
            while ln > 0:
                c = min(ln, 128)
                if c >= 2:
                    out.append((257 - c) & 0xFF)
                    out.append(v)
                else:
                    out.append(0)
                    out.append(v)
                ln -= c
        else:
            literal.extend(data[s:s + ln])
    flush_literal()
    return bytes(out)


def _rle_channel(plane):
    """Return (row_byte_counts, packed_data) for a 2D uint8 plane."""
    counts = []
    chunks = []
    for row in plane:
        packed = _packbits_row(row)
        counts.append(len(packed))
        chunks.append(packed)
    return counts, b"".join(chunks)


def _pascal_name(name, pad=4):
    raw = name.encode("latin-1", "replace")[:255]
    data = bytes([len(raw)]) + raw
    while len(data) % pad:
        data += b"\x00"
    return data


def _luni_block(name):
    text = name.encode("utf-16-be")
    body = struct.pack(">I", len(name)) + text
    while len(body) % 4:
        body += b"\x00"
    return b"8BIM" + b"luni" + struct.pack(">I", len(body)) + body


def _to_u8(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img, np.full_like(img, 255)], axis=-1)
    if img.shape[-1] == 3:
        img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    return np.ascontiguousarray(img[..., :4])


def _alpha_bbox(alpha):
    ys = np.flatnonzero(alpha.any(axis=1))
    if ys.size == 0:
        return 0, 0, 0, 0
    xs = np.flatnonzero(alpha.any(axis=0))
    return int(ys[0]), int(xs[0]), int(ys[-1]) + 1, int(xs[-1]) + 1


def composite_layers(layers, width, height):
    """Simple 'normal' blend of top-to-bottom layers (list of dicts) into RGBA uint8."""
    acc = np.zeros((height, width, 3), np.float32)
    acc_a = np.zeros((height, width, 1), np.float32)
    for layer in reversed(layers):  # bottom first
        if not layer.get("visible", True):
            continue
        px = _to_u8(layer["image"]).astype(np.float32) / 255.0
        a = px[..., 3:4] * float(layer.get("opacity", 1.0))
        acc = px[..., :3] * a + acc * (1.0 - a)
        acc_a = a + acc_a * (1.0 - a)
    rgb = np.where(acc_a > 1e-6, acc / np.maximum(acc_a, 1e-6), 0.0)
    out = np.concatenate([rgb, acc_a], axis=-1)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def write_psd(path, layers, width=None, height=None, composite=None, crop=True):
    """Write a layered PSD.

    layers: list of dicts ordered TOP-most first, each with
        image   : HxWx4 (or 3) array, float 0..1 or uint8
        name    : str
        visible : bool (default True)
        opacity : float 0..1 (default 1)
        blend   : key of BLEND_KEYS (default "normal")
        offset  : optional (top, left) of the image on the canvas
    composite: optional HxWx4 merged image; computed when omitted.
    """
    if not layers:
        raise ValueError("write_psd needs at least one layer")
    first = _to_u8(layers[0]["image"])
    if height is None or width is None:
        height, width = first.shape[:2]

    records = bytearray()
    channel_data = bytearray()
    # Photoshop stores layer records bottom-most first
    for layer in reversed(layers):
        px = _to_u8(layer["image"])
        oy, ox = layer.get("offset", (0, 0))
        if crop:
            t, l, b, r = _alpha_bbox(px[..., 3])
        else:
            t, l, b, r = 0, 0, px.shape[0], px.shape[1]
        sub = px[t:b, l:r]
        top, left, bottom, right = t + oy, l + ox, b + oy, r + ox
        chans = []
        for cid, idx in ((-1, 3), (0, 0), (1, 1), (2, 2)):
            if sub.size == 0:
                data = struct.pack(">H", 0)
            else:
                counts, packed = _rle_channel(sub[..., idx])
                data = struct.pack(">H", 1) + struct.pack(">%dH" % len(counts), *counts) + packed
            chans.append((cid, data))

        name = str(layer.get("name", "Layer"))
        opacity = int(round(max(0.0, min(1.0, float(layer.get("opacity", 1.0)))) * 255))
        flags = 0x08  # bit 3: bit 4 is meaningful
        if not layer.get("visible", True):
            flags |= 0x02
        blend = BLEND_KEYS.get(str(layer.get("blend", "normal")).lower(), b"norm")

        extra = struct.pack(">I", 0)  # layer mask data
        extra += struct.pack(">I", 0)  # blending ranges
        extra += _pascal_name(name)
        extra += _luni_block(name)

        records += struct.pack(">iiii", top, left, bottom, right)
        records += struct.pack(">H", len(chans))
        for cid, data in chans:
            records += struct.pack(">hI", cid, len(data))
        records += b"8BIM" + blend
        records += struct.pack(">BBBB", opacity, 0, flags, 0)
        records += struct.pack(">I", len(extra)) + extra
        for _, data in chans:
            channel_data += data

    layer_info = struct.pack(">h", -len(layers)) + bytes(records) + bytes(channel_data)
    if len(layer_info) % 4:
        layer_info += b"\x00" * (4 - len(layer_info) % 4)
    layer_info_block = struct.pack(">I", len(layer_info)) + layer_info
    global_mask = struct.pack(">I", 0)
    layer_mask_section = layer_info_block + global_mask

    if composite is None:
        composite = composite_layers(layers, width, height)
    comp = _to_u8(composite)
    all_counts = []
    all_data = []
    for idx in (0, 1, 2, 3):
        counts, packed = _rle_channel(comp[..., idx])
        all_counts.extend(counts)
        all_data.append(packed)

    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H", 1) + b"\x00" * 6)
        f.write(struct.pack(">HIIHH", 4, height, width, 8, 3))
        f.write(struct.pack(">I", 0))  # color mode data
        f.write(struct.pack(">I", 0))  # image resources
        f.write(struct.pack(">I", len(layer_mask_section)))
        f.write(layer_mask_section)
        f.write(struct.pack(">H", 1))
        f.write(struct.pack(">%dH" % len(all_counts), *all_counts))
        for chunk in all_data:
            f.write(chunk)
    return path
