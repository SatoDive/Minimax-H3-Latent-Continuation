"""Structured (Figma-style JSON) prompt builder for Ming-Image Design."""

import json
import re
from math import gcd

TRANSPARENT_TRIGGERS = [
    "RGBA, 4-channel, transparent background",
    "isolated subject, alpha matte, no background",
    "cutout PNG, alpha=0 outside the object",
    "transparent canvas, not white, not checkerboard",
    "production RGBA layer for compositing",
    "带透明通道，4通道RGBA图像",
    "透明背景，alpha通道，无底图",
    "抠图素材，背景alpha=0",
    "孤立主体，透明PNG图层",
    "不要白底，不要棋盘格，只要透明通道",
]

STYLES = {
    "Modern minimal": ("Clean modern minimal design, generous whitespace, crisp geometric sans-serif typography, soft shadows, 8px grid, rounded 16px corners.", ["#FFFFFF", "#F4F5F7", "#111827", "#2563EB", "#94A3B8"]),
    "Dark tech / SaaS": ("Premium dark SaaS aesthetic, near-black surfaces, luminous gradient accents, thin 1px borders, glowing highlights, Inter-like typography.", ["#0A0A0F", "#16161F", "#EDEDF2", "#7C3AED", "#22D3EE"]),
    "Glassmorphism": ("Glassmorphism: frosted translucent panels with background blur, vivid blurred gradient blobs behind, white hairline borders, soft depth.", ["#0F172A", "#6366F1", "#EC4899", "#F8FAFC", "#38BDF8"]),
    "Neo-brutalism": ("Neo-brutalist design: thick black outlines, hard offset shadows, flat saturated color blocks, chunky bold grotesk typography.", ["#FFF4E0", "#000000", "#FF5C5C", "#FFD23F", "#3EC1D3"]),
    "Swiss / International": ("Swiss International Typographic Style: strict grid, asymmetric layout, huge Helvetica-like headings, flush-left text, red accent.", ["#FFFFFF", "#111111", "#E30613", "#D9D9D9"]),
    "Retro 50s print": ("Mid-century 1950s printed aesthetic: aged cream paper, halftone grain, screen-print color planes, vintage script and slab typography.", ["#F3E9D2", "#1F4D3A", "#E76F51", "#2A9D8F", "#264653"]),
    "Pastel playful": ("Playful pastel design: soft candy colors, rounded bubbly shapes, friendly rounded typography, cute flat illustrations.", ["#FFF7F0", "#FFC8DD", "#BDE0FE", "#CDB4DB", "#3D3D5C"]),
    "Luxury editorial": ("Luxury editorial design: elegant high-contrast serif headings, thin gold rules, deep tones, generous margins, magazine layout.", ["#0E0E0E", "#F5F0E6", "#C9A227", "#5A4632"]),
    "Corporate clean": ("Trustworthy corporate design: blue and white, clear hierarchy, iconography, data-driven visuals, accessible contrast.", ["#FFFFFF", "#0B3C8C", "#1E88E5", "#F1F5F9", "#0F172A"]),
    "Cyberpunk neon": ("Cyberpunk neon: dark city tones, magenta and cyan neon glows, glitch accents, futuristic condensed typography.", ["#07020F", "#FF2BD6", "#00F0FF", "#FDE047", "#1E1B4B"]),
    "Hand-drawn / organic": ("Organic hand-drawn style: textured paper, brush lettering, watercolor washes, imperfect lines, natural earthy colors.", ["#FAF6EE", "#6B8F71", "#D9A066", "#3F3A36", "#C96F53"]),
    "3D clay": ("Soft 3D clay render style: rounded inflated shapes, matte plasticine materials, soft studio lighting, pastel gradients.", ["#F2EEFF", "#8B5CF6", "#F472B6", "#FDBA74", "#1F2937"]),
}

# slot: (kind, cx, cy, w, h)
TYPES = {
    "Landing page (web)": ("A modern website landing page UI design, full-page web layout rendered as a crisp flat screenshot", [
        ("nav", .5, .03, 1, .06), ("headline", .3, .19, .52, .12), ("subheadline", .3, .29, .5, .06),
        ("cta", .24, .37, .36, .05), ("visual", .73, .26, .46, .36), ("items", .5, .66, .9, .3), ("footer", .5, .955, 1, .09)]),
    "Mobile app screen": ("A polished native mobile app screen UI, phone-sized canvas, flat screenshot without device frame", [
        ("statusbar", .5, .015, 1, .03), ("headline", .5, .09, .88, .06), ("subheadline", .5, .14, .88, .035),
        ("visual", .5, .31, .9, .26), ("items", .5, .64, .9, .34), ("cta", .5, .86, .88, .055), ("tabbar", .5, .955, 1, .07)]),
    "Dashboard UI": ("A data-rich SaaS analytics dashboard UI, desktop web app rendered as a crisp flat screenshot", [
        ("sidebar", .08, .5, .16, 1), ("topbar", .58, .04, .84, .08), ("headline", .4, .13, .44, .05),
        ("items", .58, .26, .8, .14), ("visual", .46, .6, .56, .46), ("table", .87, .6, .22, .46), ("cta", .89, .13, .16, .05)]),
    "Poster": ("A striking graphic design poster with bold typographic hierarchy", [
        ("headline", .5, .14, .86, .16), ("subheadline", .5, .25, .8, .05), ("visual", .5, .55, .84, .48),
        ("items", .5, .85, .84, .08), ("cta", .5, .93, .5, .05)]),
    "Infographic": ("A clear, information-dense infographic with icons, numbered sections and data visualisations", [
        ("headline", .5, .07, .9, .08), ("subheadline", .5, .13, .86, .04), ("items", .5, .5, .9, .62),
        ("visual", .5, .88, .9, .14)]),
    "Social media post": ("A scroll-stopping square social media post graphic", [
        ("logo", .12, .08, .16, .08), ("headline", .5, .3, .84, .2), ("subheadline", .5, .45, .8, .07),
        ("visual", .5, .7, .8, .36), ("cta", .5, .92, .5, .07)]),
    "Presentation slide": ("A keynote presentation slide with clean layout", [
        ("headline", .32, .14, .56, .1), ("subheadline", .32, .24, .56, .05), ("items", .3, .58, .52, .52),
        ("visual", .76, .55, .42, .62), ("footer", .5, .96, 1, .05)]),
    "YouTube thumbnail": ("A high-energy YouTube thumbnail, 16:9, huge readable text, strong subject", [
        ("visual", .7, .55, .6, .9), ("headline", .3, .4, .56, .4), ("badge", .2, .82, .3, .12)]),
    "Business card": ("A premium business card design, front side, flat print layout", [
        ("logo", .2, .3, .2, .3), ("headline", .62, .35, .6, .14), ("subheadline", .62, .5, .6, .08), ("items", .62, .72, .6, .28)]),
    "Restaurant menu / flyer": ("A beautifully typeset restaurant menu flyer", [
        ("logo", .5, .07, .3, .08), ("headline", .5, .15, .8, .07), ("items", .5, .55, .86, .64), ("visual", .5, .92, .8, .1)]),
    "Product packaging label": ("A product packaging label design, flat dieline-free front panel", [
        ("logo", .5, .14, .5, .12), ("headline", .5, .3, .8, .1), ("visual", .5, .58, .7, .4), ("items", .5, .86, .8, .14)]),
    "Logo / icon (transparent)": ("A professional vector logo / app icon, centered, isolated", [
        ("visual", .5, .45, .6, .6), ("headline", .5, .85, .8, .12)]),
    "Sticker / cutout (transparent)": ("A die-cut sticker / isolated cutout asset with a clean silhouette", [
        ("visual", .5, .5, .86, .86)]),
    "Free layout (custom layers only)": ("A graphic design composition", []),
}

TRANSPARENT_TYPES = {"Logo / icon (transparent)", "Sticker / cutout (transparent)"}


def coord(cx, cy, w, h):
    return "cx: %.3f, cy: %.3f, w: %.3f, h: %.3f" % (cx, cy, w, h)


def parse_palette(text, fallback):
    found = re.findall(r"#?([0-9a-fA-F]{6})\b", text or "")
    return ["#" + c.upper() for c in found] or list(fallback)


def aspect_string(width, height):
    g = gcd(int(width), int(height)) or 1
    return "%d:%d, %d × %d px" % (width // g, height // g, width, height)


def _q(t):
    return '"%s"' % t.replace('"', "'")


def parse_custom_layers(text, palette):
    """Lines: description | cx, cy, w, h | #hex #hex   (coords and colors optional).
    Lines may also be raw JSON layer objects."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line.rstrip(","))
                if isinstance(obj, dict) and "description" in obj:
                    obj.setdefault("coordinates", coord(.5, .5, 1, 1))
                    obj.setdefault("hierarchy_and_relation", "Sits above the layers listed before it.")
                    obj.setdefault("color_specs", palette[:3])
                    out.append(obj)
                    continue
            except ValueError:
                pass
        parts = [p.strip() for p in line.split("|")]
        desc = parts[0]
        c = coord(.5, .5, 1, 1)
        colors = palette[:3]
        for p in parts[1:]:
            nums = re.findall(r"-?\d*\.?\d+", p)
            if "#" in p:
                colors = parse_palette(p, colors)
            elif len(nums) == 4:
                c = coord(*[max(0.0, min(1.0, float(n))) for n in nums])
        out.append({"description": desc, "coordinates": c,
                    "hierarchy_and_relation": "Sits above the layers listed before it.", "color_specs": colors})
    return out


def build(design_type, style, brief, brand, headline, subheadline, cta, items_text, visual,
          palette_text, width, height, transparent, custom_layers, extra_style=""):
    type_desc, slots = TYPES.get(design_type, TYPES["Free layout (custom layers only)"])
    style_desc, style_palette = STYLES.get(style, STYLES["Modern minimal"])
    palette = parse_palette(palette_text, style_palette)
    transparent = transparent or design_type in TRANSPARENT_TYPES
    items = [i.strip() for i in (items_text or "").splitlines() if i.strip()]

    image_style = "%s. %s" % (type_desc, style_desc)
    if brief.strip():
        image_style += " Brief: %s." % brief.strip().rstrip(".")
    if extra_style.strip():
        image_style += " " + extra_style.strip()
    image_style += " Color palette: %s." % ", ".join(palette)
    if transparent:
        image_style += " %s; %s." % (TRANSPARENT_TRIGGERS[0], TRANSPARENT_TRIGGERS[3])

    layers = []
    if not transparent:
        layers.append({
            "description": "Background filling the whole canvas in %s, matching the %s style." % (palette[0], style.lower()),
            "coordinates": coord(.5, .5, 1, 1),
            "hierarchy_and_relation": "Base layer: fills the whole canvas, behind every other layer.",
            "color_specs": palette[:2],
        })

    brand = brand.strip()
    used_brand = False
    for kind, cx, cy, w, h in slots:
        c = coord(cx, cy, w, h)
        d = None
        rel = "Sits above the background."
        if kind == "nav":
            d = "Top navigation bar"
            if brand:
                d += " with the logo wordmark %s on the left" % _q(brand)
                used_brand = True
            d += ", four short menu links and a small rounded sign-in button on the right."
            rel = "Pinned to the top edge of the canvas."
        elif kind in ("statusbar",):
            d = "Minimal phone status bar with time, signal and battery icons."
            rel = "Pinned to the top edge."
        elif kind == "tabbar":
            d = "Bottom tab bar with four outlined icons, the first one highlighted in %s." % (palette[3] if len(palette) > 3 else palette[-1])
            rel = "Pinned to the bottom edge."
        elif kind == "sidebar":
            d = "Left sidebar navigation"
            if brand:
                d += " topped by the logo %s" % _q(brand)
                used_brand = True
            d += ", with a vertical list of icon + label menu items and the active item highlighted."
            rel = "Full-height panel docked to the left edge."
        elif kind == "topbar":
            d = "Top bar with a search field, notification bell and round user avatar."
        elif kind == "logo":
            if brand and not used_brand:
                d = "Logo mark with the wordmark %s." % _q(brand)
                used_brand = True
            else:
                d = "Simple geometric logo mark."
        elif kind == "headline" and headline.strip():
            d = "Headline text reading %s in large bold display typography." % _q(headline.strip())
            rel = "Primary text, highest in the reading order."
        elif kind == "subheadline" and subheadline.strip():
            d = "Supporting text reading %s in a lighter weight." % _q(subheadline.strip())
            rel = "Directly below the headline, aligned with it."
        elif kind == "cta" and cta.strip():
            d = "Call-to-action button with the label %s, solid %s fill, rounded corners." % (_q(cta.strip()), palette[3] if len(palette) > 3 else palette[-1])
            rel = "Below the supporting text."
        elif kind == "badge" and (cta.strip() or subheadline.strip()):
            t = cta.strip() or subheadline.strip()
            d = "Bright badge sticker reading %s." % _q(t)
        elif kind == "visual":
            d = (visual.strip() or "Main illustration / hero visual that embodies the brief") + "."
            rel = "Main subject of the composition."
        elif kind == "footer":
            d = "Footer strip with small muted links and a copyright line"
            if brand and not used_brand:
                d += " for %s" % _q(brand)
                used_brand = True
            d += "."
            rel = "Pinned to the bottom edge."
        elif kind == "table":
            d = "Compact list panel of recent activity rows with avatars, labels and status pills."
        elif kind == "items":
            if not items:
                continue
            n = len(items)
            cols = 1 if h > w * 0.9 and n > 2 else min(n, 3 if w > .6 else 2)
            if design_type in ("Infographic", "Restaurant menu / flyer", "Presentation slide", "Business card", "Product packaging label") or (h > w and n > 2):
                cols = 1 if design_type in ("Presentation slide", "Business card", "Restaurant menu / flyer", "Product packaging label") else 2
            rows = (n + cols - 1) // cols
            cw, ch = w / cols, h / rows
            for i, text in enumerate(items):
                r, k = divmod(i, cols)
                icx = cx - w / 2 + cw * (k + .5)
                icy = cy - h / 2 + ch * (r + .5)
                layers.append({
                    "description": "Card / item block (row %d, column %d) with a small icon and the text %s." % (r + 1, k + 1, _q(text)),
                    "coordinates": coord(icx, icy, cw * .92, ch * .88),
                    "hierarchy_and_relation": "One of %d evenly spaced items in a %d-column grid." % (n, cols),
                    "color_specs": palette[1:4] or palette,
                })
            continue
        if d:
            layers.append({"description": d, "coordinates": c, "hierarchy_and_relation": rel,
                           "color_specs": palette[:4]})

    layers.extend(parse_custom_layers(custom_layers, palette))
    return json.dumps({
        "canvas_settings": {
            "aspect_ratio": aspect_string(width, height),
            "ambient_lighting": "Even, flat design lighting with soft subtle shadows." if not transparent else "Soft neutral studio light on the isolated subject, no backdrop.",
            "image_style": image_style,
        },
        "layers": layers,
    }, ensure_ascii=False, indent=2)


def parse_num_layers(text, default=5):
    """Same rule as the official infer.py."""
    m = re.search(r"decompose this image into\s+(\d+)\s+layers?", text.lower())
    if m:
        return int(m.group(1))
    for line in text.splitlines():
        s = line.strip().lower()
        if s.startswith("number of layers:"):
            try:
                return int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
    return default


def decompose_prompt(num_layers, plan_lines):
    plan = [p.strip() for p in plan_lines if p.strip()]
    if not plan:
        return "Decompose this image into %d layers." % num_layers, num_layers
    n = len(plan)
    body = "\n".join("Layer %d: %s" % (i + 1, p) for i, p in enumerate(plan))
    return ("Decompose this image into %d layers with the following specifications:\n\nNumber of layers: %d\n%s" % (n, n, body)), n


def clean_llm_output(text):
    """Strip thinking blocks / markdown fences from an LLM rewrite."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    fence = re.search(r"```(?:json|text)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    if "Decompose this image" in t:
        return t[t.index("Decompose this image"):].strip()
    if "{" in t and "}" in t:
        cand = t[t.index("{"): t.rindex("}") + 1]
        try:
            return json.dumps(json.loads(cand), ensure_ascii=False, indent=2)
        except ValueError:
            return cand
    return t
