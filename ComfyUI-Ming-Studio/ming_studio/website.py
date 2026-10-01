"""Website design extraction: fetch a page (or pasted HTML), read its structure,
palette and typography, optionally screenshot it, and turn it into a Ming-Image
Figma-style JSON prompt or a layer-decomposition plan."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
import urllib.request
from collections import Counter
from html.parser import HTMLParser

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

HEX_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
RGB_RE = re.compile(r"rgba?\(\s*(\d{1,3})[\s,]+(\d{1,3})[\s,]+(\d{1,3})")
FONT_RE = re.compile(r"font-family\s*:\s*([^;}{]+)", re.I)
GENERIC_FONTS = {
    "sans-serif", "serif", "monospace", "system-ui", "inherit", "initial", "cursive",
    "-apple-system", "blinkmacsystemfont", "ui-sans-serif", "ui-serif", "ui-monospace",
    "fantasy", "emoji", "unset", "var", "apple color emoji", "segoe ui emoji", "segoe ui symbol",
    "noto color emoji",
}


def _clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_url(url, timeout=20, max_bytes=4_000_000):
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(max_bytes)
        charset = resp.headers.get_content_charset() or "utf-8"
        final_url = resp.geturl()
    return raw.decode(charset, errors="replace"), final_url


class _PageParser(HTMLParser):
    SECTION_TAGS = {"header", "nav", "main", "section", "article", "aside", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta = {}
        self.stylesheets = []
        self.inline_css = []
        self.stack = []
        self.sections = []  # list of dicts
        self.current = None
        self.headings = []
        self.nav_links = []
        self.buttons = []
        self.paragraphs = []
        self.images = []
        self.logo_text = ""
        self._capture = None
        self._buf = []
        self._in_style = False
        self._in_script = False
        self._in_nav = 0
        self._in_header = 0
        self._in_title = False

    # ---- helpers
    def _start_capture(self, kind, attrs):
        self._capture = (kind, attrs)
        self._buf = []

    def _end_capture(self):
        if not self._capture:
            return
        kind, attrs = self._capture
        text = _clean("".join(self._buf))
        self._capture = None
        if not text:
            return
        if kind.startswith("h"):
            self.headings.append((kind, text))
            if self.current is not None:
                self.current["headings"].append(text)
        elif kind == "button":
            if len(text) <= 40 and text not in self.buttons:
                self.buttons.append(text)
            if self.current is not None:
                self.current["buttons"].append(text)
        elif kind == "navlink":
            if len(text) <= 30 and text not in self.nav_links:
                self.nav_links.append(text)
        elif kind == "p":
            if len(text) > 25:
                self.paragraphs.append(text)
                if self.current is not None:
                    self.current["text"].append(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "") + " " + (a.get("id") or "")
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (a.get("name") or a.get("property") or "").lower()
            if key:
                self.meta[key] = a.get("content") or ""
        elif tag == "link" and "stylesheet" in (a.get("rel") or "").lower() and a.get("href"):
            self.stylesheets.append(a["href"])
        elif tag == "style":
            self._in_style = True
        elif tag in ("script", "noscript", "svg"):
            self._in_script = True
        if a.get("style"):
            self.inline_css.append(a["style"])

        if tag in self.SECTION_TAGS or (tag == "div" and re.search(r"\b(hero|banner|section|features?|pricing|testimonials?|cta|footer|header|navbar)\b", cls, re.I)):
            if self.current is None or tag in ("header", "footer", "nav", "section"):
                self.current = {"tag": tag, "cls": _clean(cls)[:60], "headings": [], "buttons": [], "text": [], "images": 0}
                self.sections.append(self.current)
        if tag == "nav":
            self._in_nav += 1
        if tag == "header":
            self._in_header += 1

        if tag in ("h1", "h2", "h3"):
            self._start_capture(tag, a)
        elif tag == "button" or (tag == "a" and re.search(r"\b(btn|button|cta)\b", cls, re.I)):
            self._start_capture("button", a)
        elif tag == "a" and (self._in_nav or self._in_header):
            self._start_capture("navlink", a)
        elif tag == "p":
            self._start_capture("p", a)
        elif tag == "img":
            alt = _clean(a.get("alt"))
            self.images.append({"src": a.get("src") or "", "alt": alt})
            if self.current is not None:
                self.current["images"] += 1
            if not self.logo_text and re.search(r"logo", cls + " " + (a.get("src") or "") + " " + alt, re.I):
                self.logo_text = alt

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "style":
            self._in_style = False
        elif tag in ("script", "noscript", "svg"):
            self._in_script = False
        if tag == "nav":
            self._in_nav = max(0, self._in_nav - 1)
        if tag == "header":
            self._in_header = max(0, self._in_header - 1)
        if self._capture and (tag == self._capture[0] or (tag in ("a", "button") and self._capture[0] in ("button", "navlink"))):
            self._end_capture()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._in_style:
            self.inline_css.append(data)
            return
        if self._in_script:
            return
        if self._capture:
            self._buf.append(data)


def _norm_hex(h):
    h = h.lower().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h.upper()


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def extract_palette(css_text, limit=10):
    counter = Counter()
    for m in HEX_RE.finditer(css_text):
        counter[_norm_hex(m.group(0))] += 1
    for m in RGB_RE.finditer(css_text):
        r, g, b = (min(255, int(v)) for v in m.groups())
        counter["#%02X%02X%02X" % (r, g, b)] += 1
    # merge near-duplicates
    picked = []
    for color, _ in counter.most_common(80):
        rgb = _hex_to_rgb(color)
        if all(sum((a - b) ** 2 for a, b in zip(rgb, _hex_to_rgb(p))) > 900 for p in picked):
            picked.append(color)
        if len(picked) >= limit:
            break
    return picked


def extract_fonts(css_text, limit=4):
    counter = Counter()
    for m in FONT_RE.finditer(css_text):
        for name in m.group(1).split(","):
            name = name.strip().strip("'\"").strip()
            if not name or name.lower() in GENERIC_FONTS or name.lower().startswith("var("):
                continue
            counter[name] += 1
    return [f for f, _ in counter.most_common(limit)]


def luminance(hex_color):
    r, g, b = (c / 255.0 for c in _hex_to_rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def analyze(url="", html="", fetch_css=True, timeout=20):
    """Return a dict describing the page design."""
    final_url = url
    if not html and url:
        html, final_url = fetch_url(url, timeout=timeout)
    parser = _PageParser()
    parser.feed(html or "")
    css = "\n".join(parser.inline_css)
    if fetch_css and final_url:
        for href in parser.stylesheets[:4]:
            try:
                css_url = urllib.parse.urljoin(final_url, href)
                text, _ = fetch_url(css_url, timeout=timeout, max_bytes=1_500_000)
                css += "\n" + text
            except Exception:
                pass
    palette = extract_palette(css)
    theme = parser.meta.get("theme-color")
    if theme and HEX_RE.match(theme.strip()):
        t = _norm_hex(theme.strip())
        palette = [t] + [p for p in palette if p != t]
    fonts = extract_fonts(css)
    title = _clean(parser.title)
    site_name = parser.meta.get("og:site_name") or parser.logo_text or (title.split("|")[0].split("–")[0].split("-")[0].strip() if title else "")
    sections = [s for s in parser.sections if s["headings"] or s["buttons"] or s["text"] or s["images"]]
    return {
        "url": final_url,
        "title": title,
        "site_name": _clean(site_name)[:40],
        "description": _clean(parser.meta.get("description") or parser.meta.get("og:description") or ""),
        "og_image": parser.meta.get("og:image") or "",
        "palette": palette,
        "fonts": fonts,
        "nav": parser.nav_links[:7],
        "buttons": parser.buttons[:6],
        "headings": [h for _, h in parser.headings][:16],
        "h1": next((h for k, h in parser.headings if k == "h1"), ""),
        "paragraphs": parser.paragraphs[:8],
        "images": parser.images[:20],
        "sections": sections[:10],
        "dark": bool(palette) and luminance(palette[0]) < 0.35,
    }


# --------------------------------------------------------------------------- prompts

MODES = {
    "faithful clone": "Faithfully recreate the existing website design: same structure, hierarchy, palette and typography, rendered as a crisp pixel-perfect full-page UI screenshot.",
    "modern redesign": "A modern premium redesign of this website: same content and brand palette, refined spacing, bold hero, rounded cards, subtle depth, award-winning 2026 web design, rendered as a crisp full-page UI screenshot.",
    "dark mode": "A dark-mode version of this website: deep near-black surfaces, the brand colors used as luminous accents, high contrast typography, rendered as a crisp full-page UI screenshot.",
    "mobile app": "This website adapted as a native mobile app screen: single column, large touch targets, bottom tab bar, rendered as a crisp phone UI mockup on a plain canvas.",
    "minimal wireframe": "A clean grayscale lo-fi wireframe of this website layout: boxes for images, real text for headings and buttons, thin outlines, rendered flat.",
    "poster / social ad": "A bold marketing poster promoting this website and brand, big headline typography, key value propositions, call to action, product-style visuals.",
}


def _coord(cx, cy, w, h):
    return "cx: %.3f, cy: %.3f, w: %.3f, h: %.3f" % (cx, cy, w, h)


def _q(text, limit=80):
    text = _clean(text)[:limit].replace('"', "'")
    return '"%s"' % text


def build_design_prompt(info, width, height, mode="faithful clone", extra=""):
    palette = info["palette"][:8] or (["#0B0B0F", "#FFFFFF", "#6366F1"] if info.get("dark") else ["#FFFFFF", "#111827", "#2563EB"])
    fonts = ", ".join(info["fonts"]) or "a clean geometric sans-serif"
    from math import gcd
    g = gcd(width, height) or 1
    aspect = "%d:%d, %d × %d px" % (width // g, height // g, width, height)
    style = MODES.get(mode, MODES["faithful clone"])
    style += " Typography: %s. Brand palette: %s." % (fonts, ", ".join(palette))
    if extra:
        style += " " + extra.strip()

    layers = []
    bg = palette[0]
    layers.append({
        "description": "Full page background in %s%s." % (bg, " with soft dark gradients" if info.get("dark") else " with subtle neutral section bands"),
        "coordinates": _coord(0.5, 0.5, 1.0, 1.0),
        "hierarchy_and_relation": "Base layer behind every other layer.",
        "color_specs": palette[:3],
    })

    tall = height / max(width, 1) > 1.2
    y = 0.0
    nav_h = 0.035 if tall else 0.07
    brand = info["site_name"] or "Brand"
    nav_desc = "Top navigation bar: logo wordmark %s on the left" % _q(brand, 30)
    if info["nav"]:
        nav_desc += ", menu items " + ", ".join(_q(n, 20) for n in info["nav"][:6])
    if info["buttons"]:
        nav_desc += ", and a pill button %s on the right" % _q(info["buttons"][0], 24)
    layers.append({
        "description": nav_desc + ".",
        "coordinates": _coord(0.5, nav_h / 2, 1.0, nav_h),
        "hierarchy_and_relation": "Pinned to the top edge above the hero.",
        "color_specs": palette[:3],
    })
    y = nav_h

    hero_h = 0.30 if tall else 0.62
    h1 = info["h1"] or info["title"] or brand
    hero = "Hero section with the large headline %s" % _q(h1, 90)
    if info["description"]:
        hero += ", a supporting sentence %s" % _q(info["description"], 140)
    btns = info["buttons"][1:3] if info["buttons"] else []
    if btns:
        hero += ", call-to-action buttons " + " and ".join(_q(b, 24) for b in btns)
    hero += ", and a large product visual / illustration beside it"
    layers.append({
        "description": hero + ".",
        "coordinates": _coord(0.5, y + hero_h / 2, 1.0, hero_h),
        "hierarchy_and_relation": "Directly below the navigation bar, first section of the page.",
        "color_specs": palette[:4],
    })
    y += hero_h

    used = {h1}
    remaining = 1.0 - y - (0.09 if tall else 0.0)
    body_sections = []
    for s in info["sections"]:
        heads = [h for h in s["headings"] if h not in used]
        if not heads:
            continue
        used.update(heads)
        body_sections.append((heads, s))
    if tall and body_sections:
        body_sections = body_sections[:5]
        sec_h = remaining / len(body_sections)
        for heads, s in body_sections:
            d = "Content section titled %s" % _q(heads[0], 60)
            if len(heads) > 1:
                d += " containing cards headed " + ", ".join(_q(h, 36) for h in heads[1:5])
            if s["text"]:
                d += ", short body copy about %s" % _clean(s["text"][0])[:90].rstrip(". ")
            if s["images"]:
                d += ", with %d image%s" % (min(s["images"], 6), "s" if s["images"] > 1 else "")
            layers.append({
                "description": d + ".",
                "coordinates": _coord(0.5, y + sec_h / 2, 0.92, sec_h * 0.92),
                "hierarchy_and_relation": "Stacked vertically below the previous section, centered in the page column.",
                "color_specs": palette[:4],
            })
            y += sec_h
    if tall:
        layers.append({
            "description": "Footer with the wordmark %s, link columns and a copyright line." % _q(brand, 30),
            "coordinates": _coord(0.5, 1.0 - 0.045, 1.0, 0.09),
            "hierarchy_and_relation": "Pinned to the bottom edge of the page.",
            "color_specs": palette[:3],
        })

    return json.dumps({
        "canvas_settings": {
            "aspect_ratio": aspect,
            "ambient_lighting": "Even flat screen lighting, no camera perspective, no device frame.",
            "image_style": style,
        },
        "layers": layers,
    }, ensure_ascii=False, indent=2)


def build_decompose_plan(info, num_layers):
    """Layer plan for decomposing a screenshot of the site (layer 1 = front-most)."""
    items = []
    brand = info["site_name"] or "the brand"
    if info["h1"]:
        items.append("Hero headline text reading %s and its supporting copy." % _q(info["h1"], 80))
    if info["buttons"]:
        items.append("Call-to-action buttons " + ", ".join(_q(b, 24) for b in info["buttons"][:3]) + ".")
    items.append("Top navigation bar with the %s logo and menu links." % brand)
    items.append("Main hero image / product illustration.")
    for s in info["sections"]:
        if s["headings"] and s["headings"][0] != info["h1"] and len(items) < num_layers - 1:
            items.append("Section %s with its cards and text." % _q(s["headings"][0], 50))
    items = items[: max(0, num_layers - 1)]
    while len(items) < num_layers - 1:
        items.append("Remaining UI cards, icons and decorative elements.")
    items.append("Page background and section color bands.")
    return items


# --------------------------------------------------------------------------- screenshot

def _find_chrome():
    names = ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome", "msedge", "microsoft-edge"]
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/opt/pw-browsers/chromium",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def screenshot(url, width=1440, height=2400, full_page=True, timeout=45):
    """Return a PIL image of the page or raise. Tries Playwright, then a headless Chrome/Edge binary."""
    from PIL import Image

    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    errors = []
    result = {}

    def _pw():
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": width, "height": min(height, 1200)}, device_scale_factor=1)
                page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                page.wait_for_timeout(800)
                fd, path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                page.screenshot(path=path, full_page=full_page)
                browser.close()
                result["path"] = path
        except Exception as e:  # noqa: BLE001
            errors.append("playwright: %s" % e)

    # Playwright's sync API refuses to run inside an asyncio loop, so use a thread.
    t = threading.Thread(target=_pw, daemon=True)
    t.start()
    t.join(timeout + 15)
    if "path" not in result:
        chrome = _find_chrome()
        if chrome:
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cmd = [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-sandbox",
                   "--window-size=%d,%d" % (width, height), "--screenshot=%s" % path, url]
            try:
                subprocess.run(cmd, timeout=timeout, capture_output=True)
                if os.path.getsize(path) > 0:
                    result["path"] = path
            except Exception as e:  # noqa: BLE001
                errors.append("chrome: %s" % e)
        else:
            errors.append("no Chrome/Edge/Chromium found")
    if "path" not in result:
        raise RuntimeError("; ".join(errors) or "screenshot failed")
    img = Image.open(result["path"]).convert("RGB")
    img.load()
    try:
        os.remove(result["path"])
    except OSError:
        pass
    if full_page and img.height > height:
        img = img.crop((0, 0, img.width, height))
    return img
