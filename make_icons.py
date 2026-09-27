"""Generate the PWA icons (run once in CI after model build)."""
from PIL import Image, ImageDraw, ImageFont
import os

def make_icon(size, path):
    img = Image.new("RGB", (size, size), "#0b1020")
    d = ImageDraw.Draw(img)
    r = int(size * 0.22)
    for y in range(size):
        t = y / size
        cr = int(0x6c + (0x22 - 0x6c) * t)
        cg = int(0x8c + (0xd3 - 0x8c) * t)
        cb = int(0xff + (0xee - 0xff) * t)
        d.line([(0, y), (size, y)], fill=(cr, cg, cb))
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    out = Image.new("RGB", (size, size), "#0b1020")
    out.paste(img, (0, 0), mask)
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size * 0.36))
    except Exception:
        font = ImageFont.load_default()
    tb = d.textbbox((0, 0), "TG", font=font)
    d.text(((size - (tb[2] - tb[0])) / 2 - tb[0], (size - (tb[3] - tb[1])) / 2 - tb[1]), "TG",
           fill="#071022", font=font)
    out.save(path)
    print("wrote", path)

make_icon(192, "icon-192.png")
make_icon(512, "icon-512.png")
