#!/usr/bin/env python3
"""Collect everything for the project page on the website in showcase/.

    python scripts/make_showcase.py                 # copy demo + figures, make cover/thumbnail
    python scripts/make_showcase.py --screenshots   # additionally screenshot the Streamlit app and the demo
                                                    # (needs: pip install playwright && playwright install chromium)

The folder is meant to be uploaded as it is (see showcase/README.md). Run scripts/make_gallery.py first;
cover and thumbnail are cut from gallery pictures.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "showcase"
GALLERY = ROOT / "gallery"

FIGURES = [  # (source, published name)
    ("figures/slacs_hst/hst_fit.svg", "hst-fit.svg"),
    ("figures/jackpot/jackpot_fit.svg", "jackpot-two-source-planes.svg"),
    ("figures/h0/h0_tension.svg", "h0-tension.svg"),
    ("figures/slacs/cosmography.svg", "slacs-cosmography.svg"),
    ("figures/frontier/frontier_maps.svg", "frontier-fields-maps.svg"),
]


def copy_demo() -> None:
    (OUT / "demo").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "web" / "index.html", OUT / "demo" / "index.html")
    print("  showcase/demo/index.html")


def copy_figures() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for src, name in FIGURES:
        shutil.copy(ROOT / src, OUT / "figures" / name)
        print(f"  showcase/figures/{name}")


def crop_resize(src: Path, size: tuple[int, int]) -> Image.Image:
    """Centre crop to the aspect ratio of ``size``, then resize."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    tw, th = size
    if w / h > tw / th:
        nw = round(h * tw / th)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = round(w * th / tw)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im.resize(size, Image.LANCZOS)


def cover_and_thumbnail() -> None:
    # Open Graph / social preview: 1200 × 630
    crop_resize(GALLERY / "sim_black_hole_star_field.png", (1200, 630)).save(OUT / "cover.jpg", quality=90)
    # square project tile: the real HST image of the "Jackpot" double ring
    crop_resize(GALLERY / "hst_j0946_f814w_magma.png", (600, 600)).save(OUT / "thumbnail.jpg", quality=90)
    print("  showcase/cover.jpg\n  showcase/thumbnail.jpg")


def screenshots() -> None:
    from playwright.sync_api import sync_playwright

    shots = OUT / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    port = 8611
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                            "--server.headless", "true", "--server.port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(4)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
            page.goto(f"http://localhost:{port}", wait_until="networkidle")
            page.wait_for_selector("text=Gravitational-lens simulator", timeout=120_000)
            page.wait_for_timeout(10_000)
            page.add_style_tag(content='[data-testid="stToolbar"], [data-testid="stHeader"] { display: none; }')

            def shot(name):
                path = shots / f"{name}.png"
                page.screenshot(path=str(path))
                Image.open(path).convert("RGB").save(path.with_suffix(".webp"), quality=88)
                path.unlink()
                print(f"  showcase/screenshots/{name}.webp")

            shot("app-simulator")
            if page.get_by_text("Real lens (SLACS)").count():
                page.get_by_text("Real lens (SLACS)").click()
                page.wait_for_timeout(10_000)
                shot("app-real-lens")
                for tab, name, wait in [("SLACS sample", "app-slacs", 6_000),
                                        ("H₀ from time delays", "app-h0", 25_000)]:
                    if page.get_by_role("tab", name=tab).count():
                        page.get_by_role("tab", name=tab).click()
                        page.wait_for_timeout(wait)
                        page.mouse.move(900, 500)
                        page.mouse.wheel(0, 380)  # scroll the main area down to the plots
                        page.wait_for_timeout(800)
                        shot(name)
                        page.mouse.wheel(0, -2000)
                        page.wait_for_timeout(500)

            # the photo-lens demo; its KaTeX comes from a CDN. Without network access, point
            # LENSCAPE_KATEX_DIR to a local katex/dist folder (npm i katex@0.16.11) to serve it from there.
            katex = os.environ.get("LENSCAPE_KATEX_DIR")
            if katex:
                page.route("https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/**",
                           lambda r: r.fulfill(path=str(Path(katex) / r.request.url.split("/dist/")[1])))
            page.goto((OUT / "demo" / "index.html").as_uri(), wait_until="load")
            page.wait_for_timeout(3_000)
            page.click('button[data-scn="slowsun"]')
            page.wait_for_timeout(1_500)
            shot("demo")
            browser.close()
    finally:
        app.terminate()


def main(argv: list[str]) -> int:
    print("showcase:")
    copy_demo()
    copy_figures()
    cover_and_thumbnail()
    if "--screenshots" in argv:
        screenshots()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
