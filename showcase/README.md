# Showcase: files for the project page on the website

Upload this folder as it is (e.g. to `/projects/lenscape/` on the website). Re-create it with
`python scripts/make_showcase.py --screenshots` (after `python scripts/make_gallery.py`).

| File | Use |
|---|---|
| `demo/index.html` | the interactive photo lens (single file). Link to it directly or embed it with `<iframe src="demo/index.html?embed">`; `?embed` hides the header and the physics note |
| `thumbnail.jpg` | 600 × 600 project tile (real HST image of the "Jackpot" double ring) |
| `cover.jpg` | 1200 × 630 social preview (`og:image`) |
| `screenshots/app-*.webp` | the Streamlit app (it needs a Python server, so on a static site it can only be shown as pictures) |
| `screenshots/demo.webp` | the photo-lens demo, e.g. as a preview before the iframe is loaded |
| `figures/*.svg` | five scientific figures (TikZ/pgfplots), scale without loss |
| `description.md` | texts for the page: teaser, description, key results |
| `snippet.html` | example markup for the project section, to adapt to the site's layout |

The demo loads KaTeX for its formulas from cdn.jsdelivr.net. If the site has a strict Content Security Policy,
allow that host (or the formulas are shown as TeX source; everything else works).
