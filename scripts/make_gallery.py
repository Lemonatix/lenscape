#!/usr/bin/env python3
"""Render the image gallery in gallery/: square, high-resolution pictures (e.g. for a profile picture).

    python scripts/make_gallery.py          # all images
    python scripts/make_gallery.py sim_     # only images whose name starts with "sim_"

Three kinds of pictures:

* real data: the HST/ACS F814W image of SDSS J0946+1006 ("Jackpot", data/slacs_hst/),
  with and without the lens galaxy,
* models: the lenscape model of J0946+1006 (lens + both pixelated sources from analysis/jackpot.py,
  figures/jackpot/data/model.npz) ray-traced at much higher resolution than the data,
* simulations: Einstein ring, Einstein cross, a double source plane and a black hole in front of a
  star field, plus a Frontier Fields mass map (needs data/frontier/, see scripts/fetch_frontier.py).

All images are SIZE x SIZE pixels and centred on the lens, so they also survive a round crop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter, zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lenscape.imagefit import LensImageModel, estimate_noise, q_phi_from_e, sersic_unit  # noqa: E402
from lenscape.lenses import SIE, SIS, ExternalShear, PointMass  # noqa: E402
from lenscape.raytrace import make_grid  # noqa: E402
from lenscape.sources import SersicSource  # noqa: E402

OUT = ROOT / "gallery"
SIZE = 1024
HST = ROOT / "data" / "slacs_hst" / "J0946+1006_F814W.fits"
JACKPOT_MODEL = ROOT / "figures" / "jackpot" / "data" / "model.npz"
HALF = 3.4  # arcsec, half width of the J0946 images (the outer ring has a radius of ~2.1″)


# ---------------------------------------------------------------- helpers
def asinh_stretch(img, lo, hi, soft=10.0):
    return np.clip(np.arcsinh((img - lo) / (hi - lo) * soft) / np.arcsinh(soft), 0, 1)


def save(name: str, img: np.ndarray, cmap: str | None = None) -> None:
    """``img``: (SIZE, SIZE) in [0, 1] with a colour map, or (SIZE, SIZE, 3) RGB; row 0 = bottom."""
    OUT.mkdir(exist_ok=True)
    img = np.clip(img, 0, 1)[::-1]
    if cmap is None:
        plt.imsave(OUT / f"{name}.png", img)
    else:
        plt.imsave(OUT / f"{name}.png", img, cmap=cmap, vmin=0, vmax=1)
    print(f"  gallery/{name}.png")


def supersampled(fn, half: float, n: int = SIZE, ss: int = 2):
    """Evaluate ``fn(X, Y)`` on an ss×ss supersampled n×n grid and average back to n×n."""
    X, Y = make_grid(half, n * ss)
    v = fn(X, Y)
    return v.reshape(n, ss, n, ss, *v.shape[2:]).mean(axis=(1, 3))


def tint(gray: np.ndarray, rgb) -> np.ndarray:
    return gray[..., None] * np.asarray(rgb, float)[None, None, :]


def screen(*layers) -> np.ndarray:
    """Combine light layers like slides in a projector ("screen" blending)."""
    out = np.zeros_like(layers[0])
    for layer in layers:
        out = 1 - (1 - out) * (1 - np.clip(layer, 0, 1))
    return out


# ---------------------------------------------------------------- real data: HST J0946+1006
def load_hst():
    from astropy.io import fits

    data = fits.getdata(HST).astype(float)
    hdr = fits.getheader(HST)
    pix = float(hdr["PIXSCALE"])
    sigma = estimate_noise(data, exptime=float(hdr.get("EXPTIME", 0) or 0), border=12)
    bkg = np.median(np.concatenate([data[:12].ravel(), data[-12:].ravel()]))
    ny, nx = data.shape
    n = int(round(2 * HALF / pix)) // 2 * 2
    j0, i0 = (ny - n) // 2, (nx - n) // 2
    cut = data[j0:j0 + n, i0:i0 + n] - bkg
    sig = sigma[j0:j0 + n, i0:i0 + n]
    # cosmic rays and defects: replace by the local median (only for the pictures)
    med = median_filter(cut, size=5, mode="reflect")
    bad = (cut - med > 5 * sig) | (cut == -bkg)
    cut = np.where(bad, med, cut)
    return cut, pix


def hst_pictures():
    if not HST.exists():
        print("  (skipped: no HST cutout, run scripts/fetch_hst_slacs.py)")
        return
    cut, pix = load_hst()
    lo, hi = np.percentile(cut, [5, 99.7])
    big = zoom(cut, SIZE / cut.shape[0], order=3)
    save("hst_j0946_f814w_magma", asinh_stretch(big, lo, hi), "magma")
    save("hst_j0946_f814w_grey", asinh_stretch(big, lo, hi), "gray")

    if JACKPOT_MODEL.exists():
        m = np.load(JACKPOT_MODEL)
        model = LensImageModel(cut.shape, pix, psf_fwhm=0.1, supersample=2)
        light = float(m["light_amp"]) * model._convolve(sersic_unit(model.X, model.Y, *m["light"]))
        rings = cut - light
        # the light model is not perfect in the very centre: fade out the inner 0.5″
        n = cut.shape[0]
        yy, xx = (np.mgrid[0:n, 0:n] - (n - 1) / 2) * pix
        r = np.hypot(xx - m["light"][0], yy - m["light"][1])
        rings = rings * np.clip((r - 0.35) / 0.3, 0, 1)
        lo2, hi2 = np.percentile(rings, [50, 99.5])  # background (median) black
        save("hst_j0946_rings_magma", asinh_stretch(zoom(rings, SIZE / cut.shape[0], order=3), lo2, hi2), "magma")


# ---------------------------------------------------------------- model: J0946+1006 at high resolution
def jackpot_model_pictures():
    if not JACKPOT_MODEL.exists():
        print("  (skipped: no figures/jackpot/data/model.npz, run analysis/jackpot.py)")
        return
    m = np.load(JACKPOT_MODEL)
    th, e1, e2, g1, g2 = m["lens"]
    xl, yl = m["center"]
    q, phi = q_phi_from_e(e1, e2)
    lens = SIE(th, q=q, phi=phi, x0=xl, y0=yl) + ExternalShear(g1, g2)
    eta = float(m["eta"])

    def sample(src, grid, bx, by, floor, smooth=0.8):
        """Bicubic source lookup. The reconstruction also contains noise everywhere on its grid; for a
        clean picture the source is smoothed slightly and everything below ``floor`` × max is dropped."""
        n, half, x0, y0 = grid
        step = 2 * half / (n - 1)
        u = (bx - (x0 - half)) / step
        v = (by - (y0 - half)) / step
        s = gaussian_filter(np.clip(src, 0, None), smooth)
        s = np.clip(s - floor * s.max(), 0, None)
        return map_coordinates(s, [v, u], order=3, mode="constant", cval=0.0)

    def rings(X, Y):
        ax, ay = lens.deflection(X, Y)
        r1 = sample(m["source1"], m["grid1"], X - ax, Y - ay, floor=0.15)
        r2 = sample(m["source2"], m["grid2"], X - eta * ax, Y - eta * ay, floor=0.25, smooth=1.5)
        return np.stack([r1, r2], axis=-1)

    # centre the picture on the lens
    r = supersampled(lambda X, Y: rings(X + xl, Y + yl), HALF)
    r1, r2 = np.clip(r[..., 0], 0, None), np.clip(r[..., 1], 0, None)
    tot = r1 + r2
    lo, hi = 0.0, np.percentile(tot, 99.7)
    save("jackpot_model_magma", asinh_stretch(tot, lo, hi), "magma")

    # colour-coded: inner ring (z = 0.609) warm, outer ring (z = 2.035) blue
    a = asinh_stretch(r1, 0, np.percentile(r1, 99.7))
    b = asinh_stretch(r2, 0, np.percentile(r2, 99.8), soft=20)
    X, Y = make_grid(HALF, SIZE)
    galaxy = sersic_unit(X + xl, Y + yl, *m["light"])
    gal = asinh_stretch(galaxy, 0, np.percentile(galaxy, 99.9), soft=30) * 0.55
    save("jackpot_model_two_planes",
         screen(tint(a, (1.0, 0.62, 0.25)), tint(b, (0.35, 0.65, 1.0)), tint(gal, (1.0, 0.93, 0.82))))


# ---------------------------------------------------------------- simulations
def render_sersic(lens, sources, half):
    def f(X, Y):
        bx, by = lens.ray_shoot(X, Y)
        return sum(s.brightness(bx, by) for s in sources)
    return supersampled(f, half)


def sim_pictures():
    # 1) Einstein ring: source almost exactly behind an isothermal galaxy
    img = render_sersic(SIS(1.0), [SersicSource(0.03, 0.015, R_e=0.10, n=1.0, q=0.8, phi=0.6)], 1.7)
    save("sim_einstein_ring", asinh_stretch(img, 0, np.percentile(img, 99.8)), "inferno")

    # 2) Einstein cross: elliptical galaxy, source slightly off-centre
    cross = SIE(1.0, q=0.6, phi=0.6)
    img = render_sersic(cross, [SersicSource(0.025, 0.012, R_e=0.03, n=1.0, q=0.8, phi=1.0)], 1.7)
    save("sim_einstein_cross", asinh_stretch(img, 0, np.percentile(img, 99.8)), "inferno")

    # 3) giant arc: the source straddles a caustic cusp
    lens = SIE(1.0, q=0.72, phi=0.5) + ExternalShear(0.03, 0.0)
    img = render_sersic(lens, [SersicSource(0.16, 0.02, R_e=0.09, n=1.0, q=0.6, phi=0.2)], 1.7)
    save("sim_giant_arc", asinh_stretch(img, 0, np.percentile(img, 99.8)), "magma")

    # 4) double source plane: two sources behind each other, the farther one lensed more strongly
    lens = SIE(1.0, q=0.85, phi=0.3)
    s1 = SersicSource(0.05, 0.02, R_e=0.08, n=1.0, q=0.7, phi=0.4)
    s2 = SersicSource(-0.04, 0.05, R_e=0.07, n=1.0, q=0.8, phi=1.2)
    eta = 1.45

    def two(X, Y):
        ax, ay = lens.deflection(X, Y)
        return np.stack([s1.brightness(X - ax, Y - ay), s2.brightness(X - eta * ax, Y - eta * ay)], -1)

    r = supersampled(two, 2.0)
    a = asinh_stretch(r[..., 0], 0, np.percentile(r[..., 0], 99.8))
    b = asinh_stretch(r[..., 1], 0, np.percentile(r[..., 1], 99.8))
    X, Y = make_grid(2.0, SIZE)
    gal = sersic_unit(X, Y, 0, 0, 0.6, 4.0, 0.08, 0.0)
    gal = asinh_stretch(gal, 0, np.percentile(gal, 99.95), soft=40) * 0.6
    save("sim_double_source_plane",
         screen(tint(a, (1.0, 0.6, 0.22)), tint(b, (0.35, 0.66, 1.0)), tint(gal, (1.0, 0.93, 0.8))))

    # 5) black hole in front of a star field (weak-field lens equation + dark shadow)
    stars = star_field(1400, seed=7)
    bh = PointMass(1.0)
    half = 2.6
    tex_half = 2.6  # the texture covers ±tex_half and repeats beyond

    def lensed(X, Y):
        bx, by = bh.ray_shoot(X, Y)
        u = (bx + tex_half) / (2 * tex_half) * stars.shape[1] - 0.5
        v = (by + tex_half) / (2 * tex_half) * stars.shape[0] - 0.5
        rgb = np.stack([map_coordinates(stars[..., c], [v, u], order=1, mode="grid-wrap") for c in range(3)], -1)
        r = np.hypot(X, Y)
        shadow = np.clip((r - 0.36) / 0.04, 0, 1)  # soft edge
        return rgb * shadow[..., None]

    img = supersampled(lensed, half, ss=3)
    save("sim_black_hole_star_field", img)


def star_field(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic RGB star field with a few small background galaxies, values in [0, 1]."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n, 3))
    yy, xx = np.mgrid[0:n, 0:n]
    # stars: points with a slight colour spread, blurred to a few pixels
    # faint, patchy nebula: makes the distortion visible everywhere
    neb = gaussian_filter(rng.normal(size=(n, n)), n / 25, mode="wrap")
    neb2 = gaussian_filter(rng.normal(size=(n, n)), n / 12, mode="wrap")
    neb = np.clip((neb - neb.mean()) / neb.std(), 0, None) * 0.10
    neb2 = np.clip((neb2 - neb2.mean()) / neb2.std(), 0, None) * 0.08
    img += neb[..., None] * np.array([0.45, 0.35, 0.9]) + neb2[..., None] * np.array([0.9, 0.4, 0.5])
    k = 1800
    x, y = rng.integers(0, n, k), rng.integers(0, n, k)
    b = rng.pareto(2.0, k) * 0.6 + 0.35
    temp = rng.uniform(0, 1, k)
    col = np.stack([0.85 + 0.15 * temp, 0.85 + 0.1 * np.ones(k), 1.0 - 0.3 * temp], -1)
    pts = np.zeros((n, n, 3))
    np.add.at(pts, (y, x), b[:, None] * col)
    img += gaussian_filter(pts, (1.3, 1.3, 0), mode="wrap") * 12
    # galaxies: small elliptical blobs
    for _ in range(30):
        x0, y0 = rng.uniform(0, n, 2)
        a, q, ang = rng.uniform(5, 16), rng.uniform(0.25, 0.8), rng.uniform(0, np.pi)
        c, s = np.cos(ang), np.sin(ang)
        dx, dy = (xx - x0 + n / 2) % n - n / 2, (yy - y0 + n / 2) % n - n / 2
        r2 = ((dx * c + dy * s) / a) ** 2 + ((-dx * s + dy * c) / (a * q)) ** 2
        hue = np.array([1.0, 0.75, 0.45]) if rng.uniform() < 0.5 else np.array([0.55, 0.7, 1.0])
        img += np.exp(-np.sqrt(r2) * 2.5)[..., None] * hue * rng.uniform(0.5, 1.0)
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- Frontier Fields: Abell 2744
def frontier_pictures():
    folder = ROOT / "data" / "frontier" / "abell2744" / "cats"
    if not folder.exists():
        print("  (skipped: no Frontier Fields maps, run scripts/fetch_frontier.py)")
        return
    sys.path.insert(0, str(ROOT / "analysis"))
    import contourpy
    from frontier import central_peak, find, load

    from lenscape.cosmology import LensGeometry, make_cosmology

    m = load(folder)
    cx, cy = m.sky_to_offset(central_peak(find(folder, "_kappa.fits")))
    half = 55.0
    ratio = m.at_source_redshift(LensGeometry(0.308, 2.0, make_cosmology(H0=70, Om0=0.3))).ratio
    X, Y = make_grid(half, SIZE)
    kap = ratio * m.convergence(X + cx, Y + cy)
    img = (np.log10(np.clip(kap, 0.05, 5)) - np.log10(0.05)) / 2.0

    # critical curves (z_s = 2) from the deflection maps, on a 0.25″ grid
    Xc, Yc = make_grid(half, 440)
    ax, ay = m.deflection(Xc + cx, Yc + cy)
    ax, ay = ratio * ax, ratio * ay  # m is normalized to D_ds/D_s = 1
    d = 2 * half / 439
    axy, axx = np.gradient(ax, d)
    ayy, ayx = np.gradient(ay, d)
    det = (1 - axx) * (1 - ayy) - axy * ayx
    fig = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100)
    a = fig.add_axes([0, 0, 1, 1])
    a.imshow(img, origin="lower", extent=[-half, half, -half, half], cmap="magma", vmin=0, vmax=1,
             interpolation="bicubic")
    for line in contourpy.contour_generator(Xc, Yc, det).lines(0.0):
        if len(line) > 8:
            a.plot(line[:, 0], line[:, 1], color="white", lw=1.1, alpha=0.85)
    a.set_xlim(-half, half)
    a.set_ylim(-half, half)
    a.axis("off")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "frontier_abell2744_kappa.png", dpi=100, facecolor="black")
    plt.close(fig)
    print("  gallery/frontier_abell2744_kappa.png")


def main(argv: list[str]) -> int:
    prefix = argv[0] if argv else ""
    groups = {"hst_": hst_pictures, "jackpot_": jackpot_model_pictures, "sim_": sim_pictures,
              "frontier_": frontier_pictures}
    for key, fn in groups.items():
        if key.startswith(prefix) or prefix.startswith(key):
            print(key.rstrip("_") + ":")
            fn()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
