"""Ein echtes HST-Bild einer SLACS-Linse selbst modellieren.

    python scripts/fetch_hst_slacs.py J0946+1006   # einmalig (MAST, braucht astroquery)
    python analysis/slacs_hst_fit.py J0946+1006

Fit: Sérsic-Licht + SIE + externe Scherung + Sérsic-Quelle, PSF als Gauß (FWHM 0,1″),
Startwerte aus dem SLACS-Katalog (Bolton et al. 2008). Verglichen werden θ_E, q und
Positionswinkel mit den publizierten Werten.

Hinweis zu J0946+1006 ("Jackpot", Gavazzi et al. 2008): Das System hat zwei
Quellen auf verschiedenen Rotverschiebungen und damit zwei Ringe. Modelliert wird
der innere, helle Ring; der äußere, schwache Ring (~2,1″) bleibt im Residuum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

from lenscape.catalogs import load_slacs
from lenscape.imagefit import LensImageModel, fit_lens_image, q_phi_from_e, sersic_unit
from lenscape.tikzexport import write_macros

ROOT = Path(__file__).resolve().parents[1]
HALF_SIZE = 3.0  # arcsec: Ausschnitt ±3″ um die Linse


def pixel_angle_of_pa(wcs: WCS, x: float, y: float, pa_deg: float) -> float:
    """Winkel (rad, gegen die +x-Pixelachse) der Himmelsrichtung PA (Grad östlich von Nord)."""
    c0 = wcs.pixel_to_world(x, y)
    c1 = c0.directional_offset_by(pa_deg * u.deg, 0.5 * u.arcsec)
    x1, y1 = wcs.world_to_pixel(c1)
    return float(np.arctan2(y1 - y, x1 - x))


def pa_of_pixel_angle(wcs: WCS, x: float, y: float, phi: float) -> float:
    """Umkehrung: Pixelwinkel -> Positionswinkel (Grad östlich von Nord, 0..180)."""
    c0 = wcs.pixel_to_world(x, y)
    c1 = wcs.pixel_to_world(x + np.cos(phi), y + np.sin(phi))
    return float(c0.position_angle(c1).to(u.deg).value % 180)


def stretch(img, lo, hi):
    return np.clip(np.arcsinh((img - lo) / (hi - lo) * 10) / np.arcsinh(10), 0, 1)


def run(name: str, out: Path) -> dict:
    path = ROOT / "data" / "slacs_hst" / f"{name}_F814W.fits"
    if not path.exists():
        raise SystemExit(f"{path} fehlt. Bitte: python scripts/fetch_hst_slacs.py {name}")
    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header
    wcs = WCS(hdr)
    pix = float(hdr["PIXSCALE"])
    exptime = float(hdr.get("EXPTIME", 0) or 0)

    # zentraler Ausschnitt
    ny, nx = data.shape
    n = int(round(2 * HALF_SIZE / pix)) // 2 * 2
    j0, i0 = (ny - n) // 2, (nx - n) // 2
    cut = data[j0:j0 + n, i0:i0 + n]
    xc, yc = i0 + (n - 1) / 2, j0 + (n - 1) / 2  # Mittelpunkt des Ausschnitts in Pixeln des Ausschnittsbilds

    # Rauschen: Hintergrund aus dem ganzen Ausschnittsbild (Rand), plus Poisson
    from lenscape.imagefit import estimate_noise
    sig_full = estimate_noise(data, exptime=exptime, border=12)
    sigma = sig_full[j0:j0 + n, i0:i0 + n]
    bkg = np.median(np.concatenate([data[:12].ravel(), data[-12:].ravel()]))

    cat = load_slacs()
    row = cat[cat["name"] == name][0]
    phi0 = pixel_angle_of_pa(wcs, xc, yc, float(row["pa"]))
    print(f"{name}: Katalog θE = {row['theta_E']}″, q = {row['q']}, PA = {row['pa']}° "
          f"-> Startwinkel im Bild {np.degrees(phi0):.1f}°")

    res = fit_lens_image(cut - bkg, sigma, pix, theta_E0=float(row["theta_E"]), q0=float(row["q"]),
                         phi0=phi0, psf_fwhm=0.1, supersample=2, n_starts=8, verbose=True)
    p = res.params
    q_fit, phi_fit = res.mass_q_phi
    pa_fit = pa_of_pixel_angle(wcs, xc + p["xl"] / pix, yc + p["yl"] / pix, phi_fit)
    gamma = float(np.hypot(p["g1"], p["g2"]))
    print(f"Fit: θE = {p['thetaE']:.3f}″, q = {q_fit:.3f}, PA = {pa_fit:.1f}°, γ_ext = {gamma:.3f}, "
          f"χ²_red = {res.reduced_chi2:.2f}")

    # ---------------------------------------------------------- Bilder für die Abbildung
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = np.percentile(cut - bkg, [5, 99.7])
    tag = name.replace("+", "p").replace("-", "m")
    for key, img, cmap in [("data", cut - bkg, "magma"), ("model", res.model, "magma"),
                           ("source_only", cut - bkg - res.lens_light, "magma")]:
        plt.imsave(out / f"{tag}_{key}.png", stretch(img, lo, hi)[::-1], cmap=cmap, vmin=0, vmax=1)
    plt.imsave(out / f"{tag}_resid.png", np.clip(res.residual, -5, 5)[::-1], cmap="RdBu_r", vmin=-5, vmax=5)
    # rekonstruierte Quelle (ohne Linse, ohne PSF) auf feinem Gitter
    m = LensImageModel((n, n), pix / 2, psf_fwhm=0.0001, supersample=1)
    s = sersic_unit(m.X, m.Y, p["xs"], p["ys"], p["Rs"], p["ns"], p["e1s"], p["e2s"])
    plt.imsave(out / f"{tag}_srcplane.png", (s / s.max())[::-1], cmap="magma", vmin=0, vmax=1)

    scal = {
        "fitthetaE": p["thetaE"], "fitq": q_fit, "fitpa": pa_fit, "fitgamma": gamma, "fitchi": res.reduced_chi2,
        "catthetaE": float(row["theta_E"]), "catq": float(row["q"]), "catpa": float(row["pa"]),
        "halfsize": HALF_SIZE, "srchalf": HALF_SIZE / 2,
    }
    write_macros(out / f"{tag}_macros.tex", scal)
    return {"name": name, **scal}


def main(argv):
    names = argv or ["J0946+1006"]
    out = ROOT / "figures" / "slacs_hst" / "data"
    rows = [run(n, out) for n in names]
    lines = ["# HST-Bildfit von SLACS-Linsen", "",
             "Automatisch erzeugt von `analysis/slacs_hst_fit.py` (HST/ACS F814W, Gauß-PSF 0,1″).", "",
             "| Linse | θ_E Fit | θ_E Bolton+08 | q Fit | q Bolton+08 | PA Fit | PA Bolton+08 | γ_ext | χ²_red |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['fitthetaE']:.3f}″ | {r['catthetaE']:.2f}″ | {r['fitq']:.2f} | {r['catq']:.2f} | "
                     f"{r['fitpa']:.0f}° | {r['catpa']:.0f}° | {r['fitgamma']:.3f} | {r['fitchi']:.2f} |")
    lines += ["", "![Fit](../figures/slacs_hst/hst_fit.svg)", ""]
    (ROOT / "analysis" / "slacs_hst_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
