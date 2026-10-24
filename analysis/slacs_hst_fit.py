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

    # Ausreißer maskieren: kosmische Strahlung und Detektorfehler sind schärfer als die PSF.
    # Kriterium: > 5σ über dem lokalen Median (5×5 px) oder exakt 0; Maske um 1 px erweitert.
    from scipy.ndimage import binary_dilation, median_filter
    hp = cut - median_filter(cut, size=5, mode="reflect")
    bad = (hp > 5 * sigma) | (cut == 0)
    bad = binary_dilation(bad, iterations=1)
    sig_fit = np.where(bad, np.inf, sigma)
    print(f"  maskiert (Strahlung/Defekte): {bad.sum()} von {bad.size} Pixeln")

    res = fit_lens_image(cut - bkg, sig_fit, pix, theta_E0=float(row["theta_E"]), q0=float(row["q"]),
                         phi0=phi0, psf_fwhm=0.1, supersample=2, n_starts=8, verbose=True)
    # zweiter Durchgang: verbleibende starke Ausreißer (> 6σ im Residuum, z. B. Nachbarobjekte) maskieren
    extra = binary_dilation(np.abs(res.residual) > 6, iterations=1) & ~bad
    if extra.any():
        print(f"  zweiter Durchgang: {extra.sum()} weitere Pixel maskiert")
        bad |= extra
        sig_fit = np.where(bad, np.inf, sigma)
        res = fit_lens_image(cut - bkg, sig_fit, pix, theta_E0=res.params["thetaE"], q0=res.mass_q_phi[0],
                             phi0=res.mass_q_phi[1], psf_fwhm=0.1, supersample=2, n_starts=3)
    # Vergleichsmodell exakt wie Bolton et al. (2008): SIE ohne externe Scherung
    res_sie = fit_lens_image(cut - bkg, sig_fit, pix, theta_E0=float(row["theta_E"]), q0=float(row["q"]),
                             phi0=phi0, psf_fwhm=0.1, supersample=2, n_starts=8, fixed={"g1": 0.0, "g2": 0.0})

    def describe(r):
        q_, phi_ = r.mass_q_phi
        pa_ = pa_of_pixel_angle(wcs, xc + r.params["xl"] / pix, yc + r.params["yl"] / pix, phi_)
        return r.params["thetaE"], q_, pa_, float(np.hypot(r.params["g1"], r.params["g2"])), r.reduced_chi2

    th_s, q_s, pa_s, _, chi_s = describe(res_sie)
    th_g, q_g, pa_g, gamma, chi_g = describe(res)
    print(f"SIE        : θE = {th_s:.3f}″, q = {q_s:.3f}, PA = {pa_s:.1f}°, χ²_red = {chi_s:.2f}")
    print(f"SIE+Scherung: θE = {th_g:.3f}″, q = {q_g:.3f}, PA = {pa_g:.1f}°, γ_ext = {gamma:.3f}, χ²_red = {chi_g:.2f}")
    p = res.params

    # ---------------------------------------------------------- Bilder für die Abbildung
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = np.percentile(cut - bkg, [5, 99.7])
    tag = name.replace("+", "p").replace("-", "m")
    for key, img, cmap in [("data", cut - bkg, "magma"), ("model", res.model, "magma"),
                           ("source_only", cut - bkg - res.lens_light, "magma")]:
        plt.imsave(out / f"{tag}_{key}.png", stretch(img, lo, hi)[::-1], cmap=cmap, vmin=0, vmax=1)
    resid_show = np.where(bad, np.nan, res.residual)
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#9CA3AF")  # maskierte Pixel grau
    plt.imsave(out / f"{tag}_resid.png", np.clip(resid_show, -5, 5)[::-1], cmap=cm, vmin=-5, vmax=5)
    # rekonstruierte Quelle (ohne Linse, ohne PSF) auf feinem Gitter
    m = LensImageModel((n, n), pix / 2, psf_fwhm=0.0001, supersample=1)
    s = sersic_unit(m.X, m.Y, p["xs"], p["ys"], p["Rs"], p["ns"], p["e1s"], p["e2s"])
    plt.imsave(out / f"{tag}_srcplane.png", (s / s.max())[::-1], cmap="magma", vmin=0, vmax=1)

    scal = {
        "fitthetaE": th_g, "fitq": q_g, "fitpa": pa_g, "fitgamma": gamma, "fitchi": chi_g,
        "siethetaE": th_s, "sieq": q_s, "siepa": pa_s, "siechi": chi_s,
        "catthetaE": float(row["theta_E"]), "catq": float(row["q"]), "catpa": float(row["pa"]),
        "halfsize": HALF_SIZE, "srchalf": HALF_SIZE / 2, "nmasked": int(bad.sum()),
    }
    write_macros(out / f"{tag}_macros.tex", scal)
    return {"name": name, **scal}


def main(argv):
    names = argv or ["J0946+1006"]
    out = ROOT / "figures" / "slacs_hst" / "data"
    rows = [run(n, out) for n in names]
    lines = ["# HST-Bildfit von SLACS-Linsen", "",
             "Automatisch erzeugt von `analysis/slacs_hst_fit.py` (HST/ACS F814W, Gauß-PSF 0,1″).", "",
             "| Linse | Modell | θ_E | q | PA | γ_ext | χ²_red |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | Bolton et al. 2008 (SIE) | {r['catthetaE']:.2f}″ | {r['catq']:.2f} | {r['catpa']:.0f}° | – | – |")
        lines.append(f"| | lenscape, SIE | {r['siethetaE']:.3f}″ | {r['sieq']:.2f} | {r['siepa']:.0f}° | – | {r['siechi']:.2f} |")
        lines.append(f"| | lenscape, SIE + Scherung | {r['fitthetaE']:.3f}″ | {r['fitq']:.2f} | {r['fitpa']:.0f}° | "
                     f"{r['fitgamma']:.3f} | {r['fitchi']:.2f} |")
    lines += ["",
              "Der Einsteinradius ist robust; Elliptizität und externe Scherung sind teilweise entartet "
              "(beide erzeugen ein Quadrupol-Feld). Vergleichbar mit dem Katalog ist das reine SIE.",
              "Kosmische Strahlung und Defektpixel sind maskiert (grau im Residuum).", "",
              "![Fit](../figures/slacs_hst/hst_fit.svg)", ""]
    (ROOT / "analysis" / "slacs_hst_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
