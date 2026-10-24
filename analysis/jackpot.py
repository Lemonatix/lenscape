"""SDSS J0946+1006 ("Jackpot"): zwei Quellebenen -> Kosmologie.

    python analysis/jackpot.py

1. Parametrischer Fit (Sérsic-Licht, SIE + Scherung) -> Linsenlicht und Startmodell
2. Innerer Ring (Quelle 1, z = 0,609): pixelierte Quelle, Linsenparameter über die
   Bayes-Evidenz (Suyu et al. 2006)
3. Äußerer Ring (Quelle 2, z = 2,035; Smith & Collett 2021): dieselbe Linse, aber für
   die fernere Quelle um den Faktor

       η = (D_ds2 / D_s2) / (D_ds1 / D_s1)

   stärker ablenkend; optional lenkt Quelle 1 (als SIS) das Licht von Quelle 2 zusätzlich ab
   (Mehrfachebenen-Linsengleichung). η hängt nur von Ωₘ und w ab, nicht von H₀.

Vergleich: Collett & Auger (2014) messen 1/β = η = 1,404 ± 0,016 (mit Masse von Quelle 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slacs_hst_fit import pixel_angle_of_pa, stretch  # noqa: E402

from lenscape.catalogs import load_slacs  # noqa: E402
from lenscape.cosmography import model_distance_ratio  # noqa: E402
from lenscape.imagefit import estimate_noise, fit_lens_image, gaussian_psf, q_phi_from_e  # noqa: E402
from lenscape.lenses import SIE, SIS, ExternalShear  # noqa: E402
from lenscape.pixsrc import SourceGrid, SourceInverter, fit_lens_by_evidence  # noqa: E402
from lenscape.tikzexport import write_macros, write_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "jackpot" / "data"
NAME = "J0946+1006"
Z_D, Z_S1, Z_S2 = 0.222, 0.609, 2.035   # Linse/Quelle 1: Bolton+08; Quelle 2: Smith & Collett (2021)
HALF = 3.4                               # Ausschnitt ±3,4″
PIX_PSF_FWHM = 0.1


def eta_model(Om, w):
    return model_distance_ratio(Z_D, Z_S2, Om, w) / model_distance_ratio(Z_D, Z_S1, Om, w)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data" / "slacs_hst" / f"{NAME}_F814W.fits"
    with fits.open(path) as h:
        data = np.asarray(h[0].data, float)
        hdr = h[0].header
    wcs = WCS(hdr)
    pix = float(hdr["PIXSCALE"])
    exptime = float(hdr.get("EXPTIME", 0) or 0)
    ny, nx = data.shape
    n = int(round(2 * HALF / pix)) // 2 * 2
    j0, i0 = (ny - n) // 2, (nx - n) // 2
    cut = data[j0:j0 + n, i0:i0 + n]
    sigma = estimate_noise(data, exptime=exptime, border=12)[j0:j0 + n, i0:i0 + n]
    bkg = np.median(np.concatenate([data[:12].ravel(), data[-12:].ravel()]))
    img = cut - bkg
    xc, yc = i0 + (n - 1) / 2, j0 + (n - 1) / 2

    # Ausreißer (Strahlung, Defekte) maskieren
    hp = cut - median_filter(cut, size=5, mode="reflect")
    bad = binary_dilation((hp > 5 * sigma) | (cut == 0), iterations=1)
    sig = np.where(bad, np.inf, sigma)

    cat = load_slacs()
    row = cat[cat["name"] == NAME][0]
    phi0 = pixel_angle_of_pa(wcs, xc, yc, float(row["pa"]))

    # ------------------------------------------------ 1. parametrisch (Licht + Startmodell)
    print("1) parametrischer Fit …")
    par = fit_lens_image(img, sig, pix, theta_E0=float(row["theta_E"]), q0=float(row["q"]), phi0=phi0,
                         psf_fwhm=PIX_PSF_FWHM, supersample=2, n_starts=6)
    p = par.params
    xl, yl = p["xl"], p["yl"]
    print(f"   θE = {p['thetaE']:.3f}″, Lichtzentrum ({xl:.2f}, {yl:.2f})″, χ²_red = {par.reduced_chi2:.2f}")
    resid_light = img - par.lens_light - par.amplitudes["background"]

    yy, xx = (np.mgrid[0:n, 0:n] - (n - 1) / 2) * pix
    rr = np.hypot(xx - xl, yy - yl)
    psf = gaussian_psf(PIX_PSF_FWHM, pix)

    def main_lens(q):
        qq, ph = q_phi_from_e(q["e1m"], q["e2m"])
        return SIE(q["thetaE"], q=qq, phi=ph, x0=xl, y0=yl) + ExternalShear(q["g1"], q["g2"])

    # ------------------------------------------------ 2. innerer Ring: pixelierte Quelle
    print("2) innerer Ring, pixelierte Quelle …")
    mask1 = (rr > 0.75) & (rr < 1.9) & ~bad
    g1 = SourceGrid(n=36, half=0.55, x0=p["xs"], y0=p["ys"])
    inv1 = SourceInverter(resid_light, sigma, mask1, pix, psf, g1, supersample=2, reg="gradient")
    lens_p = {k: p[k] for k in ("thetaE", "e1m", "e2m", "g1", "g2")}
    lens_p, res1 = fit_lens_by_evidence(inv1, main_lens, lens_p, ["thetaE", "e1m", "e2m", "g1", "g2"],
                                        scales={"thetaE": 0.01, "e1m": 0.02, "e2m": 0.02, "g1": 0.01, "g2": 0.01},
                                        n_rounds=2, verbose=True)
    q1, ph1 = q_phi_from_e(lens_p["e1m"], lens_p["e2m"])
    print(f"   θE = {lens_p['thetaE']:.4f}″, q = {q1:.3f}, γ = {np.hypot(lens_p['g1'], lens_p['g2']):.3f}, "
          f"χ²_red = {res1.reduced_chi2:.2f}, λ = {res1.lam:.3g}")
    lens1 = main_lens(lens_p)

    # Schwerpunkt der rekonstruierten Quelle 1 (Lage der "Linse" für Quelle 2)
    gx, gy = g1.axes
    s1 = np.clip(res1.source, 0, None)
    GX, GY = np.meshgrid(gx, gy)
    s1x, s1y = float((GX * s1).sum() / s1.sum()), float((GY * s1).sum() / s1.sum())

    # ------------------------------------------------ 3. äußerer Ring: zweite Quellebene
    print("3) äußerer Ring, zweite Quellebene …")
    resid2 = resid_light - res1.model
    mask2 = (rr > 1.85) & (rr < 2.9) & ~bad & (np.abs(xx) < HALF - 0.1) & (np.abs(yy) < HALF - 0.1)

    def ray_s2(eta, b_s1):
        def f(x, y):
            ax, ay = lens1.deflection(x, y)
            # Ebene 1: Position, an der der Strahl die Quelle 1 passiert
            x1, y1 = x - ax, y - ay
            bx, by = x - eta * ax, y - eta * ay
            if b_s1 > 0:
                sx, sy = SIS(b_s1, x0=s1x, y0=s1y).deflection(x1, y1)
                bx, by = bx - sx, by - sy
            return bx, by
        return f

    g2 = SourceGrid(n=36, half=0.9, x0=0.0, y0=0.0)
    inv2 = SourceInverter(resid2, sigma, mask2, pix, psf, g2, supersample=2, reg="gradient")

    def lens_obj(fn):
        class _L:  # Adapter: fit_lens_by_evidence erwartet ein Objekt mit ray_shoot
            ray_shoot = staticmethod(fn)
        return _L()

    # Quellgitter auf die grob gefundene Quelle 2 zentrieren
    first = inv2.invert(ray_s2(1.4, 0.0))
    s2 = np.clip(first.source, 0, None)
    g2x, g2y = g2.axes
    G2X, G2Y = np.meshgrid(g2x, g2y)
    g2.x0, g2.y0 = float((G2X * s2).sum() / s2.sum()), float((G2Y * s2).sum() / s2.sum())
    # Wichtig: kompaktes Quellgitter. Ein großes Gitter erlaubt bei zu kleinem η eine Lösung,
    # in der die Quelle das kaum mehrfach abgebildete Bild einfach "abkopiert" (inkl. Resten des
    # Linsenlichts); die Evidenz bevorzugt dann diese entartete Lösung.
    g2.half = 0.3
    from scipy.ndimage import median_filter as _mf
    resid2_hp = resid2 - _mf(np.where(bad, 0.0, resid2), size=17, mode="reflect")

    def scan(inverter, b_s1, etas, broad=None, refine_at=None):
        """Grober Scan, dann fein um das Maximum. Rückgabe: (η-Gitter, ln E, beste Inversion, η_max).

        Mit ``broad`` (Inverter mit großem Gitter) wird das kompakte Gitter für jedes η auf den
        Schwerpunkt der dort grob rekonstruierten Quelle gesetzt ("mitlaufendes Gitter").
        """
        def one(e):
            fn = ray_s2(e, b_s1)
            if broad is not None:
                sb = np.clip(broad.invert(fn).source, 0, None)
                bx_, by_ = np.meshgrid(*broad.grid.axes)
                if sb.sum() > 0:
                    inverter.grid.x0 = float((bx_ * sb).sum() / sb.sum())
                    inverter.grid.y0 = float((by_ * sb).sum() / sb.sum())
            return inverter.invert(fn)

        def run(es):
            res = [one(e) for e in es]
            return np.array([r.log_evidence for r in res]), res
        lnE, res = run(etas)
        # fein um die beste Stelle in beiden Bereichen (die Evidenz kann zwei Maxima haben)
        fines, lnFs, resFs = [], [], []
        centers = []
        for lo, hi in ((1.28, 1.37), (1.39, 1.53)):
            sel = (etas >= lo) & (etas <= hi)
            centers.append(float(etas[int(np.argmax(np.where(sel, lnE, -np.inf)))]))
        if refine_at is not None:
            centers = [refine_at(etas, lnE)]
        for c in centers:
            fine = np.round(np.arange(c - 0.016, c + 0.0161, 0.004), 4)
            lnF, resF = run(fine)
            fines.append(fine)
            lnFs.append(lnF)
            resFs.extend(resF)
        all_e = np.concatenate([etas, *fines])
        all_l = np.concatenate([lnE, *lnFs])
        order = np.argsort(all_e)
        all_res = list(res) + resFs
        kk = int(np.argmax(all_l))
        return all_e[order], all_l[order], all_res[kk], float(all_e[kk])

    etas = np.round(np.arange(1.30, 1.6001, 0.02), 4)

    # Basismodell + Varianten der Analyse (Systematik)
    # Basis: Hochpass (Median 17 px ≈ 0,85″) entfernt großräumige Reste des Linsenlichts, die das
    # einfache Sérsic-Lichtmodell hinterlässt; die schmalen Bögen bleiben erhalten.
    variants = {
        "Basis (Hochpass, Gitter ±0,30″)": dict(r_in=1.85, half=0.30, reg="gradient", hp=True),
        "Gitter ±0,25″": dict(r_in=1.85, half=0.25, reg="gradient", hp=True),
        "Gitter ±0,35″": dict(r_in=1.85, half=0.35, reg="gradient", hp=True),
        "Maske ab 1,95″": dict(r_in=1.95, half=0.30, reg="gradient", hp=True),
        "Krümmungs-Regularisierung": dict(r_in=1.85, half=0.30, reg="curvature", hp=True),
        "ohne Hochpass (nur zur Kontrolle)": dict(r_in=1.85, half=0.30, reg="gradient", hp=False),
    }
    results = {}
    for vname, v in variants.items():
        m2 = (rr > v["r_in"]) & (rr < 2.9) & ~bad & (np.abs(xx) < HALF - 0.1) & (np.abs(yy) < HALF - 0.1)
        dat = resid2_hp if v["hp"] else resid2
        g = SourceGrid(n=30, half=v["half"], x0=g2.x0, y0=g2.y0)
        inv = SourceInverter(dat, sigma, m2, pix, psf, g, supersample=2, reg=v["reg"])
        broad = SourceInverter(dat, sigma, m2, pix, psf, SourceGrid(30, 0.9, 0.0, 0.0), supersample=2)
        e_grid, lnE, best, e_best = scan(inv, 0.0, etas, broad)
        results[vname] = dict(eta=e_best, lnE=lnE, etas=e_grid, res=best)
        print(f"   {vname:40s}: η_max = {e_best:.3f}, χ²_red = {best.reduced_chi2:.2f}")
    base = results["Basis (Hochpass, Gitter ±0,30″)"]

    def mode_lnE(r, lo, hi):
        sel = (r["etas"] >= lo) & (r["etas"] <= hi)
        k = int(np.argmax(np.where(sel, r["lnE"], -np.inf)))
        return float(r["etas"][k]), float(r["lnE"][k])

    for r in results.values():
        r["hi_mode"] = mode_lnE(r, 1.40, 1.52)
        r["lo_mode"] = mode_lnE(r, 1.28, 1.36)
        r["dlnE"] = r["hi_mode"][1] - r["lo_mode"][1]
    good = [r for k, r in results.items() if "Kontrolle" not in k]
    hi_vals = np.array([r["hi_mode"][0] for r in good])
    eta_iso = float(np.median(hi_vals))
    eta_sys = float(np.max(np.abs(hi_vals - eta_iso)))

    # Masse von Quelle 1 (SIS mit Einsteinradius b_s1 für Quelle 2): Entartung mit η
    m2 = (rr > 1.85) & (rr < 2.9) & ~bad & (np.abs(xx) < HALF - 0.1) & (np.abs(yy) < HALF - 0.1)
    inv_b = SourceInverter(resid2_hp, sigma, m2, pix, psf, SourceGrid(30, 0.30, g2.x0, g2.y0), supersample=2)
    broad_b = SourceInverter(resid2_hp, sigma, m2, pix, psf, SourceGrid(30, 0.9, 0.0, 0.0), supersample=2)
    def nearest_local_max(es, ls, target):
        """Lokales Maximum der Evidenz, das ``target`` am nächsten liegt (Lösung A stetig verfolgen)."""
        idx = [i for i in range(len(ls)) if (i == 0 or ls[i] >= ls[i - 1]) and (i == len(ls) - 1 or ls[i] >= ls[i + 1])]
        return float(es[min(idx, key=lambda i: abs(es[i] - target))])

    bvals = [0.0, 0.1, 0.2]
    eta_of_b = []
    track = 1.46
    for bs in bvals:
        e_grid_b, lnE_b, _, _ = scan(inv_b, bs, etas, broad_b,
                                     refine_at=lambda es, ls, t=track: nearest_local_max(es, ls, t))
        track = nearest_local_max(e_grid_b, lnE_b, track)
        eta_of_b.append(track)
        print(f"   b_s1 = {bs:.1f}″: Lösung A bei η = {track:.3f}")
    write_table(OUT / "eta_scan.dat", ["eta", "dlnE"], base["etas"], base["lnE"] - base["lnE"].max())

    # ------------------------------------------------ 4. Kosmologie
    # η hängt an der Steigung γ' des Dichteprofils: η = (θ_E2/θ_E1)^(γ'-1); SIE: γ' = 2.
    ratio = eta_iso
    gammas = np.linspace(1.8, 2.2, 41)
    write_table(OUT / "eta_vs_gamma.dat", ["gamma", "eta", "lo", "hi"], gammas, ratio ** (gammas - 1),
                (ratio - eta_sys) ** (gammas - 1), (ratio + eta_sys) ** (gammas - 1))
    ws = np.linspace(-2.5, 0.0, 101)
    eta_w = np.array([eta_model(0.315, w) for w in ws])
    write_table(OUT / "eta_vs_w.dat", ["w", "eta"], ws, eta_w)
    eta_lcdm = float(eta_model(0.315, -1.0))
    gamma_lcdm = 1 + np.log(eta_lcdm) / np.log(ratio)

    def w_from_eta(e):
        order = np.argsort(eta_w)
        return float(np.interp(e, eta_w[order], ws[order]))

    macros = {"etalcdm": eta_lcdm, "zd": Z_D, "zsa": Z_S1, "zsb": Z_S2,
              "thetaEpix": lens_p["thetaE"], "qpix": q1, "chipix": res1.reduced_chi2, "lampix": res1.lam,
              "thetaEpar": p["thetaE"], "chipar": par.reduced_chi2,
              "etaiso": eta_iso, "etasys": eta_sys, "wiso": w_from_eta(eta_iso), "gammalcdm": gamma_lcdm,
              "chiouter": base["res"].reduced_chi2, "etabone": eta_of_b[1], "etabtwo": eta_of_b[2],
              "etalow": float(np.median([r["lo_mode"][0] for r in good])),
              "etaca": 1.404, "etacaerr": 0.016}
    write_macros(OUT / "macros.tex", macros)
    rmain = base

    # ------------------------------------------------ Bilder für die Abbildung
    lo, hi = np.percentile(img[~bad], [5, 99.7])
    model_full = par.lens_light + par.amplitudes["background"] + res1.model + rmain["res"].model
    resid = np.where(bad, np.nan, (img - model_full) / sigma)
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#9CA3AF")
    plt.imsave(OUT / "data.png", stretch(img, lo, hi)[::-1], cmap="magma", vmin=0, vmax=1)
    rings = np.where(bad, np.nan, resid_light)
    lo2, hi2 = np.nanpercentile(rings, [5, 99.5])
    cm2 = plt.get_cmap("magma").copy()
    cm2.set_bad("#000000")
    plt.imsave(OUT / "rings.png", stretch(np.nan_to_num(rings), lo2, hi2)[::-1], cmap=cm2, vmin=0, vmax=1)
    plt.imsave(OUT / "model.png", stretch(res1.model + rmain["res"].model, lo2, hi2)[::-1], cmap="magma", vmin=0, vmax=1)
    plt.imsave(OUT / "resid.png", np.clip(resid, -5, 5)[::-1], cmap=cm, vmin=-5, vmax=5)
    plt.imsave(OUT / "source1.png", (s1 / s1.max())[::-1], cmap="magma", vmin=0, vmax=1)
    s2img = np.clip(rmain["res"].source, 0, None)
    plt.imsave(OUT / "source2.png", (s2img / max(s2img.max(), 1e-12))[::-1], cmap="magma", vmin=0, vmax=1)
    write_macros(OUT / "extent.tex", {
        "half": HALF, "xl": xl, "yl": yl,
        "sax": g1.x0 - g1.half, "sbx": g1.x0 + g1.half, "say": g1.y0 - g1.half, "sby": g1.y0 + g1.half,
        "tax": g2.x0 - g2.half, "tbx": g2.x0 + g2.half, "tay": g2.y0 - g2.half, "tby": g2.y0 + g2.half,
    })

    # ------------------------------------------------ Zusammenfassung
    lines = [
        "# SDSS J0946+1006: pixelierte Quellen und zweite Quellebene", "",
        "Automatisch erzeugt von `analysis/jackpot.py`.", "",
        f"Rotverschiebungen: Linse {Z_D}, Quelle 1 {Z_S1} (Bolton et al. 2008), Quelle 2 {Z_S2} (Smith & Collett 2021).", "",
        "## Innerer Ring mit pixelierter Quelle", "",
        f"- parametrische Sérsic-Quelle: θ_E = {p['thetaE']:.3f}″, χ²_red = {par.reduced_chi2:.2f}",
        f"- pixelierte Quelle (36×36, Gradient-Regularisierung, λ = {res1.lam:.3g} aus der Evidenz): "
        f"θ_E = {lens_p['thetaE']:.3f}″, q = {q1:.3f}, γ_ext = {np.hypot(lens_p['g1'], lens_p['g2']):.3f}, "
        f"χ²_red im Ringbereich = {res1.reduced_chi2:.2f}", "",
        "## Zweite Quellebene", "",
        "Gescannt wurde η von 1,30 bis 1,60 (grob, dann fein um das Maximum); Ergebnis ist das Maximum der "
        "Evidenz. Das Quellgitter für Quelle 2 muss kompakt sein: Mit ±0,5″ gewinnt eine entartete Lösung bei "
        "kleinem η, in der die Quelle das kaum mehrfach abgebildete Bild abkopiert. Für jedes η wird das kompakte "
        "Gitter auf die grob rekonstruierte Quelle zentriert (mitlaufendes Gitter). Ohne Hochpass nutzt die "
        "Inversion großräumige Reste des Linsenlichts aus (schlechteres χ²); diese Variante ist nur zur Kontrolle "
        "aufgeführt und geht nicht in die Systematik ein.", "",
        "| Analyse | globales Maximum η | Lösung A (η ≈ 1,46) | Lösung B (η ≈ 1,31) | ln E(A) − ln E(B) |",
        "|---|---|---|---|---|",
    ]
    for vname, r in results.items():
        lines.append(f"| {vname} | {r['eta']:.3f} | {r['hi_mode'][0]:.3f} | {r['lo_mode'][0]:.3f} | {r['dlnE']:+.0f} |")
    lines += [
        "",
        "- Die Evidenz ist **bimodal**: Lösung A (η ≈ 1,46) und Lösung B (η ≈ 1,31). Welche gewinnt, hängt "
        "von Analysedetails ab; die Daten (eine einzelne F814W-Aufnahme, 1048 s) trennen sie nicht sicher. "
        "Lösung B liegt am Rand des plausiblen Bereichs und entspricht einem kaum mehrfach abgebildeten äußeren Ring.",
        f"- Lösung A: θ_E2/θ_E1 = {eta_iso:.3f} ± {eta_sys:.3f} (Spannweite der Varianten) – passt zu Gavazzi et al. "
        "(2008), die aus tieferen Daten 2,07″/1,43″ = 1,45 fanden.",
        f"- Masse von Quelle 1 als SIS (Lösung A stetig verfolgt): b_s1 = 0,1″ → η = {eta_of_b[1]:.3f}, "
        f"b_s1 = 0,2″ → η = {eta_of_b[2]:.3f}. Mehr Masse in Quelle 1 lenkt Quelle 2 zusätzlich ab; "
        "die Hauptlinse muss dann weniger leisten.",
        f"- Flaches ΛCDM (Ωₘ = 0,315) sagt η = {eta_lcdm:.3f} voraus. Isotherm und ohne Masse von Quelle 1 "
        f"entspräche das gemessene η einem w ≈ {w_from_eta(eta_iso):.1f}; mit γ′ = {gamma_lcdm:.2f} statt 2 wäre es "
        f"genau ΛCDM.",
        "- Collett & Auger (2014) marginalisieren über die Steigung γ′ und die Masse von Quelle 1 und finden "
        "η = 1,404 ± 0,016; mit der spektroskopischen Rotverschiebung z = 2,035 (Smith & Collett 2021) und einem "
        "CMB-Prior ergibt sich w = −1,04 ± 0,20.", "",
        "Fazit: Der innere Ring ist mit pixelierter Quelle sehr gut bestimmt. Für den schwachen äußeren Ring reicht "
        "diese eine Aufnahme nicht für eine eindeutige Messung: Die physikalisch plausible Lösung A reproduziert "
        "den Literaturwert, eine zweite Lösung lässt sich aber nicht sicher ausschließen. Selbst mit eindeutiger "
        "Geometrie hinge die Kosmologie an der Profilsteigung γ′ und der Masse von Quelle 1 – genau den Größen, "
        "die die Forschung mit tieferen Mehrband-Daten und Sternkinematik einschränkt.", "",
        "![Jackpot](../figures/jackpot/jackpot_fit.svg)", "",
        "![Kosmologie](../figures/jackpot/jackpot_eta.svg)", "",
    ]
    (ROOT / "analysis" / "jackpot_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
