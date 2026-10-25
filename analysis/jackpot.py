"""SDSS J0946+1006 ("Jackpot"): two source planes -> cosmology.

    python analysis/jackpot.py

1. Parametric fit (Sérsic light, SIE + shear) -> lens light and starting model
2. Inner ring (source 1, z = 0.609): pixelated source, lens parameters via the
   Bayesian evidence (Suyu et al. 2006)
3. Outer ring (source 2, z = 2.035; Smith & Collett 2021): the same lens, but for
   the more distant source deflecting more strongly by the factor

       η = (D_ds2 / D_s2) / (D_ds1 / D_s1)

   optionally, source 1 (as an SIS) additionally deflects the light of source 2
   (multi-plane lens equation). η depends only on Ωₘ and w, not on H₀.

Comparison: Collett & Auger (2014) measure 1/β = η = 1.404 ± 0.016 (including the mass of source 1).
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
Z_D, Z_S1, Z_S2 = 0.222, 0.609, 2.035   # lens/source 1: Bolton+08; source 2: Smith & Collett (2021)
HALF = 3.4                               # cutout ±3.4″
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

    # mask outliers (cosmic rays, defects)
    hp = cut - median_filter(cut, size=5, mode="reflect")
    bad = binary_dilation((hp > 5 * sigma) | (cut == 0), iterations=1)
    sig = np.where(bad, np.inf, sigma)

    cat = load_slacs()
    row = cat[cat["name"] == NAME][0]
    phi0 = pixel_angle_of_pa(wcs, xc, yc, float(row["pa"]))

    # ------------------------------------------------ 1. parametric (light + starting model)
    print("1) parametric fit …")
    par = fit_lens_image(img, sig, pix, theta_E0=float(row["theta_E"]), q0=float(row["q"]), phi0=phi0,
                         psf_fwhm=PIX_PSF_FWHM, supersample=2, n_starts=6)
    p = par.params
    xl, yl = p["xl"], p["yl"]
    print(f"   θE = {p['thetaE']:.3f}″, light centre ({xl:.2f}, {yl:.2f})″, χ²_red = {par.reduced_chi2:.2f}")
    resid_light = img - par.lens_light - par.amplitudes["background"]

    yy, xx = (np.mgrid[0:n, 0:n] - (n - 1) / 2) * pix
    rr = np.hypot(xx - xl, yy - yl)
    psf = gaussian_psf(PIX_PSF_FWHM, pix)

    def main_lens(q):
        qq, ph = q_phi_from_e(q["e1m"], q["e2m"])
        return SIE(q["thetaE"], q=qq, phi=ph, x0=xl, y0=yl) + ExternalShear(q["g1"], q["g2"])

    # ------------------------------------------------ 2. inner ring: pixelated source
    print("2) inner ring, pixelated source …")
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

    # centroid of the reconstructed source 1 (position of the "lens" for source 2)
    gx, gy = g1.axes
    s1 = np.clip(res1.source, 0, None)
    GX, GY = np.meshgrid(gx, gy)
    s1x, s1y = float((GX * s1).sum() / s1.sum()), float((GY * s1).sum() / s1.sum())

    # ------------------------------------------------ 3. outer ring: second source plane
    print("3) outer ring, second source plane …")
    resid2 = resid_light - res1.model
    mask2 = (rr > 1.85) & (rr < 2.9) & ~bad & (np.abs(xx) < HALF - 0.1) & (np.abs(yy) < HALF - 0.1)

    def ray_s2(eta, b_s1):
        def f(x, y):
            ax, ay = lens1.deflection(x, y)
            # plane 1: position where the ray passes source 1
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
        class _L:  # adapter: fit_lens_by_evidence expects an object with ray_shoot
            ray_shoot = staticmethod(fn)
        return _L()

    # centre the source grid on the roughly located source 2
    first = inv2.invert(ray_s2(1.4, 0.0))
    s2 = np.clip(first.source, 0, None)
    g2x, g2y = g2.axes
    G2X, G2Y = np.meshgrid(g2x, g2y)
    g2.x0, g2.y0 = float((G2X * s2).sum() / s2.sum()), float((G2Y * s2).sum() / s2.sum())
    # Important: compact source grid. A large grid allows, for too small η, a solution in which
    # the source simply "copies" the barely multiply imaged image (including residuals of the
    # lens light); the evidence then prefers this degenerate solution.
    g2.half = 0.3
    from scipy.ndimage import median_filter as _mf
    resid2_hp = resid2 - _mf(np.where(bad, 0.0, resid2), size=17, mode="reflect")

    def scan(inverter, b_s1, etas, broad=None, refine_at=None):
        """Coarse scan, then fine around the maximum. Returns (η grid, ln E, best inversion, η_max).

        With ``broad`` (an inverter with a large grid) the compact grid is centred, for each η, on
        the centroid of the source roughly reconstructed there ("tracking grid").
        """
        def one(e):
            fn = ray_s2(e, b_s1)
            if broad is not None:
                sb = np.clip(broad.invert(fn).source, 0, None)
                bx_, by_ = np.meshgrid(*broad.grid.axes)
                if sb.sum() > 0:
                    inverter.grid.x0 = float((bx_ * sb).sum() / sb.sum())
                    inverter.grid.y0 = float((by_ * sb).sum() / sb.sum())
            r = inverter.invert(fn)
            # the tracking grid is mutated for every η: keep a copy of the one used here
            gr = inverter.grid
            r.grid = SourceGrid(gr.n, gr.half, gr.x0, gr.y0)
            return r

        def run(es):
            res = [one(e) for e in es]
            return np.array([r.log_evidence for r in res]), res
        lnE, res = run(etas)
        # refine around the best point in both ranges (the evidence can have two maxima)
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

    # baseline model + analysis variants (systematics)
    # baseline: a high-pass (median 17 px ≈ 0.85″) removes large-scale lens-light residuals left by
    # the simple Sérsic light model; the narrow arcs are preserved.
    variants = {
        "baseline (high-pass, grid ±0.30″)": dict(r_in=1.85, half=0.30, reg="gradient", hp=True),
        "grid ±0.25″": dict(r_in=1.85, half=0.25, reg="gradient", hp=True),
        "grid ±0.35″": dict(r_in=1.85, half=0.35, reg="gradient", hp=True),
        "mask from 1.95″": dict(r_in=1.95, half=0.30, reg="gradient", hp=True),
        "curvature regularization": dict(r_in=1.85, half=0.30, reg="curvature", hp=True),
        "no high-pass (control only)": dict(r_in=1.85, half=0.30, reg="gradient", hp=False),
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
    base = results["baseline (high-pass, grid ±0.30″)"]

    def mode_lnE(r, lo, hi):
        sel = (r["etas"] >= lo) & (r["etas"] <= hi)
        k = int(np.argmax(np.where(sel, r["lnE"], -np.inf)))
        return float(r["etas"][k]), float(r["lnE"][k])

    for r in results.values():
        r["hi_mode"] = mode_lnE(r, 1.40, 1.52)
        r["lo_mode"] = mode_lnE(r, 1.28, 1.36)
        r["dlnE"] = r["hi_mode"][1] - r["lo_mode"][1]
    good = [r for k, r in results.items() if "control" not in k]
    hi_vals = np.array([r["hi_mode"][0] for r in good])
    eta_iso = float(np.median(hi_vals))
    eta_sys = float(np.max(np.abs(hi_vals - eta_iso)))

    # mass of source 1 (SIS with Einstein radius b_s1 for source 2): degenerate with η
    m2 = (rr > 1.85) & (rr < 2.9) & ~bad & (np.abs(xx) < HALF - 0.1) & (np.abs(yy) < HALF - 0.1)
    inv_b = SourceInverter(resid2_hp, sigma, m2, pix, psf, SourceGrid(30, 0.30, g2.x0, g2.y0), supersample=2)
    broad_b = SourceInverter(resid2_hp, sigma, m2, pix, psf, SourceGrid(30, 0.9, 0.0, 0.0), supersample=2)
    def nearest_local_max(es, ls, target):
        """Local maximum of the evidence closest to ``target`` (track solution A continuously)."""
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
        print(f"   b_s1 = {bs:.1f}″: solution A at η = {track:.3f}")
    write_table(OUT / "eta_scan.dat", ["eta", "dlnE"], base["etas"], base["lnE"] - base["lnE"].max())

    # ------------------------------------------------ 4. cosmology
    # η depends on the slope γ' of the density profile: η = (θ_E2/θ_E1)^(γ'-1); SIE: γ' = 2.
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
    gs2 = rmain["res"].grid  # source grid of the best inversion for source 2

    # ------------------------------------------------ images for the figure
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
        "tax": gs2.x0 - gs2.half, "tbx": gs2.x0 + gs2.half, "tay": gs2.y0 - gs2.half, "tby": gs2.y0 + gs2.half,
    })

    # ------------------------------------------------ summary
    lines = [
        "# SDSS J0946+1006: pixelated sources and a second source plane", "",
        "Generated automatically by `analysis/jackpot.py`.", "",
        f"Redshifts: lens {Z_D}, source 1 {Z_S1} (Bolton et al. 2008), source 2 {Z_S2} (Smith & Collett 2021).", "",
        "## Inner ring with a pixelated source", "",
        f"- parametric Sérsic source: θ_E = {p['thetaE']:.3f}″, χ²_red = {par.reduced_chi2:.2f}",
        f"- pixelated source (36×36, gradient regularization, λ = {res1.lam:.3g} from the evidence): "
        f"θ_E = {lens_p['thetaE']:.3f}″, q = {q1:.3f}, γ_ext = {np.hypot(lens_p['g1'], lens_p['g2']):.3f}, "
        f"χ²_red in the ring region = {res1.reduced_chi2:.2f}", "",
        "## Second source plane", "",
        "η was scanned from 1.30 to 1.60 (coarse, then fine around the maximum); the result is the maximum of "
        "the evidence. The source grid for source 2 has to be compact: with ±0.5″ a degenerate solution at small "
        "η wins, in which the source copies the barely multiply imaged image. For each η the compact grid is "
        "centred on the roughly reconstructed source (tracking grid). Without the high-pass the inversion exploits "
        "large-scale lens-light residuals (worse χ²); this variant is listed only as a control and does not enter "
        "the systematics.", "",
        "| Analysis | global maximum η | solution A (η ≈ 1.46) | solution B (η ≈ 1.31) | ln E(A) − ln E(B) |",
        "|---|---|---|---|---|",
    ]
    for vname, r in results.items():
        lines.append(f"| {vname} | {r['eta']:.3f} | {r['hi_mode'][0]:.3f} | {r['lo_mode'][0]:.3f} | {r['dlnE']:+.0f} |")
    lines += [
        "",
        "- The evidence is **bimodal**: solution A (η ≈ 1.46) and solution B (η ≈ 1.31). Which one wins depends "
        "on analysis details; the data (a single F814W exposure, 1048 s) do not separate them reliably. "
        "Solution B lies at the edge of the plausible range and corresponds to a barely multiply imaged outer ring.",
        f"- Solution A: θ_E2/θ_E1 = {eta_iso:.3f} ± {eta_sys:.3f} (spread of the variants) – consistent with Gavazzi et al. "
        "(2008), who found 2.07″/1.43″ = 1.45 from deeper data.",
        f"- Mass of source 1 as an SIS (solution A tracked continuously): b_s1 = 0.1″ → η = {eta_of_b[1]:.3f}, "
        f"b_s1 = 0.2″ → η = {eta_of_b[2]:.3f}. More mass in source 1 deflects source 2 additionally, "
        "so the main lens has to do less.",
        f"- Flat ΛCDM (Ωₘ = 0.315) predicts η = {eta_lcdm:.3f}. Isothermal and without mass in source 1, the measured "
        f"η would correspond to w ≈ {w_from_eta(eta_iso):.1f}; with γ′ = {gamma_lcdm:.2f} instead of 2 it would be "
        f"exactly ΛCDM.",
        "- Collett & Auger (2014) marginalize over the slope γ′ and the mass of source 1 and find "
        "η = 1.404 ± 0.016; with the spectroscopic redshift z = 2.035 (Smith & Collett 2021) and a "
        "CMB prior this gives w = −1.04 ± 0.20.", "",
        "Conclusion: the inner ring is very well constrained with a pixelated source. For the faint outer ring this "
        "single exposure is not enough for an unambiguous measurement: the physically plausible solution A "
        "reproduces the literature value, but a second solution cannot be excluded with confidence. Even with an "
        "unambiguous geometry the cosmology would hinge on the profile slope γ′ and the mass of source 1 – exactly "
        "the quantities that research constrains with deeper multi-band data and stellar kinematics.", "",
        "![Jackpot](../figures/jackpot/jackpot_fit.svg)", "",
        "![Cosmology](../figures/jackpot/jackpot_eta.svg)", "",
    ]
    (ROOT / "analysis" / "jackpot_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
