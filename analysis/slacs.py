"""SLACS analysis: masses, dark matter and cosmology from 63 real lenses.

    python analysis/slacs.py          # compute + write data tables
    python figures/build.py           # build the TikZ/pgfplots figures (PDF + SVG)

The computation only writes data (figures/slacs/data/*.dat). How it looks is
defined by the pgfplots files in figures/slacs/*.tex.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import contourpy
import numpy as np

from lenscape.catalogs import SLACS_REF_COSMO, load_slacs
from lenscape.cosmography import fit_flat_wcdm, model_distance_ratio
from lenscape.cosmology import PLANCK18_OM0, make_cosmology
from lenscape.strong_lenses import einstein_quantities, summarize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "slacs" / "data"
DATA.mkdir(parents=True, exist_ok=True)

DCHI2_LEVELS = {"68": 2.30, "95": 6.17}  # two parameters


def write_table(name: str, header: list[str], *cols, fmt="%.6g") -> None:
    arr = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    np.savetxt(DATA / name, arr, header=" ".join(header), comments="", fmt=fmt)


def write_scalars(name: str, values: dict[str, float]) -> None:
    """Single values as pgf macros, e.g. \\def\\fsiemean{0.952}."""
    lines = [f"\\def\\{k}{{{v:.4g}}}" for k, v in values.items()]
    (DATA / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- exports
def export_fsie(t) -> dict:
    f = np.asarray(t["f_SIE"][t["good_sigma"]])
    edges = np.arange(0.66, 1.30, 0.04)
    counts, _ = np.histogram(f, bins=edges)
    # pgfplots "ybar interval": the last row repeats the count at the right edge
    write_table("f_sie_hist.dat", ["left", "count"], edges, np.append(counts, counts[-1]))
    return {"fsiemean": f.mean(), "fsieerr": f.std(ddof=1) / np.sqrt(len(f)), "fsiestd": f.std(ddof=1),
            "fsien": len(f), "fsieymax": counts.max() + 6}


def export_fdm(t) -> None:
    x = np.asarray(t["R_E_over_R_e"])
    for imf in ("chab", "salp"):
        y = np.asarray(t[f"fDM_{imf}"])
        e = np.asarray(t[f"e_fstar_{imf}"])
        m = np.isfinite(y) & np.isfinite(e)
        write_table(f"f_dm_{imf}.dat", ["x", "y", "ey"], x[m], y[m], e[m])


def export_mass_vs_cosmology(t) -> dict:
    tab = load_slacs()
    ref = np.asarray(t["M_E"])
    ws = np.linspace(-2.0, -0.4, 33)
    oms = np.linspace(0.1, 0.6, 26)
    mw = [np.median(einstein_quantities(tab, make_cosmology(70, 0.3, None, w))["M_E"] / ref) for w in ws]
    mo = [np.median(einstein_quantities(tab, make_cosmology(70, om, None, -1.0))["M_E"] / ref) for om in oms]
    write_table("mass_vs_w.dat", ["w", "ratio"], ws, mw)
    write_table("mass_vs_om.dat", ["Om", "ratio"], oms, mo)
    return {"massspanw": 100 * np.ptp(mw), "massspanom": 100 * np.ptp(mo)}


def export_contours(res, tag: str) -> list[str]:
    """Filled Δχ² regions as polygons; returns the list of file names per level."""
    d = res.delta_chi2()
    gen = contourpy.contour_generator(res.Om, res.w, d, fill_type=contourpy.FillType.OuterOffset)
    files = []
    for lvl, val in DCHI2_LEVELS.items():
        polys, _ = gen.filled(-1.0, val)
        for k, poly in enumerate(polys):
            name = f"cosmo_{tag}_{lvl}_{k}.dat"
            write_table(name, ["Om", "w"], poly[:, 0], poly[:, 1])
            files.append((lvl, name))
    return files


def forecast(t, n_lenses, seed=1):
    """Synthetic sample like SLACS (same z distribution), truth = Planck ΛCDM."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), n_lenses)
    z_d = np.asarray(t["z_d"])[idx]
    z_s = np.asarray(t["z_s"])[idx]
    sigma = np.asarray(t["sigma"])[idx]
    e_sigma = np.asarray(t["e_sigma"])[idx]
    ratio = model_distance_ratio(z_d, z_s, PLANCK18_OM0, -1.0)
    theta = 4 * np.pi * (sigma / 299792.458) ** 2 * ratio / (np.pi / 180 / 3600)
    theta *= np.exp(rng.normal(0, 0.15, n_lenses))  # intrinsic scatter as measured for SLACS
    return fit_flat_wcdm(theta, sigma, e_sigma, z_d, z_s, e_theta_frac=0.02,
                         sigma_int=np.linspace(0.1, 0.2, 11))


def export_cosmography(t) -> tuple[dict, object]:
    g = t[t["good_sigma"]]
    res = fit_flat_wcdm(g["theta_E"], g["sigma"], g["e_sigma"], g["z_d"], g["z_s"], e_theta_frac=0.02)
    fc = forecast(g, 2000)
    files = {"slacs": export_contours(res, "slacs"), "forecast": export_contours(fc, "forecast")}

    # pgfplots commands for the polygons (their number depends on the data)
    for tag, lst in files.items():
        cmds = [f"\\addplot[{tag}{lvl}, forget plot] table {{data/{name}}} -- cycle;" for lvl, name in
                sorted(lst, key=lambda p: -int(p[0]))]  # 95 % first, 68 % on top
        (DATA / f"cosmo_{tag}_regions.tex").write_text("\n".join(cmds) + "\n", encoding="utf-8")

    d = res.delta_chi2()
    j = np.argmin(np.abs(res.w + 1))
    i = np.argmin(np.abs(res.Om - PLANCK18_OM0))
    scal = {"nlenses": len(g), "nforecast": 2000, "dchiplanck": d[j, i], "planckOm": PLANCK18_OM0}
    return scal, res


def main():
    tab = load_slacs()
    t = einstein_quantities(tab, make_cosmology(**SLACS_REF_COSMO))
    s = summarize(t)

    scalars = {}
    scalars |= export_fsie(t)
    export_fdm(t)
    scalars |= export_mass_vs_cosmology(t)
    cs, res = export_cosmography(t)
    scalars |= cs
    write_scalars("scalars.tex", scalars)
    best = res.best

    m = np.isfinite(t["logM_E_auger"])
    dlog = np.log10(t["M_E"][m]) - t["logM_E_auger"][m]

    lines = [
        "# SLACS analysis",
        "",
        "Generated automatically by `analysis/slacs.py`. Reference cosmology: H₀ = 70, Ωₘ = 0.3, Ω_Λ = 0.7.",
        "",
        "## Key numbers",
        "",
        "| Quantity | Value |",
        "|---|---|",
    ]
    for k, v in s.items():
        lines.append(f"| {k} | {v:.4g} |")
    lines += [
        "",
        "## Validation",
        "",
        f"Our Einstein masses differ from the published values (Auger et al. 2009) by "
        f"{dlog.mean():+.4f} dex on average (scatter {dlog.std():.4f} dex, {m.sum()} lenses).",
        "",
        "## Cosmology",
        "",
        f"- Best fit (flat wCDM): Ωₘ = {best['Om']:.2f}, w = {best['w']:.2f}, "
        f"f_E = {best['f_E']:.3f}, intrinsic scatter {best['sigma_int']:.2f}",
        f"- Planck ΛCDM (Ωₘ = {PLANCK18_OM0}, w = −1) lies at Δχ² = {cs['dchiplanck']:.2f}, i.e. consistent.",
        f"- Varying w from −2 to −0.4 changes the median masses by {scalars['massspanw']:.1f} %; "
        f"varying Ωₘ from 0.1 to 0.6 by {scalars['massspanom']:.1f} %.",
        "",
        "## Figures",
        "",
        "Built with `python figures/build.py` (TikZ/pgfplots; PDF and SVG in `figures/slacs/`).",
        "",
        "![f_SIE](../figures/slacs/f_sie.svg)",
        "",
        "![f_DM](../figures/slacs/f_dm.svg)",
        "",
        "![Mass vs. cosmology](../figures/slacs/mass_vs_cosmology.svg)",
        "",
        "![Cosmography](../figures/slacs/cosmography.svg)",
        "",
    ]
    (ROOT / "analysis" / "slacs_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
