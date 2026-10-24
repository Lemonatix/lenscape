"""SLACS-Auswertung: Massen, Dunkle Materie und Kosmologie aus 63 echten Linsen.

    python analysis/slacs.py

Schreibt Abbildungen nach figures/slacs/ und eine Zusammenfassung nach
analysis/slacs_results.md.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lenscape.catalogs import SLACS_REF_COSMO, load_slacs
from lenscape.cosmography import fit_flat_wcdm, model_distance_ratio
from lenscape.cosmology import PLANCK18_OM0, make_cosmology
from lenscape.strong_lenses import einstein_quantities, summarize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "slacs"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#777777"


def fig_fsie(t):
    good = t["good_sigma"]
    f = np.asarray(t["f_SIE"][good])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(f, bins=np.arange(0.65, 1.3, 0.04), color=BLUE, alpha=0.85)
    ax.axvline(1, color="k", lw=1)
    ax.axvline(f.mean(), color=ORANGE, ls="--", label=f"Mittel {f.mean():.3f} ± {f.std(ddof=1)/np.sqrt(len(f)):.3f}")
    ax.set_xlabel(r"$f_{SIE} = \sigma_{SDSS} / \sigma_{SIE}$")
    ax.set_ylabel("Anzahl Linsen")
    ax.set_title("Sterne vs. Linse: Ist die Gesamtmasse isotherm?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "f_sie.png")
    plt.close(fig)


def fig_fdm(t):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.asarray(t["R_E_over_R_e"])
    for imf, col, lab in [("chab", BLUE, "Chabrier-IMF"), ("salp", ORANGE, "Salpeter-IMF")]:
        y = np.asarray(t[f"fDM_{imf}"])
        e = np.asarray(t[f"e_fstar_{imf}"])
        m = np.isfinite(y)
        ax.errorbar(x[m], y[m], yerr=e[m], fmt="o", ms=3.5, color=col, alpha=0.8, elinewidth=0.7, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$R_E / R_e$ (Einsteinradius / Effektivradius)")
    ax.set_ylabel(r"$f_{DM}(<R_E)$")
    ax.set_title("Anteil Dunkler Materie innerhalb des Einsteinradius")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "f_dm.png")
    plt.close(fig)


def fig_mass_vs_cosmology(base):
    tab = load_slacs()
    ws = np.linspace(-2.0, -0.4, 17)
    oms = np.linspace(0.1, 0.6, 11)
    ref = np.asarray(base["M_E"])
    mw = [np.median(einstein_quantities(tab, make_cosmology(70, 0.3, None, w))["M_E"] / ref) for w in ws]
    mo = [np.median(einstein_quantities(tab, make_cosmology(70, om, None, -1.0))["M_E"] / ref) for om in oms]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    a1.plot(ws, mw, "o-", color=BLUE, ms=3)
    a1.set_xlabel("w (Dunkle Energie)")
    a1.set_ylabel(r"Median $M_E / M_E^{\rm ref}$")
    a2.plot(oms, mo, "o-", color=ORANGE, ms=3)
    a2.set_xlabel(r"$\Omega_m$")
    for a in (a1, a2):
        a.axhline(1, color="k", lw=0.8)
    fig.suptitle("Wie stark hängen die gemessenen Galaxienmassen von der Kosmologie ab? (H₀ = 70)")
    fig.tight_layout()
    fig.savefig(FIG / "mass_vs_cosmology.png")
    plt.close(fig)
    return float(np.ptp(mw)), float(np.ptp(mo))


def contours(ax, res, color, label):
    d = res.delta_chi2()
    ax.contourf(res.Om, res.w, d, levels=[0, 2.30, 6.17], colors=[color, color], alpha=[0.55, 0.25])
    ax.contour(res.Om, res.w, d, levels=[2.30, 6.17], colors=[color], linewidths=0.8)
    ax.plot([], [], color=color, lw=6, alpha=0.5, label=label)


def forecast(t, n_lenses, seed=1):
    """Synthetische Stichprobe wie SLACS (gleiche z- und Streuungsverteilung), Wahrheit = Planck-ΛCDM."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), n_lenses)
    z_d = np.asarray(t["z_d"])[idx]
    z_s = np.asarray(t["z_s"])[idx]
    sigma = np.asarray(t["sigma"])[idx]
    e_sigma = np.asarray(t["e_sigma"])[idx]
    ratio = model_distance_ratio(z_d, z_s, PLANCK18_OM0, -1.0)
    theta = 4 * np.pi * (sigma / 299792.458) ** 2 * ratio / (np.pi / 180 / 3600)
    theta *= np.exp(rng.normal(0, 0.15, n_lenses))  # intrinsische Streuung wie bei SLACS gemessen
    return fit_flat_wcdm(theta, sigma, e_sigma, z_d, z_s, e_theta_frac=0.02,
                         sigma_int=np.linspace(0.1, 0.2, 11))


def fig_cosmography(t):
    g = t[t["good_sigma"]]
    res = fit_flat_wcdm(g["theta_E"], g["sigma"], g["e_sigma"], g["z_d"], g["z_s"], e_theta_frac=0.02)
    fc = forecast(g, 2000)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    contours(a1, res, BLUE, f"SLACS ({len(g)} Linsen)")
    contours(a2, fc, ORANGE, "Prognose: 2000 Linsen")
    for a in (a1, a2):
        a.plot(PLANCK18_OM0, -1, "*", color="k", ms=10, label="Planck ΛCDM")
        a.set_xlabel(r"$\Omega_m$")
        a.legend(loc="lower right", fontsize=8)
    a1.set_ylabel("w")
    a1.set_title("Echte Daten")
    a2.set_title("Gleiches Verfahren, Euclid/LSST-Größenordnung")
    fig.suptitle(r"Kosmographie aus $\theta_E$ und $\sigma$: 68 %/95 %-Bereiche (flaches wCDM)")
    fig.tight_layout()
    fig.savefig(FIG / "cosmography.png")
    plt.close(fig)

    d = res.delta_chi2()
    j = np.argmin(np.abs(res.w + 1))
    i = np.argmin(np.abs(res.Om - PLANCK18_OM0))
    return res, float(d[j, i])


def main():
    tab = load_slacs()
    t = einstein_quantities(tab, make_cosmology(**SLACS_REF_COSMO))
    s = summarize(t)
    fig_fsie(t)
    fig_fdm(t)
    span_w, span_om = fig_mass_vs_cosmology(t)
    res, dchi2_planck = fig_cosmography(t)
    best = res.best

    m = np.isfinite(t["logM_E_auger"])
    dlog = np.log10(t["M_E"][m]) - t["logM_E_auger"][m]

    lines = [
        "# SLACS-Auswertung",
        "",
        "Automatisch erzeugt von `analysis/slacs.py`. Referenzkosmologie: H₀ = 70, Ωₘ = 0,3, Ω_Λ = 0,7.",
        "",
        "## Kennzahlen",
        "",
        "| Größe | Wert |",
        "|---|---|",
    ]
    for k, v in s.items():
        lines.append(f"| {k} | {v:.4g} |")
    lines += [
        "",
        "## Validierung",
        "",
        f"Unsere Einsteinmassen weichen von den publizierten Werten (Auger et al. 2009) im Mittel um "
        f"{dlog.mean():+.4f} dex ab (Streuung {dlog.std():.4f} dex, {m.sum()} Linsen).",
        "",
        "## Kosmologie",
        "",
        f"- Bestfit (flaches wCDM): Ωₘ = {best['Om']:.2f}, w = {best['w']:.2f}, "
        f"f_E = {best['f_E']:.3f}, intrinsische Streuung {best['sigma_int']:.2f}",
        f"- Planck-ΛCDM (Ωₘ = {PLANCK18_OM0}, w = −1) liegt bei Δχ² = {dchi2_planck:.2f}, also verträglich.",
        f"- Variiert man w von −2 bis −0,4, ändern sich die Mediane der Massen um {100*span_w:.1f} %; "
        f"für Ωₘ von 0,1 bis 0,6 um {100*span_om:.1f} %.",
        "",
        "![f_SIE](../figures/slacs/f_sie.png)",
        "![f_DM](../figures/slacs/f_dm.png)",
        "![Masse vs. Kosmologie](../figures/slacs/mass_vs_cosmology.png)",
        "![Kosmographie](../figures/slacs/cosmography.png)",
        "",
    ]
    (ROOT / "analysis" / "slacs_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
