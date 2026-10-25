"""H0 from the six H0LiCOW lenses -- and the Hubble tension.

    python scripts/fetch_h0licow.py   # once, ~120 MB
    python analysis/h0licow.py        # compute, write data for the figures
    python figures/build.py h0/...    # build the figures

Reproduces Wong et al. (2020, MNRAS 498, 1420): H0 = 73.3 +1.7 -1.8 km/s/Mpc
in flat ΛCDM, and additionally shows flat wCDM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lenscape.h0_inference import infer_h0
from lenscape.h0licow import load_h0licow
from lenscape.tikzexport import export_regions, write_macros, write_table

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "h0" / "data"
DATA.mkdir(parents=True, exist_ok=True)

# reference values (median, -error, +error) in km/s/Mpc
REFERENCES = {
    "Planck 2018 (CMB, ΛCDM)": (67.4, 0.5, 0.5),             # Planck Collaboration 2020, A&A 641, A6
    "SH0ES (Cepheids + SN Ia)": (73.04, 1.04, 1.04),        # Riess et al. 2022, ApJL 934, L7
    "TDCOSMO 2025": (71.6, 3.3, 3.9),                        # TDCOSMO Collab. / Birrer et al. 2025
}
KEYS = ["B", "R", "H", "J", "W", "P"]  # short keys for TeX macros (letters only)


def main():
    lenses = load_h0licow()
    H0 = np.arange(50.0, 110.0001, 0.1)

    single = [infer_h0([lens], H0=H0) for lens in lenses]
    joint = infer_h0(lenses, H0=H0)

    # --------------------------------------------------- marginals p(H0)
    cols = [p.marginal_H0() for p in single] + [joint.marginal_H0()]
    write_table(DATA / "h0_posteriors.dat", ["H0"] + KEYS + ["all"], H0, *cols)

    # --------------------------------------------------- measurements for the whisker plot
    rows = []
    macros = {}
    for key, lens, post in zip(KEYS, lenses, single):
        med, lo, hi = post.summary()
        rows.append((lens.name, med, lo, hi))
        macros[f"hzero{key}"] = med
    med, lo, hi = joint.summary()
    macros |= {"hzeromed": med, "hzerolo": lo, "hzerohi": hi}

    planck = REFERENCES["Planck 2018 (CMB, ΛCDM)"]
    shoes = REFERENCES["SH0ES (Cepheids + SN Ia)"]
    sig = lambda a, b: abs(a[0] - b[0]) / np.hypot(0.5 * (a[1] + a[2]), 0.5 * (b[1] + b[2]))
    ours = (med, lo, hi)
    macros |= {"tensionplanck": sig(ours, planck), "tensionshoes": sig(ours, shoes)}

    # --------------------------------------------------- flat wCDM: H0-w plane
    Hw = np.arange(50.0, 120.0001, 0.25)
    w = np.linspace(-2.5, 0.5, 61)
    Om = np.linspace(0.05, 0.5, 19)
    jw = infer_h0(lenses, H0=Hw, Om=Om, w=w)
    p = np.exp(jw.logpost - jw.logpost.max())
    p_hw = p.sum(axis=1)                       # marginalize over Omega_m -> (w, H0)
    # Δχ² from the marginal density via contours enclosing equal probability mass
    dchi2 = _mass_levels_as_dchi2(p_hw)
    export_regions(DATA, "hw", Hw, w, dchi2, style_prefix="lens")
    mw = jw.summary()
    macros |= {"hzerowmed": mw[0], "hzerowlo": mw[1], "hzerowhi": mw[2]}

    macros |= {"planckmed": planck[0], "planckerr": planck[1], "shoesmed": shoes[0], "shoeserr": shoes[1]}
    write_macros(DATA / "macros.tex", macros)

    # whisker plot: top to bottom individual lenses, combination, references
    tex_names = {
        "B1608+656": "B1608$+$656", "RXJ1131-1231": "RXJ1131$-$1231", "HE0435-1223": "HE0435$-$1223",
        "J1206+4332": "J1206$+$4332", "WFI2033-4723": "WFI2033$-$4723", "PG1115+080": "PG1115$+$080",
    }
    entries = [(tex_names[n], m, l, h, "lens") for n, m, l, h in rows]
    entries.append((r"\textbf{H0LiCOW, 6 lenses}", med, lo, hi, "joint"))
    entries.append((r"TDCOSMO 2025", *REFERENCES["TDCOSMO 2025"], "ref"))
    entries.append((r"SH0ES 2022", *shoes, "late"))
    entries.append((r"Planck 2018", *planck, "early"))
    n = len(entries)
    ypos = {i: n - i for i in range(n)}  # top entry gets the largest y value
    for group in ("lens", "joint", "ref", "late", "early"):
        sel = [(ypos[i], e) for i, e in enumerate(entries) if e[4] == group]
        write_table(DATA / f"whisker_{group}.dat", ["y", "x", "em", "ep"],
                    [y for y, _ in sel], [e[1] for _, e in sel], [e[2] for _, e in sel], [e[3] for _, e in sel])
    ticks = ",".join(str(ypos[i]) for i in range(n))
    labels = ",".join("{" + e[0] + "}" for e in entries)
    (DATA / "whisker_axis.tex").write_text(
        "\\pgfplotsset{whisker axis/.style={ytick={" + ticks + "}, yticklabels={" + labels + "},"
        f" ymin=0.4, ymax={n + 0.6}}}}}\n", encoding="utf-8")

    # table of individual values + references (for the figure and the README)
    lines = ["| Measurement | H₀ [km/s/Mpc] |", "|---|---|"]
    for name, m, l, h in rows:
        lines.append(f"| {name} | {m:.1f} −{l:.1f} +{h:.1f} |")
    lines.append(f"| **H0LiCOW combined (flat ΛCDM, this work)** | **{med:.1f} −{lo:.1f} +{hi:.1f}** |")
    lines.append(f"| H0LiCOW combined (flat wCDM, this work) | {mw[0]:.1f} −{mw[1]:.1f} +{mw[2]:.1f} |")
    for name, (m, l, h) in REFERENCES.items():
        lines.append(f"| {name} | {m} −{l} +{h} |")
    table = "\n".join(lines)

    text = f"""# H0 from time delays (H0LiCOW)

Generated automatically by `analysis/h0licow.py`.

{table}

- Published (Wong et al. 2020): 73.3 +1.7 −1.8 km/s/Mpc. This work: {med:.1f} +{hi:.1f} −{lo:.1f}.
- Tension with Planck: {macros['tensionplanck']:.1f}σ, with SH0ES: {macros['tensionshoes']:.1f}σ (Gaussian approximation).
- TDCOSMO 2025 relaxes the assumptions about the mass profile (mass-sheet degeneracy, constrained by
  stellar kinematics). The uncertainty grows; the mean stays between Planck and SH0ES.

![H0 tension](../figures/h0/h0_tension.svg)

![Posteriors](../figures/h0/h0_posteriors.svg)

![H0 and w](../figures/h0/h0_w.svg)
"""
    (ROOT / "analysis" / "h0licow_results.md").write_text(text, encoding="utf-8")
    print(text)


def _mass_levels_as_dchi2(p):
    """Convert a 2D density into a 'Δχ²' such that the levels 2.30/6.17 enclose
    exactly 68 %/95 % of the probability mass (HPD regions)."""
    flat = np.sort(p.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    # mass above each density value
    mass_above = np.interp(p.ravel(), flat[::-1], cum[::-1]).reshape(p.shape)
    from scipy.stats import chi2

    return chi2.ppf(np.clip(mass_above, 0, 1 - 1e-12), df=2)


if __name__ == "__main__":
    main()
