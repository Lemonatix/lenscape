"""H0 aus den sechs H0LiCOW-Linsen -- und die Hubble-Spannung.

    python scripts/fetch_h0licow.py   # einmalig, ~120 MB
    python analysis/h0licow.py        # rechnen, Daten für die Abbildungen schreiben
    python figures/build.py h0/...    # Abbildungen bauen

Reproduziert Wong et al. (2020, MNRAS 498, 1420): H0 = 73.3 +1.7 -1.8 km/s/Mpc
in flachem ΛCDM und zeigt zusätzlich flaches wCDM.
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

# Vergleichswerte (Median, -Fehler, +Fehler) in km/s/Mpc
REFERENCES = {
    "Planck 2018 (CMB, ΛCDM)": (67.4, 0.5, 0.5),             # Planck Collaboration 2020, A&A 641, A6
    "SH0ES (Cepheiden + SN Ia)": (73.04, 1.04, 1.04),        # Riess et al. 2022, ApJL 934, L7
    "TDCOSMO 2025": (71.6, 3.3, 3.9),                        # TDCOSMO Collab. / Birrer et al. 2025
}
KEYS = ["B", "R", "H", "J", "W", "P"]  # Kurzzeichen für TeX-Makros (nur Buchstaben)


def main():
    lenses = load_h0licow()
    H0 = np.arange(50.0, 110.0001, 0.1)

    single = [infer_h0([lens], H0=H0) for lens in lenses]
    joint = infer_h0(lenses, H0=H0)

    # --------------------------------------------------- Randverteilungen p(H0)
    cols = [p.marginal_H0() for p in single] + [joint.marginal_H0()]
    write_table(DATA / "h0_posteriors.dat", ["H0"] + KEYS + ["all"], H0, *cols)

    # --------------------------------------------------- Messwerte für das Whisker-Diagramm
    rows = []
    macros = {}
    for key, lens, post in zip(KEYS, lenses, single):
        med, lo, hi = post.summary()
        rows.append((lens.name, med, lo, hi))
        macros[f"hzero{key}"] = med
    med, lo, hi = joint.summary()
    macros |= {"hzeromed": med, "hzerolo": lo, "hzerohi": hi}

    planck = REFERENCES["Planck 2018 (CMB, ΛCDM)"]
    shoes = REFERENCES["SH0ES (Cepheiden + SN Ia)"]
    sig = lambda a, b: abs(a[0] - b[0]) / np.hypot(0.5 * (a[1] + a[2]), 0.5 * (b[1] + b[2]))
    ours = (med, lo, hi)
    macros |= {"tensionplanck": sig(ours, planck), "tensionshoes": sig(ours, shoes)}

    # --------------------------------------------------- flaches wCDM: H0-w-Ebene
    Hw = np.arange(50.0, 120.0001, 0.25)
    w = np.linspace(-2.5, 0.5, 61)
    Om = np.linspace(0.05, 0.5, 19)
    jw = infer_h0(lenses, H0=Hw, Om=Om, w=w)
    p = np.exp(jw.logpost - jw.logpost.max())
    p_hw = p.sum(axis=1)                       # über Omega_m marginalisieren -> (w, H0)
    # Δχ² aus der marginalen Dichte über Konturen gleicher Wahrscheinlichkeitsmasse
    dchi2 = _mass_levels_as_dchi2(p_hw)
    export_regions(DATA, "hw", Hw, w, dchi2, style_prefix="lens")
    mw = jw.summary()
    macros |= {"hzerowmed": mw[0], "hzerowlo": mw[1], "hzerowhi": mw[2]}

    macros |= {"planckmed": planck[0], "planckerr": planck[1], "shoesmed": shoes[0], "shoeserr": shoes[1]}
    write_macros(DATA / "macros.tex", macros)

    # Whisker-Diagramm: von oben nach unten Einzellinsen, Kombination, Referenzen
    tex_names = {
        "B1608+656": "B1608$+$656", "RXJ1131-1231": "RXJ1131$-$1231", "HE0435-1223": "HE0435$-$1223",
        "J1206+4332": "J1206$+$4332", "WFI2033-4723": "WFI2033$-$4723", "PG1115+080": "PG1115$+$080",
    }
    entries = [(tex_names[n], m, l, h, "lens") for n, m, l, h in rows]
    entries.append((r"\textbf{H0LiCOW, 6 Linsen}", med, lo, hi, "joint"))
    entries.append((r"TDCOSMO 2025", *REFERENCES["TDCOSMO 2025"], "ref"))
    entries.append((r"SH0ES 2022", *shoes, "late"))
    entries.append((r"Planck 2018", *planck, "early"))
    n = len(entries)
    ypos = {i: n - i for i in range(n)}  # oberster Eintrag bekommt den größten y-Wert
    for group in ("lens", "joint", "ref", "late", "early"):
        sel = [(ypos[i], e) for i, e in enumerate(entries) if e[4] == group]
        write_table(DATA / f"whisker_{group}.dat", ["y", "x", "em", "ep"],
                    [y for y, _ in sel], [e[1] for _, e in sel], [e[2] for _, e in sel], [e[3] for _, e in sel])
    ticks = ",".join(str(ypos[i]) for i in range(n))
    labels = ",".join("{" + e[0] + "}" for e in entries)
    (DATA / "whisker_axis.tex").write_text(
        "\\pgfplotsset{whisker axis/.style={ytick={" + ticks + "}, yticklabels={" + labels + "},"
        f" ymin=0.4, ymax={n + 0.6}}}}}\n", encoding="utf-8")

    # Tabelle der Einzelwerte + Referenzen (für die Abbildung und das README)
    lines = ["| Messung | H₀ [km/s/Mpc] |", "|---|---|"]
    for name, m, l, h in rows:
        lines.append(f"| {name} | {m:.1f} −{l:.1f} +{h:.1f} |")
    lines.append(f"| **H0LiCOW kombiniert (flaches ΛCDM, diese Rechnung)** | **{med:.1f} −{lo:.1f} +{hi:.1f}** |")
    lines.append(f"| H0LiCOW kombiniert (flaches wCDM, diese Rechnung) | {mw[0]:.1f} −{mw[1]:.1f} +{mw[2]:.1f} |")
    for name, (m, l, h) in REFERENCES.items():
        lines.append(f"| {name} | {m} −{l} +{h} |")
    table = "\n".join(lines)

    text = f"""# H0 aus Zeitverzögerungen (H0LiCOW)

Automatisch erzeugt von `analysis/h0licow.py`.

{table}

- Veröffentlicht (Wong et al. 2020): 73,3 +1,7 −1,8 km/s/Mpc. Diese Rechnung: {med:.1f} +{hi:.1f} −{lo:.1f}.
- Abstand zu Planck: {macros['tensionplanck']:.1f}σ, zu SH0ES: {macros['tensionshoes']:.1f}σ (Gauß-Näherung).
- TDCOSMO 2025 lockert die Annahmen über das Massenprofil (Massenblatt-Entartung, über Sternkinematik
  eingeschränkt). Die Unsicherheit wächst, der Mittelwert bleibt zwischen Planck und SH0ES.

![H0-Spannung](../figures/h0/h0_tension.svg)

![Posteriors](../figures/h0/h0_posteriors.svg)

![H0 und w](../figures/h0/h0_w.svg)
"""
    (ROOT / "analysis" / "h0licow_results.md").write_text(text, encoding="utf-8")
    print(text)


def _mass_levels_as_dchi2(p):
    """Wandelt eine 2D-Dichte so in 'Δχ²' um, dass die Niveaus 2.30/6.17 genau
    68 %/95 % der Wahrscheinlichkeitsmasse einschließen (HPD-Bereiche)."""
    flat = np.sort(p.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    # Masse oberhalb jedes Dichtewerts
    mass_above = np.interp(p.ravel(), flat[::-1], cum[::-1]).reshape(p.shape)
    from scipy.stats import chi2

    return chi2.ppf(np.clip(mass_above, 0, 1 - 1e-12), df=2)


if __name__ == "__main__":
    main()
