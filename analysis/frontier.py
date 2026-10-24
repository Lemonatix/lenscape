"""Hubble Frontier Fields: Massenkarten zweier Haufen und zweier Modellgruppen vergleichen.

    python scripts/fetch_frontier.py   # einmalig (MAST)
    python analysis/frontier.py        # rechnen, Daten für die Abbildungen schreiben
    python figures/build.py frontier/...

Für jede Karte:
* M(<R): projizierte Masse um das Massenzentrum (bis 300 kpc)
* θ_E,eff für eine Quelle bei z = 2 (Fläche innerhalb der tangentialen kritischen Kurve)
* Quellebenen-Fläche mit Vergrößerung > μ für Quellen bei z = 9 -- wie viel "Suchfläche"
  ein Haufen als kosmisches Teleskop für die ersten Galaxien bietet
* Kontrolle: Ablenkkarten des Teams gegen unsere FFT-Ablenkung aus κ
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import contourpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from lenscape.clustermaps import MassMap, deflection_from_kappa, effective_einstein_radius, enclosed_mass_profile
from lenscape.cosmology import LensGeometry, make_cosmology
from lenscape.tikzexport import write_macros, write_table

ROOT = Path(__file__).resolve().parents[1]
CLUSTERS = {
    "abell2744": {"z": 0.308, "label": "Abell 2744"},
    "macs0416": {"z": 0.396, "label": "MACS J0416.1$-$2403"},
}
TEAMS = {"cats": "CATS", "glafic": "GLAFIC", "sharon": "Sharon", "williams": "Williams", "keeton": "Keeton"}
COSMO = make_cosmology(H0=70, Om0=0.3)  # HFF-Modelle sind in dieser Kosmologie angegeben


def find(folder: Path, suffix: str) -> Path | None:
    hits = sorted(folder.glob(f"*{suffix}"))
    return hits[0] if hits else None


def load(folder: Path) -> MassMap | None:
    k = find(folder, "_kappa.fits")
    if k is None:
        return None
    return MassMap.from_fits(k, find(folder, "_x-arcsec-deflect.fits"), find(folder, "_y-arcsec-deflect.fits"),
                             name=folder.name)


def mass_center(m: MassMap) -> tuple[float, float]:
    sm = gaussian_filter(m.kappa_inf, 5.0 / m.pixscale)  # 5″ glätten, dann Maximum
    j, i = np.unravel_index(np.argmax(sm), sm.shape)
    return float(m.x[i]), float(m.y[j])


def source_area_above(m: MassMap, mu_thresholds, step: int) -> np.ndarray:
    """Fläche in der Quellebene (arcmin²) mit |μ| > μ0."""
    mu = np.abs(m.magnification_grid(step))
    pix_area = (m.pixscale * step / 60.0) ** 2
    out = []
    for mu0 in mu_thresholds:
        sel = np.isfinite(mu) & (mu > mu0)
        out.append(np.sum(pix_area / mu[sel]))
    return np.array(out)


def main(data_dir: Path | None = None, out_dir: Path | None = None) -> int:
    data_dir = data_dir or ROOT / "data" / "frontier"
    out = out_dir or ROOT / "figures" / "frontier" / "data"
    out.mkdir(parents=True, exist_ok=True)
    radii = np.linspace(5, 300, 60)
    mus = np.logspace(np.log10(2), np.log10(100), 30)
    macros, rows, found = {}, [], 0

    for ckey, cinfo in CLUSTERS.items():
        for tkey, tlabel in TEAMS.items():
            folder = data_dir / ckey / tkey
            m = load(folder) if folder.exists() else None
            if m is None:
                continue
            found += 1
            tag = f"{ckey}_{tkey}"
            z_d = cinfo["z"]
            step = max(1, int(round(0.5 / m.pixscale)))  # Auswertung auf ~0,5″-Gitter
            cx, cy = mass_center(m)

            # Kontrolle der Ablenkkarten gegen FFT aus κ (zentrales Drittel, Randeffekte vermeiden).
            # Große Karten werden dafür auf <= 1500 Pixel pro Achse gemittelt.
            b = max(1, int(np.ceil(max(m.kappa_inf.shape) / 1500)))
            ny, nx = (s // b * b for s in m.kappa_inf.shape)
            binned = lambda a: a[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
            kb, axb, ayb = binned(m.kappa_inf), binned(m.alpha_x_inf), binned(m.alpha_y_inf)
            fx, fy = deflection_from_kappa(kb, m.pixscale * b)
            sl = (slice(kb.shape[0] // 3, 2 * kb.shape[0] // 3), slice(kb.shape[1] // 3, 2 * kb.shape[1] // 3))
            dev = np.hypot(fx - axb, fy - ayb)[sl]
            amp = np.hypot(axb, ayb)[sl]
            fft_check = float(np.median(dev / np.maximum(amp, 1e-3)))

            m2 = m.at_source_redshift(LensGeometry(z_d, 2.0, COSMO))
            m9 = m.at_source_redshift(LensGeometry(z_d, 9.0, COSMO))
            thE2 = effective_einstein_radius(m2, step)
            prof = enclosed_mass_profile(m, z_d, COSMO, radii, center=(cx, cy))
            area9 = source_area_above(m9, mus, step)

            write_table(out / f"mass_{tag}.dat", ["R", "M"], radii, prof / 1e14)
            write_table(out / f"area_{tag}.dat", ["mu", "A"], mus, area9)

            # κ-Karte (z_s = 2) als Rasterbild + kritische Kurven für pgfplots
            half = 60.0  # ±60″ um das Massenzentrum
            i0, i1 = np.searchsorted(m.x, [cx - half, cx + half])
            j0, j1 = np.searchsorted(m.y, [cy - half, cy + half])
            kap = m2.ratio * m.kappa_inf[j0:j1, i0:i1]
            plt.imsave(out / f"kappa_{tag}.png", np.log10(np.clip(kap, 0.05, 5))[::-1], cmap="magma",
                       vmin=np.log10(0.05), vmax=np.log10(5))
            mu = m2.magnification_grid(1)[j0:j1, i0:i1]
            X, Y = np.meshgrid(m.x[i0:i1] - cx, m.y[j0:j1] - cy)
            lines = contourpy.contour_generator(X, Y, 1.0 / mu).lines(0.0)
            with open(out / f"crit_{tag}.dat", "w") as fh:
                fh.write("x y\n")
                for ln in lines:
                    if len(ln) < 8:
                        continue
                    np.savetxt(fh, ln, fmt="%.3f")
                    fh.write("\n")  # Leerzeile = Sprung in pgfplots
            ext = [m.x[i0] - cx, m.x[i1 - 1] - cx, m.y[j0] - cy, m.y[j1 - 1] - cy]
            key = "".join(ch for ch in (ckey + tkey) if ch.isalpha())
            macros |= {f"{n}{key}": v for n, v in zip(["xmin", "xmax", "ymin", "ymax"], ext)}
            macros[f"thetaE{key}"] = thE2
            macros[f"Mhundred{key}"] = float(np.interp(100, radii, prof)) / 1e14
            rows.append((cinfo["label"].replace("$-$", "−"), tlabel, thE2,
                         float(np.interp(100, radii, prof)), float(np.interp(300, radii, prof)),
                         float(np.interp(10, mus, area9)), fft_check, m.pixscale))
            print(f"{tag}: θE(z=2) = {thE2:.1f}″, M(<100 kpc) = {np.interp(100, radii, prof):.3g} M☉, "
                  f"A(μ>10, z=9) = {np.interp(10, mus, area9):.3f} arcmin², FFT-Kontrolle {100 * fft_check:.1f} %")

    if not found:
        print("Keine Frontier-Fields-Karten gefunden. Bitte: python scripts/fetch_frontier.py")
        return 1
    write_macros(out / "macros.tex", macros)
    lines = ["# Hubble Frontier Fields: Massenmodelle im Vergleich", "",
             "Automatisch erzeugt von `analysis/frontier.py`. Kosmologie: H₀ = 70, Ωₘ = 0,3 (wie die HFF-Modelle).", "",
             "| Haufen | Team | θ_E,eff (z_s = 2) | M(<100 kpc) | M(<300 kpc) | A(μ>10), z_s = 9 | Ablenkkarte vs. FFT(κ) | Pixel |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.1f}″ | {r[3]:.3g} M☉ | {r[4]:.3g} M☉ | {r[5]:.3f} arcmin² | "
                     f"{100 * r[6]:.1f} % | {r[7]:.3f}″ |")
    lines += ["", "Die Spalte „Ablenkkarte vs. FFT(κ)“ ist die mediane relative Abweichung der vom Team gelieferten "
              "Ablenkwinkel von unserer FFT-Rechnung aus der κ-Karte im zentralen Bereich (Konsistenzprüfung).", "",
              "![Karten](../figures/frontier/frontier_maps.svg)", "",
              "![Massenprofile](../figures/frontier/frontier_mass.svg)", "",
              "![Fläche](../figures/frontier/frontier_area.svg)", ""]
    if out_dir is None:
        (ROOT / "analysis" / "frontier_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
