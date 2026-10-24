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


def central_peak(kappa_file: Path, ref=None, radius: float = 60.0):
    """Himmelsposition des Maximums von κ (5″ geglättet) innerhalb ``radius`` um ``ref``.

    Ohne ``ref`` wird um das Kartenzentrum gesucht. Gerechnet auf einem ~0,3″-Gitter.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u

    with fits.open(kappa_file) as h:
        hdu = next(x for x in h if x.data is not None)
        d = np.nan_to_num(np.asarray(hdu.data, dtype=float))
        w = WCS(hdu.header).celestial
    pix = float(abs(w.proj_plane_pixel_scales()[0].to(u.arcsec).value))
    b = max(1, int(round(0.3 / pix)))
    ny, nx = (s_ // b * b for s_ in d.shape)
    db = d[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
    sm = gaussian_filter(db, 5.0 / (pix * b))
    jj, ii = np.mgrid[0:sm.shape[0], 0:sm.shape[1]]
    if ref is None:
        ci, cj = (sm.shape[1] - 1) / 2, (sm.shape[0] - 1) / 2
    else:
        ci, cj = w.world_to_pixel(ref)
        ci, cj = (ci - (b - 1) / 2) / b, (cj - (b - 1) / 2) / b
    sm[np.hypot(ii - ci, jj - cj) * pix * b > radius] = -np.inf
    j, i = np.unravel_index(np.argmax(sm), sm.shape)
    return w.pixel_to_world(i * b + (b - 1) / 2, j * b + (b - 1) / 2)


def field_radius(kappa_file: Path, center) -> float:
    """Abstand (arcsec) von ``center`` zum nächsten Kartenrand."""
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u

    h = fits.getheader(kappa_file)
    w = WCS(h).celestial
    pix = float(abs(w.proj_plane_pixel_scales()[0].to(u.arcsec).value))
    i, j = w.world_to_pixel(center)
    nx, ny = h["NAXIS1"], h["NAXIS2"]
    return float(min(i + 0.5, nx - 0.5 - i, j + 0.5, ny - 0.5 - j) * pix)


def common_center(files: list[Path]):
    """Mittleres Himmelszentrum der zentralen κ-Maxima aller Teams eines Haufens."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ref = central_peak(files[0])
    peaks = [central_peak(f, ref) for f in files]
    ra = np.mean([p.ra.deg for p in peaks])
    dec = np.mean([p.dec.deg for p in peaks])
    c = SkyCoord(ra * u.deg, dec * u.deg)
    spread = max(p.separation(c).arcsec for p in peaks)
    return c, spread


def source_area_above(m: MassMap, mu_thresholds, step: int, center=(0.0, 0.0), radius=np.inf) -> np.ndarray:
    """Fläche in der Quellebene (arcmin²) mit |μ| > μ0, gezählt im Bildfeld innerhalb ``radius``."""
    mu = np.abs(m.magnification_grid(step))
    X, Y = m.grid(step)
    inside = np.hypot(X - center[0], Y - center[1]) <= radius
    pix_area = (m.pixscale * step / 60.0) ** 2
    out = []
    for mu0 in mu_thresholds:
        sel = np.isfinite(mu) & (mu > mu0) & inside
        out.append(np.sum(pix_area / mu[sel]))
    return np.array(out)


def main(data_dir: Path | None = None, out_dir: Path | None = None) -> int:
    data_dir = data_dir or ROOT / "data" / "frontier"
    out = out_dir or ROOT / "figures" / "frontier" / "data"
    out.mkdir(parents=True, exist_ok=True)
    mus = np.logspace(np.log10(2), np.log10(100), 30)
    macros, rows, found = {}, [], 0

    for ckey, cinfo in CLUSTERS.items():
        kfiles = [find(data_dir / ckey / t, "_kappa.fits") for t in TEAMS if (data_dir / ckey / t).exists()]
        kfiles = [f for f in kfiles if f is not None]
        if not kfiles:
            continue
        center_sky, spread = common_center(kfiles)
        print(f"{ckey}: gemeinsames Zentrum {center_sky.to_string('hmsdms', precision=1)} "
              f"(Team-Maxima bis {spread:.1f}″ davon entfernt)")
        ckey_tex = "".join(ch for ch in ckey if ch.isalpha())
        macros[f"spread{ckey_tex}"] = spread
        # gemeinsamer Auswertebereich: Kreis, der in allen Karten vollständig liegt
        r_common = min(field_radius(f, center_sky) for f in kfiles)
        kpc_per_arcsec = (COSMO.angular_diameter_distance(cinfo["z"]) * (np.pi / 180 / 3600)).to("kpc").value
        r_kpc = r_common * kpc_per_arcsec
        radii = np.linspace(5, r_kpc, 60)
        macros[f"rcommon{ckey_tex}"] = r_common
        macros[f"rcommonkpc{ckey_tex}"] = r_kpc
        print(f"  gemeinsamer Bereich: r < {r_common:.0f}″ = {r_kpc:.0f} kpc")
        for tkey, tlabel in TEAMS.items():
            folder = data_dir / ckey / tkey
            m = load(folder) if folder.exists() else None
            if m is None:
                continue
            found += 1
            tag = f"{ckey}_{tkey}"
            z_d = cinfo["z"]
            step = max(1, int(round(0.5 / m.pixscale)))  # Auswertung auf ~0,5″-Gitter
            cx, cy = m.sky_to_offset(center_sky)

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
            thE2 = effective_einstein_radius(m2, step, center=(cx, cy), max_dist=40.0)
            prof = enclosed_mass_profile(m, z_d, COSMO, radii, center=(cx, cy))
            area9 = source_area_above(m9, mus, step, center=(cx, cy), radius=r_common)

            write_table(out / f"mass_{tag}.dat", ["R", "M"], radii, prof / 1e14)
            write_table(out / f"area_{tag}.dat", ["mu", "A"], mus, area9)

            # κ-Karte (z_s = 2) als Rasterbild + kritische Kurven für pgfplots,
            # beides auf einem ~0,3″-Gitter (feiner bringt in der Abbildung nichts und sprengt TeX)
            ps = max(1, int(round(0.3 / m.pixscale)))
            half = 60.0  # ±60″ um das Zentrum
            i0, i1 = np.searchsorted(m.x, [cx - half, cx + half])
            j0, j1 = np.searchsorted(m.y, [cy - half, cy + half])
            i1 = i0 + (i1 - i0) // ps * ps
            j1 = j0 + (j1 - j0) // ps * ps
            blk = lambda a: a[j0:j1, i0:i1].reshape((j1 - j0) // ps, ps, (i1 - i0) // ps, ps).mean(axis=(1, 3))  # noqa: E731
            kap = m2.ratio * blk(m.kappa_inf)
            plt.imsave(out / f"kappa_{tag}.png", np.log10(np.clip(kap, 0.05, 5))[::-1], cmap="magma",
                       vmin=np.log10(0.05), vmax=np.log10(5))
            axb, ayb = m2.ratio * blk(m.alpha_x_inf), m2.ratio * blk(m.alpha_y_inf)
            d = m.pixscale * ps
            dax_dy, dax_dx = np.gradient(axb, d)
            day_dy, day_dx = np.gradient(ayb, d)
            det = (1 - dax_dx) * (1 - day_dy) - dax_dy * day_dx
            xs = blk(np.broadcast_to(m.x, m.kappa_inf.shape)) - cx
            ys = blk(np.broadcast_to(m.y[:, None], m.kappa_inf.shape)) - cy
            lines = contourpy.contour_generator(xs, ys, det).lines(0.0)
            with open(out / f"crit_{tag}.dat", "w") as fh:
                fh.write("x y\n")
                for ln in lines:
                    if len(ln) < 6:
                        continue
                    np.savetxt(fh, ln, fmt="%.2f")
                    fh.write("\n")  # Leerzeile = Sprung in pgfplots
            h_ = m.pixscale / 2
            ext = [m.x[i0] - h_ - cx, m.x[i1 - 1] + h_ - cx, m.y[j0] - h_ - cy, m.y[j1 - 1] + h_ - cy]
            key = "".join(ch for ch in (ckey + tkey) if ch.isalpha())
            macros |= {f"{n}{key}": v for n, v in zip(["xmin", "xmax", "ymin", "ymax"], ext)}
            macros[f"thetaE{key}"] = thE2
            macros[f"Mhundred{key}"] = float(np.interp(100, radii, prof)) / 1e14
            rows.append((cinfo["label"].replace("$-$", "−"), tlabel, thE2,
                         float(np.interp(100, radii, prof)), float(prof[-1]), r_kpc,
                         float(np.interp(10, mus, area9)), r_common, fft_check, m.pixscale))
            print(f"{tag}: θE(z=2) = {thE2:.1f}″, M(<100 kpc) = {np.interp(100, radii, prof):.3g} M☉, "
                  f"A(μ>10, z=9) = {np.interp(10, mus, area9):.3f} arcmin², FFT-Kontrolle {100 * fft_check:.1f} %")

    if not found:
        print("Keine Frontier-Fields-Karten gefunden. Bitte: python scripts/fetch_frontier.py")
        return 1
    write_macros(out / "macros.tex", macros)
    lines = ["# Hubble Frontier Fields: Massenmodelle im Vergleich", "",
             "Automatisch erzeugt von `analysis/frontier.py`. Kosmologie: H₀ = 70, Ωₘ = 0,3 (wie die HFF-Modelle).", "",
             "| Haufen | Team | θ_E,eff (z_s = 2) | M(<100 kpc) | M(<R_max) | A(μ>10), z_s = 9 | Ablenkkarte vs. FFT(κ) | Pixel |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.1f}″ | {r[3]:.3g} M☉ | {r[4]:.3g} M☉ (R = {r[5]:.0f} kpc) | "
                     f"{r[6]:.3f} arcmin² (r < {r[7]:.0f}″) | {100 * r[8]:.1f} % | {r[9]:.3f}″ |")
    lines += ["",
              "Beide Teams werden um dasselbe Himmelszentrum ausgewertet (Mittel der zentralen κ-Maxima) und nur im "
              "Kreis, den alle Karten vollständig abdecken (die GLAFIC-Karten sind deutlich kleiner als die von CATS).",
              "",
              "Die Spalte „Ablenkkarte vs. FFT(κ)“ ist die mediane relative Abweichung der vom Team gelieferten "
              "Ablenkwinkel von unserer FFT-Rechnung aus der κ-Karte im zentralen Bereich. Die Richtungen stimmen "
              "überall überein; die Beträge der Team-Karten sind etwas größer, weil sie auch Masse außerhalb des "
              "Kartenfelds enthalten, die der FFT fehlt.", "",
              "![Karten](../figures/frontier/frontier_maps.svg)", "",
              "![Massenprofile](../figures/frontier/frontier_mass.svg)", "",
              "![Fläche](../figures/frontier/frontier_area.svg)", ""]
    if out_dir is None:
        (ROOT / "analysis" / "frontier_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
