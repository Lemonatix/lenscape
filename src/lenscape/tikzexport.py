"""Hilfen, um Rechenergebnisse für TikZ/pgfplots-Abbildungen zu exportieren.

Python schreibt nur Daten; das Aussehen steckt in figures/<thema>/*.tex.
"""

from __future__ import annotations

from pathlib import Path

import contourpy
import numpy as np

DCHI2_2D = {"68": 2.30, "95": 6.17}  # Δχ²-Niveaus für zwei Parameter


def write_table(path: Path, header: list[str], *cols, fmt: str = "%.6g") -> None:
    arr = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    np.savetxt(path, arr, header=" ".join(header), comments="", fmt=fmt)


def write_macros(path: Path, values: dict[str, float]) -> None:
    """Einzelwerte als TeX-Makros (\\def\\name{wert}); Namen nur aus Buchstaben."""
    lines = [f"\\def\\{k}{{{v:.8g}}}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_regions(data_dir: Path, tag: str, x, y, delta_chi2, style_prefix: str | None = None,
                   levels: dict[str, float] = DCHI2_2D) -> Path:
    """Gefüllte Δχ²-Bereiche als Polygone + eine .tex-Datei mit den \\addplot-Befehlen.

    ``delta_chi2`` hat die Form (len(y), len(x)). Die erzeugte Datei
    ``<tag>_regions.tex`` zeichnet zuerst 95 %, dann 68 % mit den pgfplots-Stilen
    ``<style_prefix>95`` und ``<style_prefix>68``.
    """
    style_prefix = style_prefix or tag
    gen = contourpy.contour_generator(np.asarray(x), np.asarray(y), np.asarray(delta_chi2),
                                      fill_type=contourpy.FillType.OuterOffset)
    cmds = []
    for lvl in sorted(levels, key=lambda k: -levels[k]):
        polys, _ = gen.filled(-1.0, levels[lvl])
        for k, poly in enumerate(polys):
            name = f"{tag}_{lvl}_{k}.dat"
            write_table(data_dir / name, ["x", "y"], poly[:, 0], poly[:, 1])
            cmds.append(f"\\addplot[{style_prefix}{lvl}, forget plot] table {{data/{name}}} -- cycle;")
    out = data_dir / f"{tag}_regions.tex"
    out.write_text("\n".join(cmds) + "\n", encoding="utf-8")
    return out
