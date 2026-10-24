#!/usr/bin/env python3
"""Baut alle TikZ/pgfplots-Abbildungen unter figures/<thema>/*.tex.

    python figures/build.py              # alle
    python figures/build.py slacs/f_dm   # eine

Erzeugt neben jeder .tex-Datei eine .pdf (für Arbeiten/LaTeX) und, falls
pdftocairo vorhanden ist (poppler), eine .svg (für README und Website).
Hilfsdateien landen in figures/<thema>/build/ (nicht versioniert).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def targets(args: list[str]) -> list[Path]:
    if args:
        return [ROOT / f"{a.removesuffix('.tex')}.tex" for a in args]
    return sorted(p for p in ROOT.glob("*/*.tex") if p.parent.name not in {"data", "build"})


def build(tex: Path) -> bool:
    folder = tex.parent
    if "{data/" in tex.read_text(encoding="utf-8") and not (folder / "data").exists():
        print(f"--  {tex.relative_to(ROOT)}  übersprungen (keine Daten; erst die Auswertung laufen lassen)")
        return True
    # alte Hilfsdateien dieser Abbildung entfernen (abgebrochene Läufe verwirren latexmk sonst)
    for old in (folder / "build").glob(tex.stem + ".*"):
        old.unlink()
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build", tex.name]
    if not shutil.which("latexmk"):
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", tex.name]
        (folder / "build").mkdir(exist_ok=True)
    # Reproduzierbare PDFs: feste Zeitstempel/IDs, damit Git nur echte Änderungen sieht
    env = os.environ | {"SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1"}
    r = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, env=env)
    pdf = folder / "build" / tex.with_suffix(".pdf").name
    if r.returncode != 0 or not pdf.exists():
        log = folder / "build" / tex.with_suffix(".log").name
        tail = log.read_text(errors="replace").splitlines()[-25:] if log.exists() else r.stdout.splitlines()[-25:]
        print(f"FEHLER {tex.relative_to(ROOT)}\n  " + "\n  ".join(tail))
        return False
    shutil.copy(pdf, tex.with_suffix(".pdf"))
    msg = "pdf"
    if shutil.which("pdftocairo"):
        subprocess.run(["pdftocairo", "-svg", str(tex.with_suffix(".pdf")), str(tex.with_suffix(".svg"))], check=True)
        msg += " + svg"
    print(f"ok  {tex.relative_to(ROOT)}  ({msg})")
    return True


def main(argv: list[str]) -> int:
    results = [build(t) for t in targets(argv)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
