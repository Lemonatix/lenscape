#!/usr/bin/env python3
"""Lädt Kataloge aus dem CDS-Archiv (VizieR) nach data/<name>/.

Nur Standardbibliothek, damit es ohne installierte Abhängigkeiten läuft:

    python scripts/fetch_cds.py            # alle unten gelisteten Kataloge
    python scripts/fetch_cds.py slacs_bolton2008

Für jeden Katalog wird die ReadMe geladen, die Dateiliste ("File Summary")
daraus gelesen und jede Tabelle heruntergeladen. Die ReadMe beschreibt das
Byte-genaue Format und wird von lenscape.data zum Einlesen benutzt.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CATALOGS = {
    # SLACS V: 131 Kandidaten, 63 Grade-A-Linsen mit SIE-Modellen
    "slacs_bolton2008": "J/ApJ/682/964",
    # SLACS IX: Photometrie und Sternmassen (Chabrier/Salpeter)
    "slacs_auger2009": "J/ApJ/705/1099",
}
BASE = "https://cdsarc.cds.unistra.fr/ftp/{cat}/{file}"
ROOT = Path(__file__).resolve().parents[1] / "data"
UA = {"User-Agent": "lenscape-fetch/0.1 (Python urllib; https://github.com/Lemonatix/lenscape)"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if data.lstrip()[:15].lower().startswith(b"<!doctype html"):
        raise RuntimeError(
            f"{url} lieferte eine HTML-Seite statt Daten (Bot-Schutz?). "
            "Bitte die Datei im Browser öffnen und manuell speichern."
        )
    return data


def files_in_readme(readme: str) -> list[str]:
    """Dateinamen aus dem Block 'File Summary' der CDS-ReadMe."""
    lines = readme.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("File Summary"))
    except StopIteration:
        return []
    dashes = 0
    names = []
    for line in lines[start + 1:]:
        if line.startswith("----"):
            dashes += 1
            if dashes == 3:
                break
            continue
        if dashes == 2 and line.strip():
            name = line.split()[0]
            if name != "ReadMe" and not name.startswith("("):
                names.append(name)
    return names


def fetch(key: str) -> None:
    cat = CATALOGS[key]
    out = ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    readme = get(BASE.format(cat=cat, file="ReadMe")).decode("latin-1")
    (out / "ReadMe").write_text(readme, encoding="utf-8")
    names = files_in_readme(readme)
    print(f"{key} ({cat}): {', '.join(names) or 'keine Dateien gefunden'}")
    for name in names:
        data = get(BASE.format(cat=cat, file=name))
        (out / name).write_bytes(data)
        print(f"  {name}: {len(data):,} Bytes")


def main(argv: list[str]) -> int:
    keys = argv or list(CATALOGS)
    for key in keys:
        if key not in CATALOGS:
            print(f"Unbekannter Katalog {key!r}; bekannt: {', '.join(CATALOGS)}")
            return 1
        fetch(key)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
