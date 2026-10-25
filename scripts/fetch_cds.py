#!/usr/bin/env python3
"""Download catalogues from the CDS archive (VizieR) to data/<name>/.

Standard library only, so it runs without any installed dependencies:

    python scripts/fetch_cds.py            # all catalogues listed below
    python scripts/fetch_cds.py slacs_bolton2008

For each catalogue the ReadMe is downloaded, the file list ("File Summary")
is read from it, and every table is downloaded. The ReadMe describes the
byte-exact format and is used by lenscape.catalogs for reading.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CATALOGS = {
    # SLACS V: 131 candidates, 63 grade-A lenses with SIE models
    "slacs_bolton2008": "J/ApJ/682/964",
    # SLACS IX: photometry and stellar masses (Chabrier/Salpeter)
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
            f"{url} returned an HTML page instead of data (bot protection?). "
            "Please open the file in a browser and save it manually."
        )
    return data


def files_in_readme(readme: str) -> list[str]:
    """File names from the 'File Summary' block of the CDS ReadMe."""
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
        # file names start at the beginning of a line; indented lines continue
        # the description of the previous file.
        if dashes == 2 and line.strip() and not line[0].isspace():
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
    print(f"{key} ({cat}): {', '.join(names) or 'no files found'}")
    for name in names:
        data = get(BASE.format(cat=cat, file=name))
        (out / name).write_bytes(data)
        print(f"  {name}: {len(data):,} bytes")


def main(argv: list[str]) -> int:
    keys = argv or list(CATALOGS)
    for key in keys:
        if key not in CATALOGS:
            print(f"Unknown catalogue {key!r}; known: {', '.join(CATALOGS)}")
            return 1
        fetch(key)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
