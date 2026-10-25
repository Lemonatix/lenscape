#!/usr/bin/env python3
"""Download Hubble Frontier Fields (HFF) mass models from the MAST archive.

    python scripts/fetch_frontier.py                     # Abell 2744 and MACS J0416, teams CATS + GLAFIC
    python scripts/fetch_frontier.py abell2744 cats      # one cluster, one team

For each model the newest version is located, and the kappa and gamma maps as
well as the deflection angles (x/y) are saved to data/frontier/<cluster>/<team>/.
The maps are normalized to D_ds/D_s = 1 (source at infinity).

Source: https://archive.stsci.edu/prepds/frontier/lensmodels/
Cite: Lotz et al. 2017 (HFF) and the respective modelling team (see the README in the model folder).
Standard library only; the files (a few 10 MB each) are not version-controlled.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://archive.stsci.edu/pub/hlsp/frontier/{cluster}/models/{team}/"
DEFAULT = [("abell2744", "cats"), ("abell2744", "glafic"), ("macs0416", "cats"), ("macs0416", "glafic")]
WANTED = ("_kappa.fits", "_gamma.fits", "_x-arcsec-deflect.fits", "_y-arcsec-deflect.fits")
OUT = Path(__file__).resolve().parents[1] / "data" / "frontier"
UA = {"User-Agent": "lenscape-fetch/0.1 (Python urllib; https://github.com/Lemonatix/lenscape)"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def links(url: str) -> list[str]:
    html = get(url).decode("utf-8", "replace")
    return [h for h in re.findall(r'href="([^"?#]+)"', html) if not h.startswith(("/", "http", ".."))]


def version_key(v: str):
    nums = [int(x) for x in re.findall(r"\d+", v)]
    return nums or [0]


def fetch(cluster: str, team: str) -> None:
    base = BASE.format(cluster=cluster, team=team)
    try:
        entries = links(base)
    except Exception as e:  # noqa: BLE001
        print(f"{cluster}/{team}: directory not readable ({e}) -> {base}")
        return
    versions = sorted((e.rstrip("/") for e in entries if e.endswith("/") and e.lower().startswith("v")),
                      key=version_key)
    if not versions:
        print(f"{cluster}/{team}: no version folders found. Contents: {entries[:20]}")
        return
    v = versions[-1]
    files = links(base + v + "/")
    chosen = [f for f in files if f.endswith(WANTED)]
    readme = [f for f in files if "readme" in f.lower()]
    out = OUT / cluster / team
    out.mkdir(parents=True, exist_ok=True)
    print(f"{cluster}/{team} {v}: {', '.join(chosen) or 'no matching maps'}")
    for f in chosen + readme[:1]:
        target = out / f
        if target.exists() and target.stat().st_size > 0:
            print(f"  {f}: present")
            continue
        data = get(base + v + "/" + f)
        target.write_bytes(data)
        print(f"  {f}: {len(data) / 1e6:.1f} MB")
    (out / "VERSION").write_text(f"{v}\n{base}{v}/\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    pairs = [(argv[0], argv[1])] if len(argv) == 2 else DEFAULT
    for cluster, team in pairs:
        fetch(cluster, team)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
