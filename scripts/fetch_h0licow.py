#!/usr/bin/env python3
"""Download the public H0LiCOW distance posteriors to data/h0licow/.

    python scripts/fetch_h0licow.py

Source: https://github.com/shsuyu/H0LiCOW-public (folder h0licow_distance_chains).
About 120 MB in total, so the files are not version-controlled.
When using them, please cite the papers listed in the README of the H0LiCOW
repository (Wong et al. 2020, MNRAS 498, 1420 and the individual lens analyses).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

COMMIT = "57cf97357e16a94ba0065472bd4185cfbc604aa3"  # pinned version for reproducibility
BASE = f"https://raw.githubusercontent.com/shsuyu/H0LiCOW-public/{COMMIT}/h0licow_distance_chains/"
FILES = [
    "HE0435_Ddt_AO+HST.dat",
    "J1206_final.csv",
    "PG1115_AO+HST_Dd_Ddt.dat",
    "RXJ1131_AO+HST_Dd_Ddt.dat",
    "wfi2033_dt_bic.dat",
]
OUT = Path(__file__).resolve().parents[1] / "data" / "h0licow"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = OUT / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  {name}: present")
            continue
        url = BASE + urllib.request.quote(name)
        print(f"  {name}: downloading …", end="", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "lenscape-fetch/0.1"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        target.write_bytes(data)
        print(f" {len(data) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
