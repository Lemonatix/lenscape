#!/usr/bin/env python3
"""Lädt das HST/ACS-F814W-Bild einer SLACS-Linse aus MAST und schneidet es aus.

    pip install astroquery
    python scripts/fetch_hst_slacs.py                  # Standard: J0946+1006 ("Jackpot", Doppelring)
    python scripts/fetch_hst_slacs.py J1430+4105 J0029-0055

Koordinaten kommen aus dem SLACS-Katalog (Bolton et al. 2008, data/slacs_bolton2008).
Gesucht wird ein gedriztes ACS/WFC-F814W-Bild (DRZ/DRC); gespeichert wird ein
12″-Ausschnitt nach data/slacs_hst/<name>_F814W.fits (klein, versionierbar).
Das volle Bild landet in data/cache/ (nicht versioniert).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "slacs_hst"
CACHE = ROOT / "data" / "cache" / "mast"
CUTOUT_ARCSEC = 12.0


def coords_for(name: str):
    from astropy.coordinates import SkyCoord
    from astropy.io import ascii
    import astropy.units as u

    folder = ROOT / "data" / "slacs_bolton2008"
    t = ascii.read(folder / "table4.dat", readme=folder / "ReadMe", format="cds")
    row = t[t["Name"] == name]
    if len(row) == 0:
        raise SystemExit(f"{name} nicht im SLACS-Katalog")
    s = str(row["SDSS"][0])  # HHMMSS.ss+DDMMSS.s
    ra = f"{s[0:2]}h{s[2:4]}m{s[4:9]}s"
    dec = f"{s[9:12]}d{s[12:14]}m{s[14:18]}s"
    return SkyCoord(ra, dec, frame="icrs"), u


def fetch(name: str) -> None:
    from astroquery.mast import Observations
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS

    coord, u = coords_for(name)
    print(f"{name}: {coord.to_string('hmsdms')}")
    obs = Observations.query_region(coord, radius=5 * u.arcsec)
    sel = (obs["obs_collection"] == "HST") & (obs["filters"] == "F814W") & \
          np.array(["ACS/WFC" in str(i) for i in obs["instrument_name"]]) & (obs["dataproduct_type"] == "image")
    obs = obs[sel]
    if len(obs) == 0:
        print("  keine ACS/WFC-F814W-Beobachtung gefunden")
        return
    obs.sort("t_exptime", reverse=True)
    prods = Observations.get_product_list(obs[:5])
    prods = prods[np.isin(prods["productSubGroupDescription"], ["DRC", "DRZ"])]
    if len(prods) == 0:
        print("  kein gedriztes Produkt (DRZ/DRC) gefunden")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    man = Observations.download_products(prods[:1], download_dir=str(CACHE), cache=True)
    path = man["Local Path"][0]
    print(f"  {Path(path).name}")

    make_cutout(path, coord, name)


def make_cutout(path, coord, name: str) -> Path:
    """12″-Ausschnitt um ``coord`` aus einem gedrizzten Bild speichern.

    HST-Pipeline-Produkte haben die Daten in der Ebene "SCI", Produkte des
    Hubble Legacy Archive (HLA) direkt in der Primär-Ebene; genommen wird die
    erste Ebene mit einem 2D-Bild.
    """
    import astropy.units as u
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS

    with fits.open(path) as hdul:
        if "SCI" in hdul:
            sci = hdul["SCI"]
        else:
            sci = next(h for h in hdul if h.data is not None and np.ndim(h.data) == 2)
        prim = hdul[0].header
        wcs = WCS(sci.header)
        pix = abs(wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value)
        size = int(round(CUTOUT_ARCSEC / pix))
        cut = Cutout2D(sci.data, coord, (size, size), wcs=wcs)
        hdr = cut.wcs.to_header()
        hdr["BUNIT"] = sci.header.get("BUNIT", prim.get("BUNIT", ""))
        hdr["PIXSCALE"] = (pix, "arcsec/pixel")
        exptime = float(prim.get("EXPTIME", sci.header.get("EXPTIME", 0.0)) or 0.0)
        if exptime <= 1.0:
            # HLA normiert EXPTIME auf 1 s; die echte Zeit steckt in den Drizzle-Einträgen D###DEXP
            exptime = sum(float(v) for k, v in prim.items() if k.startswith("D") and k.endswith("DEXP"))
        hdr["EXPTIME"] = (exptime, "s, Summe der Einzelbelichtungen")
        hdr["PHOTFLAM"] = sci.header.get("PHOTFLAM", prim.get("PHOTFLAM", 0.0))
        hdr["ORIGFILE"] = Path(path).name
        hdr["OBJECT"] = name
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / f"{name}_F814W.fits"
        fits.PrimaryHDU(cut.data.astype(np.float32), header=hdr).writeto(target, overwrite=True)
        print(f"  Ausschnitt {size}×{size} px ({pix:.3f}″/px, BUNIT={hdr['BUNIT']}, EXPTIME={hdr['EXPTIME']}) "
              f"-> {target.relative_to(ROOT)}")
    return target


def main(argv: list[str]) -> int:
    for name in argv or ["J0946+1006"]:
        fetch(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
