# Daten

Öffentliche Kataloge aus dem CDS-Archiv (VizieR), unverändert so gespeichert, wie
`scripts/fetch_cds.py` sie lädt. Jede `ReadMe` beschreibt das Byte-genaue Format.

| Ordner | Katalog | Inhalt |
|---|---|---|
| `slacs_bolton2008/` | [J/ApJ/682/964](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/682/964) | SLACS V: 131 Kandidaten (table4), SIE- und LTM-Modelle der 63 Grade-A-Linsen (table5) |
| `slacs_hst/` | [MAST/HLA](https://hla.stsci.edu/) | 12″-Ausschnitte von HST/ACS-F814W-Bildern (HST-Programm 10886), erzeugt mit `scripts/fetch_hst_slacs.py` |
| `slacs_auger2009/` | [J/ApJ/705/1099](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/705/1099) | SLACS IX: Photometrie (table3), Einsteinmassen, Sternmassen und Sternanteile (table4) |

Bitte bei Verwendung zitieren:

- Bolton, A. S., Burles, S., Koopmans, L. V. E., et al. 2008, ApJ, 682, 964
- Auger, M. W., Treu, T., Bolton, A. S., et al. 2009, ApJ, 705, 1099

Neu laden: `python scripts/fetch_cds.py` (Kataloge), `python scripts/fetch_hst_slacs.py` (HST-Ausschnitte).

Nicht versioniert (zu groß, per Skript): H0LiCOW-Distanzketten (`data/h0licow/`, `scripts/fetch_h0licow.py`) und
Frontier-Fields-Massenkarten (`data/frontier/`, `scripts/fetch_frontier.py`).
