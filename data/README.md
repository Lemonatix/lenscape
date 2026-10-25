# Data

Public catalogues from the CDS archive (VizieR), stored unchanged exactly as
`scripts/fetch_cds.py` downloads them. Each `ReadMe` describes the byte-exact format.

| Folder | Catalogue | Contents |
|---|---|---|
| `slacs_bolton2008/` | [J/ApJ/682/964](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/682/964) | SLACS V: 131 candidates (table4), SIE and LTM models of the 63 grade-A lenses (table5) |
| `slacs_hst/` | [MAST/HLA](https://hla.stsci.edu/) | 12″ cutouts of HST/ACS F814W images (HST program 10886), created with `scripts/fetch_hst_slacs.py` |
| `slacs_auger2009/` | [J/ApJ/705/1099](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/705/1099) | SLACS IX: photometry (table3), Einstein masses, stellar masses and stellar fractions (table4) |

Please cite when using them:

- Bolton, A. S., Burles, S., Koopmans, L. V. E., et al. 2008, ApJ, 682, 964
- Auger, M. W., Treu, T., Bolton, A. S., et al. 2009, ApJ, 705, 1099

To re-download: `python scripts/fetch_cds.py` (catalogues), `python scripts/fetch_hst_slacs.py` (HST cutouts).

Not version-controlled (too large, fetched by script): the H0LiCOW distance chains (`data/h0licow/`,
`scripts/fetch_h0licow.py`) and the Frontier Fields mass maps (`data/frontier/`, `scripts/fetch_frontier.py`).
