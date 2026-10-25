# SLACS analysis

Generated automatically by `analysis/slacs.py`. Reference cosmology: H₀ = 70, Ωₘ = 0.3, Ω_Λ = 0.7.

## Key numbers

| Quantity | Value |
|---|---|
| N | 63 |
| N_good_sigma | 58 |
| median M_E [M_sun] | 1.807e+11 |
| median R_E [kpc] | 4.176 |
| mean f_SIE (good sigma) | 0.9517 |
| std f_SIE (good sigma) | 0.09531 |
| median M/L_V (<R_E) | 6.694 |
| median f_DM(<R_E) chab | 0.5885 |
| median f_DM(<R_E) salp | 0.2834 |

## Validation

Our Einstein masses differ from the published values (Auger et al. 2009) by +0.0006 dex on average (scatter 0.0036 dex, 63 lenses).

## Cosmology

- Best fit (flat wCDM): Ωₘ = 0.02, w = -0.70, f_E = 1.041, intrinsic scatter 0.15
- Planck ΛCDM (Ωₘ = 0.3111, w = −1) lies at Δχ² = 0.90, i.e. consistent.
- Varying w from −2 to −0.4 changes the median masses by 6.3 %; varying Ωₘ from 0.1 to 0.6 by 0.4 %.

## Figures

Built with `python figures/build.py` (TikZ/pgfplots; PDF and SVG in `figures/slacs/`).

![f_SIE](../figures/slacs/f_sie.svg)

![f_DM](../figures/slacs/f_dm.svg)

![Mass vs. cosmology](../figures/slacs/mass_vs_cosmology.svg)

![Cosmography](../figures/slacs/cosmography.svg)
