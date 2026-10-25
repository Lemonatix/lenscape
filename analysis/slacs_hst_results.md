# HST image fit of SLACS lenses

Generated automatically by `analysis/slacs_hst_fit.py` (HST/ACS F814W, Gaussian PSF 0.1″).

| Lens | Model | θ_E | q | PA | γ_ext | χ²_red |
|---|---|---|---|---|---|---|
| J0946+1006 | Bolton et al. 2008 (SIE) | 1.38″ | 0.81 | 159° | – | – |
| | lenscape, SIE | 1.391″ | 0.78 | 162° | – | 1.64 |
| | lenscape, SIE + shear | 1.406″ | 0.92 | 42° | 0.081 | 1.47 |

The Einstein radius is robust; ellipticity and external shear are partly degenerate (both produce a quadrupole field). The pure SIE is the one comparable with the catalogue.
Cosmic rays and defective pixels are masked (grey in the residual).

![Fit](../figures/slacs_hst/hst_fit.svg)
