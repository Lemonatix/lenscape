# SDSS J0946+1006: pixelated sources and a second source plane

Generated automatically by `analysis/jackpot.py`.

Redshifts: lens 0.222, source 1 0.609 (Bolton et al. 2008), source 2 2.035 (Smith & Collett 2021).

## Inner ring with a pixelated source

- parametric Sérsic source: θ_E = 1.406″, χ²_red = 1.42
- pixelated source (36×36, gradient regularization, λ = 83.3 from the evidence): θ_E = 1.405″, q = 0.963, γ_ext = 0.073, χ²_red in the ring region = 1.16

## Second source plane

η was scanned from 1.30 to 1.60 (coarse, then fine around the maximum); the result is the maximum of the evidence. The source grid for source 2 has to be compact: with ±0.5″ a degenerate solution at small η wins, in which the source copies the barely multiply imaged image. For each η the compact grid is centred on the roughly reconstructed source (tracking grid). Without the high-pass the inversion exploits large-scale lens-light residuals (worse χ²); this variant is listed only as a control and does not enter the systematics.

| Analysis | global maximum η | solution A (η ≈ 1.46) | solution B (η ≈ 1.31) | ln E(A) − ln E(B) |
|---|---|---|---|---|
| baseline (high-pass, grid ±0.30″) | 1.460 | 1.460 | 1.320 | +47 |
| grid ±0.25″ | 1.460 | 1.460 | 1.288 | +89 |
| grid ±0.35″ | 1.312 | 1.456 | 1.312 | -30 |
| mask from 1.95″ | 1.320 | 1.460 | 1.320 | -8 |
| curvature regularization | 1.460 | 1.460 | 1.312 | +32 |
| no high-pass (control only) | 1.308 | 1.460 | 1.308 | -23 |

- The evidence is **bimodal**: solution A (η ≈ 1.46) and solution B (η ≈ 1.31). Which one wins depends on analysis details; the data (a single F814W exposure, 1048 s) do not separate them reliably. Solution B lies at the edge of the plausible range and corresponds to a barely multiply imaged outer ring.
- Solution A: θ_E2/θ_E1 = 1.460 ± 0.004 (spread of the variants) – consistent with Gavazzi et al. (2008), who found 2.07″/1.43″ = 1.45 from deeper data.
- Mass of source 1 as an SIS (solution A tracked continuously): b_s1 = 0.1″ → η = 1.400, b_s1 = 0.2″ → η = 1.340. More mass in source 1 deflects source 2 additionally, so the main lens has to do less.
- Flat ΛCDM (Ωₘ = 0.315) predicts η = 1.388. Isothermal and without mass in source 1, the measured η would correspond to w ≈ -0.1; with γ′ = 1.87 instead of 2 it would be exactly ΛCDM.
- Collett & Auger (2014) marginalize over the slope γ′ and the mass of source 1 and find η = 1.404 ± 0.016; with the spectroscopic redshift z = 2.035 (Smith & Collett 2021) and a CMB prior this gives w = −1.04 ± 0.20.

Conclusion: the inner ring is very well constrained with a pixelated source. For the faint outer ring this single exposure is not enough for an unambiguous measurement: the physically plausible solution A reproduces the literature value, but a second solution cannot be excluded with confidence. Even with an unambiguous geometry the cosmology would hinge on the profile slope γ′ and the mass of source 1 – exactly the quantities that research constrains with deeper multi-band data and stellar kinematics.

![Jackpot](../figures/jackpot/jackpot_fit.svg)

![Cosmology](../figures/jackpot/jackpot_eta.svg)
