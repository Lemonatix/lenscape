# lenscape – texts for the project page

## Title

lenscape: modelling gravitational lenses with real data

## Teaser (one sentence)

A Python toolkit and interactive simulator for gravitational lensing that measures real galaxies, galaxy
clusters and the expansion rate of the universe from public telescope data.

## Description

Massive galaxies and galaxy clusters bend the light of objects behind them – sometimes into a full Einstein
ring. lenscape models this effect from first principles (thin-lens equation, isothermal and NFW mass profiles,
caustics, time delays) in a cosmology of your choice, and applies the same physics to real observations:

- **63 SLACS lenses**: masses inside the Einstein radius, dark-matter fractions and a cosmography fit;
  the computed masses reproduce the published values to better than 1 %.
- **H₀ from time delays**: with the public H0LiCOW posteriors of six lensed quasars, lenscape reproduces
  H₀ = 73.3 (+1.7 −1.8) km/s/Mpc, 3.3σ away from the Planck value – the Hubble tension.
- **Galaxy clusters**: Hubble Frontier Fields mass maps of Abell 2744 and MACS J0416, compared between
  modelling teams.
- **A real Hubble image**: a forward-model fit of SDSS J0946+1006 ("Jackpot") with a pixelated source
  reconstruction and a second source plane – the double Einstein ring that is sensitive to dark energy.

The interactive demo bends your own photo: drag a black hole across it, change its mass, or slow down the
speed of light.

## Tags

Python · NumPy/SciPy · Astropy · Streamlit · WebGL2 · TikZ/pgfplots · gravitational lensing · cosmology

## Links

- Code: https://github.com/Lemonatix/lenscape
- Demo: `demo/index.html`
