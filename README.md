# lenscape

Model gravitational lenses, explore them interactively and analyse them with real observational data.

The idea: the same physics that produces an Einstein ring in the simulator should be able to measure real
lenses. That yields masses of galaxies and clusters, the dark-matter fraction and, through the distances, the
influence of cosmology ($\Omega_\mathrm{m}$, the dark-energy equation of state $w$, $H_0$).

## The physics in one line

A light ray that passes the lens at angle $\vec\theta$ comes from the source position

$$
\vec\beta = \vec\theta - \vec\alpha(\vec\theta), \qquad
\theta_\mathrm{E}^2 = \frac{4GM}{c^2}\,\frac{D_\mathrm{ds}}{D_\mathrm{d}\,D_\mathrm{s}} .
$$

Cosmology enters only through the angular diameter distances $D_\mathrm{d}$, $D_\mathrm{s}$ and
$D_\mathrm{ds}$, i.e. through the critical surface density

$$
\Sigma_\mathrm{crit} = \frac{c^2}{4\pi G}\,\frac{D_\mathrm{s}}{D_\mathrm{d}\,D_\mathrm{ds}} .
$$

## Contents

| Module | Contents |
|---|---|
| `lenscape.cosmology` | wCDM cosmology, distances, Σ_crit, Einstein radii, M(<θ_E), time-delay distance |
| `lenscape.lenses` | point mass, SIS, SIE (elliptical), NFW halo, external shear, superpositions |
| `lenscape.sources` | Gaussian, Sérsic and image sources |
| `lenscape.raytrace` | ray tracing with supersampling, convergence and magnification maps |
| `lenscape.caustics` | critical curves and caustics |
| `lenscape.solver` | image positions and magnifications of a point source |
| `lenscape.catalogs` | reading real lens catalogues (SLACS, CDS/VizieR format) |
| `lenscape.strong_lenses` | Einstein masses, σ_SIE, M/L and dark-matter fraction of a sample |
| `lenscape.timedelay` | Fermat potential and time delays between images |
| `lenscape.h0licow`, `lenscape.h0_inference` | the six H0LiCOW lenses and the H₀ posterior |
| `lenscape.clustermaps` | grid-based mass maps (e.g. Frontier Fields) as lenses, FFT deflection, mass profiles |
| `lenscape.imagefit` | fit a lens model directly to an image (light, SIE + shear, source, PSF) |
| `lenscape.pixsrc` | pixelated source reconstruction with Bayesian regularization |
| `lenscape.cosmography` | Ωₘ–w fit from Einstein radii and velocity dispersions |
| `analysis/` | analysis scripts with results and figures |
| `app/` | interactive simulator (Streamlit) |
| `web/` | side project for the website: upload your own photo and bend it |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[app,dev]"
```

## Running the simulator

```bash
streamlit run app/streamlit_app.py
```

Sliders for cosmology ($H_0$, $\Omega_\mathrm{m}$, $\Omega_\Lambda$, $w$), redshifts, mass model and source.
Formulas, labels and key numbers are typeset with LaTeX (KaTeX in the browser, mathtext in the plots). Views:

- **Simulator**: the source plane with caustics next to the observed image plane with critical curves; the images
  of the source centre are listed with magnification, parity and time delay.
- **Convergence & magnification**: maps of $\kappa$ and $|\mu|$.
- **Dark energy & cosmology**: how much does the mass inferred from an observed $\theta_\mathrm{E}$ depend on
  $w$ and $\Omega_\mathrm{m}$?
- **SLACS sample** and **H₀ from time delays** (see below, once the data are downloaded).

## As a library

```python
from lenscape import LensGeometry, SIE, ExternalShear, make_cosmology
from lenscape.solver import find_images

geom = LensGeometry(z_d=0.3, z_s=1.0, cosmo=make_cosmology(Om0=0.31, w0=-1.0))
lens = SIE(geom.theta_E_sis(250), q=0.7) + ExternalShear(0.05, 0.0)

print(geom.mass_in_theta_E(geom.theta_E_sis(250)))   # M_sun inside θ_E
print(find_images(lens, 0.05, 0.03))                 # 4 images: x, y, μ
```

## Real data: 63 SLACS lenses

```bash
python scripts/fetch_cds.py     # only needed if data/ is missing
python analysis/slacs.py        # compute; data tables to figures/slacs/data/, results to analysis/slacs_results.md
python figures/build.py         # build the TikZ/pgfplots figures (PDF + SVG, needs LaTeX with pgfplots)
```

The figures are pure TikZ/pgfplots (`figures/slacs/*.tex`, shared style in `figures/lenscape-style.tex`). The
PDFs can be included directly in LaTeX documents, the SVGs in the README and on the website. Python only
provides the data; the look lives entirely in TeX.

![Cosmography](figures/slacs/cosmography.svg)

Results (details in [`analysis/slacs_results.md`](analysis/slacs_results.md)):

- **Validation:** the computed Einstein masses reproduce the published values of Auger et al. (2009) to
  0.004 dex (< 1 %).
- **Isothermal galaxies:** stellar velocities and lensing mass agree,
  $\langle\sigma_\mathrm{SDSS}/\sigma_\mathrm{SIE}\rangle \approx 0.95$ with about 10 % scatter.
- **Dark matter:** inside the Einstein radius (≈ 4 kpc) roughly half of the mass is dark matter (Chabrier IMF).
  With the heavier Salpeter IMF it becomes negative for some galaxies, so that IMF is too heavy there.
- **Cosmology:** cosmography from 58 lenses is consistent with Planck ΛCDM but hardly constrains $w$. The
  forecast shows that this method only gives useful limits with thousands of lenses (Euclid, LSST).

Every SLACS lens can be selected in the simulator ("Real lens (SLACS)"). Its Einstein radius is observed, so the
cosmology sliders change the inferred mass and dark-matter fraction.

## H₀ from time delays (H0LiCOW)

```bash
python scripts/fetch_h0licow.py   # once; public distance posteriors (~120 MB, not version-controlled)
python analysis/h0licow.py        # results to analysis/h0licow_results.md
python figures/build.py           # figures
```

Light from the images of a lensed quasar arrives at different times. The delay and a lens model give the
time-delay distance

$$
D_{\Delta t} = (1 + z_\mathrm{d})\,\frac{D_\mathrm{d}\,D_\mathrm{s}}{D_\mathrm{ds}} \propto \frac{1}{H_0} .
$$

With the public posteriors of the six H0LiCOW lenses, lenscape reproduces the published result
$H_0 = 73.3^{+1.7}_{-1.8}$ km/s/Mpc (Wong et al. 2020), 3.3σ above Planck. In the simulator the image table
shows the delays in days; the "H₀ from time delays" tab lets you drop or add individual lenses.

![H0 tension](figures/h0/h0_tension.svg)

## Galaxy clusters: Hubble Frontier Fields

```bash
python scripts/fetch_frontier.py   # mass maps from MAST (Abell 2744, MACS J0416; teams CATS, GLAFIC)
python analysis/frontier.py        # mass profiles, Einstein radii, survey area for galaxies at z = 9
```

Every map becomes a lens just like the analytic models. The modelling teams are compared with each other; as a
check, lenscape recomputes the deflection angles from $\kappa$ via FFT. Result (details in
[`analysis/frontier_results.md`](analysis/frontier_results.md)): CATS and GLAFIC give nearly the same effective
Einstein radius for both clusters (Abell 2744: 21.4″ / 21.8″, MACS J0416: 24.7″ / 25.1″ for $z_\mathrm{s} = 2$)
and the same mass within 100 kpc to 1–3 %.

![Frontier Fields](figures/frontier/frontier_maps.svg)

## Modelling a real HST image

```bash
pip install -e ".[data]"
python scripts/fetch_hst_slacs.py J0946+1006   # cutout from the HST archive
python analysis/slacs_hst_fit.py J0946+1006    # fit and comparison with Bolton et al. (2008)
```

A forward model of lens light, SIE + shear, lensed source and PSF, fitted in three stages; cosmic rays are
masked. On the real HST image of J0946+1006 the pure SIE gives $\theta_\mathrm{E} = 1.391''$, $q = 0.78$,
PA = 162°, close to Bolton et al. (2008: 1.38″, 0.81, 159°). On synthetic images the fit recovers
$\theta_\mathrm{E}$ to better than 0.1 %; the lens models agree with *lenstronomy* to $10^{-10}$
(`pip install -e ".[validate]"`, then `pytest`).

![HST fit](figures/slacs_hst/hst_fit.svg)

### Pixelated sources and a second source plane (J0946+1006, "Jackpot")

```bash
python analysis/jackpot.py   # ~3 min; results in analysis/jackpot_results.md
```

`lenscape.pixsrc` reconstructs the source freely on a pixel grid (Bayesian semi-linear inversion after
Suyu et al. 2006); the regularization strength and the lens parameters follow from the Bayesian evidence. For
the inner ring this lowers $\chi^2_\mathrm{red}$ from 1.42 (smooth Sérsic source) to 1.16.

The faint outer ring belongs to a second source at $z = 2.035$. The ratio

$$
\eta = \frac{D_\mathrm{ds_2}/D_\mathrm{s_2}}{D_\mathrm{ds_1}/D_\mathrm{s_1}}
$$

depends only on $\Omega_\mathrm{m}$ and $w$. The evidence shows two solutions; the plausible one
($\eta = 1.46$ for an isothermal profile) reproduces Gavazzi et al. (2008). With some mass in source 1 it drops to
≈ 1.40, close to Collett & Auger (2014: 1.404 ± 0.016) and ΛCDM (1.388). A real statement about cosmology needs
deeper data and a constraint on the profile slope – the figure shows why.

![Jackpot](figures/jackpot/jackpot_eta.svg)

## Web: photo lens

`web/index.html` is a single file. Open it in the browser, upload an image (or drag & drop it) and move the lens
with the mouse. Everything is computed per pixel on the GPU (WebGL2 shader, with a CPU fallback); the lens is a
point mass/black hole or an elliptical galaxy (SIE). The formulas are typeset with KaTeX from a CDN; without
network access the page still works. To embed it in your own website without the header use
`<iframe src="…/index.html?embed">`. The sliders control mass, distances, image width and the
**speed of light** ($\theta_\mathrm{E} \propto 1/c$). Scenarios: a galaxy, the Sun as in Eddington's 1919
eclipse, the same Sun at 1 % of $c$, and a marble-sized black hole in your room.

## Gallery and showcase

- [`gallery/`](gallery/README.md): square high-resolution pictures (HST image of the Jackpot, high-resolution
  lenscape model, simulated rings, Frontier Fields map), e.g. for a profile picture –
  `python scripts/make_gallery.py`.
- [`showcase/`](showcase/README.md): everything for the project page on the website (demo, cover, thumbnail,
  screenshots, figures, texts, example markup) – `python scripts/make_showcase.py --screenshots`.

## Tests

```bash
pytest
```

Among other things the tests check the Poisson equation $\nabla\cdot\vec\alpha = 2\kappa$ for every model, that
$\vec\alpha$ is curl-free, analytic image positions and magnifications of the SIS, and
$M(<\theta_\mathrm{E}) = M$ for a point mass.

## Roadmap

- [x] Simulator and cosmology (point mass, SIS, SIE, NFW, shear, caustics, image solver)
- [x] Web demo "bend your own image"
- [x] **Strong lenses from real data**: SLACS catalogue, masses, dark-matter fraction, cosmography
- [x] **Galaxy clusters**: read and compare Frontier Fields mass maps (data via script)
- [x] **Time delays**: Fermat potential, H₀ from the six H0LiCOW lenses
- [x] **Model fitting**: SIE + shear directly on HST images, cross-check with `lenstronomy`
- [x] **Web demo** with WebGL2 and embedding mode
- [x] **Pixelated sources** (Suyu et al. 2006) and the **second source plane** of the Jackpot system
- [ ] Elliptical power-law profile (free slope γ′) for the Jackpot cosmology
- [ ] **Weak lensing**: shear catalogues or the Planck CMB lensing map, Ωₘ and σ₈
