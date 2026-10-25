# Hubble Frontier Fields: mass models compared

Generated automatically by `analysis/frontier.py`. Cosmology: H₀ = 70, Ωₘ = 0.3 (as for the HFF models).

| Cluster | Team | θ_E,eff (z_s = 2) | M(<100 kpc) | M(<R_max) | A(μ>10), z_s = 9 | deflection map vs. FFT(κ) | Pixel |
|---|---|---|---|---|---|---|---|
| Abell 2744 | CATS | 21.4″ | 6.02e+13 M☉ | 2.66e+14 M☉ (R = 281 kpc) | 0.030 arcmin² (r < 62″) | 13.2 % | 0.300″ |
| Abell 2744 | GLAFIC | 21.8″ | 6.02e+13 M☉ | 2.62e+14 M☉ (R = 281 kpc) | 0.032 arcmin² (r < 62″) | 13.1 % | 0.030″ |
| MACS J0416.1−2403 | CATS | 24.7″ | 5.39e+13 M☉ | 3.75e+14 M☉ (R = 375 kpc) | 0.029 arcmin² (r < 70″) | 3.5 % | 0.300″ |
| MACS J0416.1−2403 | GLAFIC | 25.1″ | 5.24e+13 M☉ | 3.34e+14 M☉ (R = 375 kpc) | 0.024 arcmin² (r < 70″) | 16.5 % | 0.030″ |

Both teams are evaluated around the same sky centre (mean of the central κ maxima) and only inside the circle that all maps cover completely (the GLAFIC maps are considerably smaller than the CATS ones).

The column "deflection map vs. FFT(κ)" is the median relative deviation of the team's deflection angles from our FFT computation from the κ map in the central region. The directions agree everywhere; the team maps have slightly larger amplitudes because they also include mass outside the map field, which the FFT misses.

![Maps](../figures/frontier/frontier_maps.svg)

![Mass profiles](../figures/frontier/frontier_mass.svg)

![Area](../figures/frontier/frontier_area.svg)
