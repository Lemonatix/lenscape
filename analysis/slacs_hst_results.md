# HST-Bildfit von SLACS-Linsen

Automatisch erzeugt von `analysis/slacs_hst_fit.py` (HST/ACS F814W, Gauß-PSF 0,1″).

| Linse | Modell | θ_E | q | PA | γ_ext | χ²_red |
|---|---|---|---|---|---|---|
| J0946+1006 | Bolton et al. 2008 (SIE) | 1.38″ | 0.81 | 159° | – | – |
| | lenscape, SIE | 1.391″ | 0.78 | 162° | – | 1.64 |
| | lenscape, SIE + Scherung | 1.406″ | 0.92 | 42° | 0.081 | 1.47 |

Der Einsteinradius ist robust; Elliptizität und externe Scherung sind teilweise entartet (beide erzeugen ein Quadrupol-Feld). Vergleichbar mit dem Katalog ist das reine SIE.
Kosmische Strahlung und Defektpixel sind maskiert (grau im Residuum).

![Fit](../figures/slacs_hst/hst_fit.svg)
