# lenscape

Gravitationslinsen modellieren, interaktiv erkunden und mit echten Beobachtungsdaten auswerten.

Die Idee: Mit derselben Physik, die einen Einsteinring im Simulator erzeugt, sollen sich echte Linsen
vermessen lassen. Daraus folgen Massen von Galaxien und Haufen, der Anteil Dunkler Materie und, über die
Distanzen, der Einfluss der Kosmologie (Ωₘ, Dunkle Energie w, H₀).

## Physik in einem Satz

Ein Lichtstrahl, der die Linse unter dem Winkel **θ** passiert, kommt von der Quellposition

```
β = θ − α(θ),        θ_E² = 4GM/c² · D_ds / (D_d · D_s)
```

Die Kosmologie geht nur über die Winkeldurchmesser-Distanzen D_d, D_s und D_ds ein, also über die kritische
Flächendichte Σ_crit = c²/(4πG) · D_s/(D_d D_ds).

## Inhalt

| Modul | Inhalt |
|---|---|
| `lenscape.cosmology` | wCDM-Kosmologie, Distanzen, Σ_crit, Einsteinradien, M(<θ_E), Zeitverzögerungs-Distanz |
| `lenscape.lenses` | Punktmasse, SIS, SIE (elliptisch), NFW-Halo, externe Scherung, Überlagerungen |
| `lenscape.sources` | Gauß-, Sérsic- und Bildquellen |
| `lenscape.raytrace` | Ray-Tracing mit Supersampling, Konvergenz- und Vergrößerungskarten |
| `lenscape.caustics` | kritische Kurven und Kaustiken |
| `lenscape.solver` | Bildpositionen und Vergrößerungen einer Punktquelle |
| `app/` | interaktiver Simulator (Streamlit) |
| `web/` | Nebenprojekt für die Website: eigenes Foto hochladen und verbiegen |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[app,dev]"
```

## Simulator starten

```bash
streamlit run app/streamlit_app.py
```

Regler für Kosmologie (H₀, Ωₘ, Ω_Λ, w), Rotverschiebungen, Massenmodell und Quelle. Drei Ansichten:

- **Simulator**: Quellebene mit Kaustiken neben der beobachteten Bildebene mit kritischen Kurven; die Bilder der
  Quellmitte werden mit Vergrößerung und Parität aufgelistet.
- **Konvergenz & Vergrößerung**: κ- und |μ|-Karten.
- **Dunkle Energie & Kosmologie**: Wie stark hängt die aus einem beobachteten θ_E abgeleitete Masse von w und Ωₘ ab?

## Als Bibliothek

```python
from lenscape import LensGeometry, SIE, ExternalShear, make_cosmology
from lenscape.solver import find_images

geom = LensGeometry(z_d=0.3, z_s=1.0, cosmo=make_cosmology(Om0=0.31, w0=-1.0))
lens = SIE(geom.theta_E_sis(250), q=0.7) + ExternalShear(0.05, 0.0)

print(geom.mass_in_theta_E(geom.theta_E_sis(250)))   # M_sun innerhalb θ_E
print(find_images(lens, 0.05, 0.03))                 # 4 Bilder: x, y, μ
```

## Web: Foto-Linse

`web/index.html` ist eine einzelne Datei ohne Abhängigkeiten. Im Browser öffnen, Bild hochladen (oder per
Drag & Drop), Linse mit der Maus verschieben. Die Regler steuern Masse, Abstände, Bildbreite und die
**Lichtgeschwindigkeit** (θ_E ∝ 1/c). Szenarien: Galaxie, Sonne wie bei Eddington 1919, dieselbe Sonne bei
1 % von c, ein murmelgroßes Schwarzes Loch im Zimmer.

## Tests

```bash
pytest
```

Die Tests prüfen unter anderem die Poisson-Gleichung ∇·α = 2κ für jedes Modell, die Rotationsfreiheit von α,
analytische Bildpositionen und Vergrößerungen der SIS und M(<θ_E) = M für eine Punktmasse.

## Fahrplan

- [x] Simulator und Kosmologie (Punktmasse, SIS, SIE, NFW, Scherung, Kaustiken, Bildlöser)
- [x] Web-Demo „eigenes Bild verbiegen“
- [ ] **Starke Linsen aus echten Daten**: SLACS-Katalog laden, Massen und Dunkle-Materie-Anteil bestimmen
      (Linsenmasse vs. Sternmasse)
- [ ] **Galaxienhaufen**: Massenkarten der Hubble Frontier Fields einlesen, Mehrfachbilder vorhersagen
- [ ] **Zeitverzögerungen**: gelinste Quasare → H₀ (Vergleich mit H0LiCOW/TDCOSMO)
- [ ] **Modell-Fit**: SIE+Scherung an beobachtete Bildpositionen anpassen, Abgleich mit `lenstronomy`
- [ ] **Schwache Linsen**: Scherungskataloge bzw. Planck-CMB-Lensing-Karte, Ωₘ und σ₈
