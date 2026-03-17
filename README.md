# Spektroskopie Pipeline - Dokumentation

Vollständige Dokumentation aller Spektroskopie-Skripte für die Datenreduktion und Analyse von astronomischen Spektren.

**Erstellt:** 18. November 2025  
**Arbeitsverzeichnis-Struktur:**
```
work_dir/
├── in/           # Eingabedateien (FITS)
└── out/          # Ausgabedateien (FITS, PNG, TXT)
```

---

## Inhaltsverzeichnis

1. [spec_stack.py](#1-spec_stackpy) - FITS-Stacking
2. [spec_extsci.py](#2-spec_extscipy) - Spektrenextraktion aus 2D-FITS
3. [spec_calsci.py](#3-spec_calscipy) - Wellenlängenkalibrierung
4. [spec_flux.py](#4-spec_fluxpy) - Flusskalibration
5. [spec_plot.py](#5-spec_plotpy) - Spektrenplot mit Linien-Detektion
6. [spec_plot_fluxcal.py](#6-spec_plot_fluxcalpy) - Flusskalibriertes Spektrum plotten
7. [Typischer Workflow](#typischer-workflow)
8. [Dateiübersicht](#dateiübersicht)

---

## 1. spec_stack.py

**Zweck:** Stacking mehrerer 2D-FITS-Dateien vor der weiteren Verarbeitung

### Funktionalität
- Sucht alle FITS mit gleichem Prefix in `in/`
- Kombiniert die Daten standardmaessig per Median
- Optional: arithmetisches Mittel mit `--arith`
- Aktualisiert Header (`NCOMBINE`, Kommentare)
- Speichert das gestackte Ergebnis direkt als `science_spectrum.fits`
- Dieses FITS wird anschließend von `spec_extsci.py` weiterverarbeitet

### Verwendung
```bash
python3 spec_stack.py ARBEITSORDNER DATEI_PREFIX [--name OUTPUT_NAME] [--arith]
```

### Beispiel
```bash
# Stackt alle science_spectrum_*.fits Dateien (Standard: Median)
python3 spec_stack.py /pfad/zu/work science_spectrum_

# Ergebnis: /pfad/zu/work/in/science_spectrum.fits

# Arithmetisches Mittel statt Median
python3 spec_stack.py /pfad/zu/work science_spectrum_ --arith

# Eigener Ausgabename (z.B. fuer Kalibrationslampe)
python3 spec_stack.py /pfad/zu/work calibration_lamp_ --name calibration_lamp
# Ergebnis: /pfad/zu/work/in/calibration_lamp.fits
```

### Eingabe
- Alle `ARBEITSORDNER/in/DATEI_PREFIX*.fits` Dateien

### Ausgabe
- `ARBEITSORDNER/in/science_spectrum.fits` - Gestacktes 2D-FITS fuer die Extraktion
- Header-Eintrag: `NCOMBINE = N` (Anzahl kombinierter Dateien)
- Header-Eintrag: `COMBMETH = MEDIAN|MEAN` (Kombinationsmethode)

### Optionen
- `--name OUTPUT_NAME` - Ausgabename ohne oder mit `.fits` (Standard: `science_spectrum`)
- `--arith` - Verwendet arithmetisches Mittel statt Median

### Anforderungen
- Alle Dateien muessen identische Dimensionen haben
- Fehlerhafte oder abweichende Dateien werden uebersprungen

### Hinweis
- Wenn nur eine einzelne Rohdatei vorliegt, kann dieser Schritt uebersprungen werden, sofern die Datei bereits als `ARBEITSORDNER/in/science_spectrum.fits` vorliegt

---

## 2. spec_extsci.py

**Zweck:** Interaktive Extraktion eines 1D-Spektrums aus einem 2D-FITS-Bild

### Funktionalität
- Lädt 2D-FITS (`science_spectrum.fits`) aus `in/`
- Typischerweise stammt diese Datei aus dem vorherigen Stacking mit `spec_stack.py`
- Spiegelt Bild horizontal (X-Achse)
- Interaktive GUI zur Definition von:
  - Spektrums-Trace (Position und Breite)
  - Himmelshintergrund-Regionen (Sky offset und Breite)
- Berechnet SNR-Profil und Trace
- Subtrahiert Himmelshintergrund
- Speichert 1D-Spektrum

### Verwendung
```bash
python3 spec_extsci.py ARBEITSORDNER
```

### Eingabe
- `ARBEITSORDNER/in/science_spectrum.fits` - 2D-FITS mit Spektrum, idealerweise bereits gestackt

### Ausgabe
- `ARBEITSORDNER/out/science_spectrum_1d.fits` - Extrahiertes 1D-Spektrum (Binär-Tabelle)
  - Spalten: `PIXEL`, `FLUX`, `SKY`

### Parameter (im Skript anpassbar)
- `spektrum_half_width_init = 10` - Halbe Breite des Spektrums (Pixel)
- `sky_offset_init = 40` - Abstand der Sky-Region vom Spektrum (Pixel)
- `sky_width = 10` - Breite der Sky-Region (Pixel)
- `snr_threshold = 5.0` - SNR-Schwelle für Trace-Berechnung

### Interaktive Bedienung
- **Slider:** Anpassung von Spektrums-Breite, Sky-Position, Sky-Breite
- **Plot:** Zeigt 2D-Bild mit markierten Regionen und extrahiertes 1D-Spektrum
- **Automatisches Speichern** beim Schließen des Fensters

---

## 3. spec_calsci.py

**Zweck:** Interaktive Wellenlängenkalibrierung eines 1D-Spektrums

### Funktionalität
- Lädt 1D-Spektrum (`science_spectrum_1d.fits`)
- Optional: Lädt Kalibrations-Lampen-Spektrum (2D-FITS)
- Interaktive GUI zum Markieren von Spektrallinien
- Polynomial-Fit (Pixel → Wellenlänge)
- Speichert Wellenlösungs-Koeffizienten
- Wendet Wellenlösung an und speichert kalibriertes Spektrum

### Verwendung
```bash
python3 spec_calsci.py ARBEITSORDNER
```

**Optionale Flags:**
- `--solution <file>` oder `--apply <file>` - Wendet eine gespeicherte Wellenlösung direkt an (keine interaktive Kalibrierung)
  ```bash
  python3 spec_calsci.py ARBEITSORDNER --solution out/wavelength_solution.txt
  ```

### Eingabe
- `ARBEITSORDNER/out/science_spectrum_1d.fits` - 1D-Spektrum (aus spec_extsci.py)
- Optional: `ARBEITSORDNER/in/calibration_lamp.fits` - Kalibrations-Lampe (2D-FITS)

### Ausgabe
- `ARBEITSORDNER/out/wavelength_solution.txt` - Polynom-Koeffizienten
- `ARBEITSORDNER/out/science_spectrum_calibrated.fits` - Wellenlängen-kalibriertes Spektrum
  - Spalten: `WAVELENGTH`, `FLUX`

### Interaktive Bedienung
1. **Linien markieren:** Klick auf Peak im Plot
2. **Wellenlänge eingeben:** Textfeld (Angström)
3. **"Add Line"** - Punkt zur Liste hinzufügen
4. **Polynomial-Grad wählen:** Textfeld (Standard: 3)
5. **"Fit Polynomial"** - Berechnet Wellenlösung
6. **"Apply & Save"** - Speichert kalibriertes Spektrum

### Wellenlösungs-Format (`wavelength_solution.txt`)
```
# Wellenlöse-Polynom
deg 3
coeffs c3 c2 c1 c0
```
Wellenlänge(Pixel) = c3·Pixel³ + c2·Pixel² + c1·Pixel + c0

---

## 4. spec_flux.py

**Zweck:** Absolute Flusskalibration mit Referenz-Spektren

### Funktionalität
- Lädt wellenlängen-kalibriertes Spektrum
- Lädt Referenz-Spektrum (Pickles Atlas oder synthetisch)
- Berechnet Instrumental Response: R(λ) = Flux_obs / Flux_ref
- Kalibriert auf absolute Flusseinheiten [erg/s/cm²/Å]
- Erstellt 4-Panel-Plot (beobachtet, Referenz, Response, kalibriert)

### Verwendung
```bash
# Mit Pickles Atlas (Standard)
python3 spec_flux.py ARBEITSORDNER SPEKTRALTYP

# Mit synthetischen Spektren
python3 spec_flux.py ARBEITSORDNER SPEKTRALTYP --no-pickles

# Nur Plot speichern (kein Fenster)
python3 spec_flux.py ARBEITSORDNER SPEKTRALTYP --no-plot
```

### Beispiele
```bash
python3 spec_flux.py vega A0V          # Vega (A0V Stern)
python3 spec_flux.py sun G2V           # Sonne (G2V Stern)
python3 spec_flux.py altair A5V        # Altair
```

### Eingabe
- `ARBEITSORDNER/out/science_spectrum_calibrated.fits` - Wellenlängen-kalibriert

### Ausgabe
- `ARBEITSORDNER/out/science_spectrum_flux_calibrated.fits` - Flusskalibriert
  - Spalten: `WAVELENGTH`, `FLUX`, `FLUX_OBSERVED`
  - Header: `SPECTYPE`, `FLUXCAL`, `METHOD`
- `ARBEITSORDNER/out/science_spectrum_flux_calibrated.png` - 4-Panel-Plot
- `ARBEITSORDNER/out/flux_calibration.fits` - Wiederverwendbare Kalibration
  - Spalten: `WAVELENGTH`, `CALIB_FACTOR`, `REFERENCE_FLUX`
- `ARBEITSORDNER/out/instrumental_response.fits` - Response-Kurve (nur Pickles)
  - Spalten: `WAVELENGTH`, `RESPONSE`

### Verfügbare Spektraltypen (Pickles Atlas)
**Hauptreihe (V):**
- O: O5V, O9V
- B: B0V, B1V, B3V, B5V, B8V
- A: A0V, A2V, A3V, A5V
- F: F0V, F2V, F5V, F8V
- G: G0V, G2V, G5V, G8V
- K: K0V, K2V, K5V, K7V
- M: M0V, M2V, M4V, M5V

**Riesen (III):**
- G5III, G8III, K0III, K3III, K5III, M0III, M5III

### Pickles Atlas
- **Quelle:** STScI HST Reference Atlases
- **Wellenlängenbereich:** 1150-25000 Å (UV bis nah-IR)
- **Auflösung:** R ≈ 10-50 (~500 Å)
- **Flux-Kalibrierung:** Normiert auf Vega = 0 mag
- **Automatischer Download:** Von `archive.stsci.edu` in `pickles_cache/`
- **Referenz:** Pickles, A. J. (1998), PASP, 110, 863

### Wiederverwendung der Kalibration
```python
from spec_flux import apply_flux_calibration

# Lade dein Spektrum
wavelength, flux_counts = load_spectrum("new_observation.fits")

# Wende gespeicherte Kalibration an
flux_calibrated = apply_flux_calibration(
    wavelength, 
    flux_counts, 
    "vega/out/flux_calibration.fits"
)
```

---

## 5. spec_plot.py

**Zweck:** Plotten eines wellenlängen-kalibrierten Spektrums mit automatischer Linien-Detektion

### Funktionalität
- Lädt wellenlängen-kalibriertes Spektrum
- Automatische Detektion von Absorptions-/Emissionslinien
- Identifizierung bekannter Linien (100+ Datenbank)
- Erstellt 2-Panel-Plot (original + normalisiert)
- Annotiert identifizierte Linien

### Verwendung
```bash
python3 spec_plot.py ARBEITSORDNER
```

### Eingabe
- `ARBEITSORDNER/out/science_spectrum_calibrated.fits`

### Ausgabe
- `ARBEITSORDNER/out/science_spectrum_with_lines_improved.png`
- Konsolen-Ausgabe: Liste aller gefundenen Linien

### Linien-Datenbank (Auszug)
**Wasserstoff Balmer-Serie:**
- Hα (6562.79 Å), Hβ (4861.33 Å), Hγ (4340.47 Å), Hδ (4101.74 Å), Hε (3970.07 Å)

**Metalle:**
- Ca II H&K (3968.47, 3933.66 Å)
- Na I D-Dublette (5889.95, 5895.92 Å)
- Mg I b-Triplett (5167.32, 5172.68, 5183.60 Å)
- Fe I/II (verschiedene Linien)

**Verbotene Linien (Nebel):**
- [O III] (4958.91, 5006.84 Å)
- [O II] (3726.03, 3728.82 Å)
- [N II] (6548.05, 6583.45 Å)
- [S II] (6716.44, 6730.82 Å)

**Tellurische Linien:**
- O2 A-Band (7594 Å), B-Band (6867 Å), γ-Band (6287 Å)
- H2O (verschiedene Banden)

### Detektions-Parameter
- **Kontinuums-Fit:** Savitzky-Golay Filter (window=501, polyorder=3)
- **Minimale Prominenz:** 0.03
- **Minimaler Abstand:** 15 Pixel
- **SNR-Schwelle:** 3.0
- **Matching-Toleranz:** 8 Å

---

## 6. spec_plot_fluxcal.py

**Zweck:** Flexibles Plotten von flusskalibrierten Spektren mit optionaler Linien-Detektion

### Funktionalität
- Lädt flusskalibriertes Spektrum
- Optionale Spektrallinien-Detektion und -Annotation
- Automatisches PNG-Speichern
- Interaktive Plot-Anzeige
- Anpassbare Achsen, Glättung, Titel

### Verwendung
```bash
# Einfach (mit Auto-Save und Fenster)
python3 spec_plot_fluxcal.py ARBEITSORDNER

# Mit Spektrallinien-Annotation
python3 spec_plot_fluxcal.py ARBEITSORDNER --lines

# Direkt FITS-Datei angeben
python3 spec_plot_fluxcal.py /pfad/zu/science_spectrum_flux_calibrated.fits

# Mit Glättung und Achsengrenzen
python3 spec_plot_fluxcal.py ARBEITSORDNER --smooth 2.0 --xlim 3500 9000 --ylim 0 1e-14

# Nur speichern, kein Fenster
python3 spec_plot_fluxcal.py ARBEITSORDNER --no-show

# Eigener PNG-Pfad
python3 spec_plot_fluxcal.py ARBEITSORDNER --save mein_plot.png

# Kein Auto-Save
python3 spec_plot_fluxcal.py ARBEITSORDNER --no-save
```

### Eingabe
- `ARBEITSORDNER/out/science_spectrum_flux_calibrated.fits` (Standard)
- Oder: Direkter FITS-Pfad via Argument oder `--file`

### Ausgabe
- **Standard:** `ARBEITSORDNER/out/science_spectrum_flux_calibrated_plot.png`
- **Custom:** Via `--save PATH`
- **Fenster:** Interaktives Matplotlib-Fenster (falls `--show` oder Standard)

### Optionen
| Option | Beschreibung |
|--------|-------------|
| `--file PATH` | Expliziter FITS-Pfad |
| `--save PATH` | PNG-Speicherpfad (überschreibt Standard) |
| `--no-save` | Deaktiviert Auto-Save |
| `--show` | Erzwingt Fenster-Anzeige |
| `--no-show` | Kein Fenster (Batch-Modus) |
| `--lines` | **Aktiviert Spektrallinien-Detektion und -Annotation** |
| `--smooth SIGMA` | Gauss-Glättung (Pixel, float) |
| `--xlim XMIN XMAX` | X-Achsen-Bereich [Å] |
| `--ylim YMIN YMAX` | Y-Achsen-Bereich [erg/s/cm²/Å] |
| `--title TEXT` | Eigener Titel |

### Spektrallinien-Annotation (--lines)
- **Detektion:** Wie in spec_plot.py (Savitzky-Golay, SNR > 3)
- **Darstellung:** 
  - Rote gepunktete vertikale Linien
  - Vertikal geschriebene Labels bei 90% Plot-Höhe
  - Keine Kontinuums-Linie (nur flusskalibriertes Spektrum)
- **Toleranz:** 8 Å für Line-Matching
- **Datenbank:** 100+ bekannte Linien (H, He, Metalle, verbotene Linien, tellurisch)

### Plot-Features
- **Titel:** `<work_dir> — Flux-Calibrated Spectrum (SPECTYPE) — Pickles Atlas`
- **X-Achse:** Wavelength [Å]
- **Y-Achse:** Flux [erg/s/cm²/Å]
- **Legende:** Zeigt "Flux (calibrated)"
- **Grid:** Transparent, Alpha 0.3

---

## Typischer Workflow

### Vollständige Spektren-Reduktion und -Analyse

```bash
# 1. FITS-Dateien stacken (mehrere Rohaufnahmen -> ein Arbeits-FITS, Standard: Median)
python3 spec_stack.py vega science_spectrum_
# -> Input: vega/in/science_spectrum_*.fits
# -> Output: vega/in/science_spectrum.fits

# Optional: Arithmetisches Mittel
python3 spec_stack.py vega science_spectrum_ --arith

# 2. Spektrum extrahieren (2D -> 1D)
python3 spec_extsci.py vega
# -> Interaktiv: Trace und Sky anpassen
# -> Output: vega/out/science_spectrum_1d.fits

# 3. Wellenlaengen-Kalibrierung
python3 spec_calsci.py vega
# -> Interaktiv: Linien markieren, Wellenlaengen eingeben
# -> Output: vega/out/science_spectrum_calibrated.fits
#            vega/out/wavelength_solution.txt

# 4. Flux-Kalibrierung (z.B. fuer A0V Stern)
python3 spec_flux.py vega A0V
# -> Laedt Pickles A0V Referenzspektrum
# -> Output: vega/out/science_spectrum_flux_calibrated.fits
#            vega/out/science_spectrum_flux_calibrated.png
#            vega/out/instrumental_response.fits
#            vega/out/flux_calibration.fits

# 5. Plot mit Spektrallinien
python3 spec_plot_fluxcal.py vega --lines
# -> Output: vega/out/science_spectrum_flux_calibrated_plot.png
# -> Zeigt Fenster mit annotiertem Spektrum

# 6. Alternative: Wellenlaengen-kalibriertes Spektrum mit Linien
python3 spec_plot.py vega
# -> Output: vega/out/science_spectrum_with_lines_improved.png
```

Wenn bereits genau eine fertige Rohdatei `ARBEITSORDNER/in/science_spectrum.fits` existiert, kann Schritt 1 entfallen.

### Batch-Verarbeitung (ohne Fenster)

```bash
# Mehrere Spektren automatisch prozessieren
for star in vega altair deneb; do
    echo "Processing $star..."
    python3 spec_flux.py $star A0V --no-plot
    python3 spec_plot_fluxcal.py $star --lines --no-show
done
```

### Stacking mehrerer Beobachtungen

```bash
# Mehrere Science-Spektren kombinieren
cp raw_spectrum_*.fits vega/in/
python3 spec_stack.py vega raw_spectrum_
# -> Output: vega/in/science_spectrum.fits

# Danach mit dem gestackten Arbeits-FITS weiter prozessieren
python3 spec_extsci.py vega
# etc.
```

---

## Dateiübersicht

### Verzeichnisstruktur nach vollständiger Pipeline

```
work_dir/
├── in/
│   ├── science_spectrum.fits          # 2D-FITS nach dem Stacken oder einzelne Rohdatei
│   ├── calibration_lamp.fits          # 2D-FITS (optional, Kalibration)
│   └── science_spectrum_*.fits        # Mehrere FITS zum Stacken (optional)
│
└── out/
    ├── science_spectrum_1d.fits                    # 1D extrahiert
    ├── wavelength_solution.txt                     # Wellenlösungs-Koeffizienten
    ├── science_spectrum_calibrated.fits            # Wellenlängen-kalibriert
    ├── science_spectrum_flux_calibrated.fits       # Flux-kalibriert
    ├── science_spectrum_flux_calibrated.png        # Plot (4-Panel, von spec_flux.py)
    ├── science_spectrum_flux_calibrated_plot.png   # Plot (1-Panel, von spec_plot_fluxcal.py)
    ├── science_spectrum_with_lines_improved.png    # Plot mit Linien (spec_plot.py)
    ├── flux_calibration.fits                       # Wiederverwendbare Kalibration
    └── instrumental_response.fits                  # Response-Kurve (nur Pickles)
```

### FITS-Format Details

#### science_spectrum_1d.fits (spec_extsci.py)
```
Extension 0: PrimaryHDU (leer)
Extension 1: BinTableHDU
  Spalten:
    - PIXEL (int): Pixel-Index
    - FLUX (float): Extrahierter Flux [counts]
    - SKY (float): Himmelshintergrund [counts]
```

#### science_spectrum_calibrated.fits (spec_calsci.py)
```
Extension 0: PrimaryHDU (leer)
Extension 1: BinTableHDU
  Spalten:
    - WAVELENGTH (float): Wellenlänge [Å]
    - FLUX (float): Flux [counts]
```

#### science_spectrum_flux_calibrated.fits (spec_flux.py)
```
Extension 0: PrimaryHDU
  Header:
    - SPECTYPE: Spektraltyp (z.B. 'A0V')
    - FLUXCAL: True
    - METHOD: 'PICKLES' oder 'SYNTHETIC'
    - BUNIT: 'erg/s/cm^2/Angstrom'

Extension 1: BinTableHDU
  Spalten:
    - WAVELENGTH (float): Wellenlänge [Å]
    - FLUX (float): Kalibrierter Flux [erg/s/cm²/Å]
    - FLUX_OBSERVED (float): Original Flux [counts]
```

#### flux_calibration.fits (spec_flux.py)
```
Extension 0: PrimaryHDU
  Header:
    - SPECTYPE: Referenz-Spektraltyp
    - PURPOSE: 'Flux calibration'
    - CAL_MED, CAL_MEAN, CAL_STD: Statistiken

Extension 1: BinTableHDU
  Spalten:
    - WAVELENGTH (float): Wellenlänge [Å]
    - CALIB_FACTOR (float): Kalibrationsfaktor [(erg/s/cm²/Å)/counts]
    - REFERENCE_FLUX (float): Referenz-Flux [normalisiert]
```

#### instrumental_response.fits (spec_flux.py, nur Pickles)
```
Extension 0: PrimaryHDU
  Header:
    - SPECTYPE: Referenz-Spektraltyp
    - PURPOSE: 'Instrumental Response'
    - METHOD: 'PICKLES'
    - RESP_MED, RESP_MEAN: Statistiken

Extension 1: BinTableHDU
  Spalten:
    - WAVELENGTH (float): Wellenlänge [Å]
    - RESPONSE (float): Response [counts/(erg/s/cm²/Å)]
```

---

## Abhängigkeiten

### Python-Pakete
```bash
pip install numpy scipy matplotlib astropy
```

**Erforderlich:**
- `numpy` - Numerische Berechnungen
- `scipy` - Signal-Verarbeitung (Filtering, Interpolation)
- `matplotlib` - Plotting und interaktive GUIs
- `astropy` - FITS I/O

**Optional:**
- `requests` - Automatischer Download von Pickles-Spektren (spec_flux.py)

### Python-Version
- **Mindestens:** Python 3.9 (wegen Type Hints wie `tuple[str, str]`)
- **Empfohlen:** Python 3.10+

---

## Tipps & Tricks

### Wellenlängen-Kalibrierung verbessern
1. **Mehr Linien verwenden:** Mindestens 5-10 bekannte Linien für stabilen Fit
2. **Höherer Polynom-Grad:** Für nicht-lineare Dispersionen (z.B. Grad 3-4)
3. **Gleichmäßige Verteilung:** Linien über gesamten Wellenlängenbereich
4. **Kalibrations-Lampe verwenden:** He-Ne, Ar, Ne bieten viele scharfe Linien

### Flux-Kalibrierung optimieren
1. **Richtigen Spektraltyp wählen:** Möglichst genau zum Stern passen
2. **Pickles bevorzugen:** Breiter Wellenlängenbereich (1150-25000 Å)
3. **Response-Kurve überprüfen:** Sollte glatt sein ohne starke Oszillationen
4. **Telluric Correction:** Bei erdgebundenen Beobachtungen wichtig

### Spektrallinien besser erkennen
1. **Glättung anpassen:** `--smooth 1.0` bis `3.0` je nach SNR
2. **Kontinuums-Normalisierung:** Verbessert schwache Linien
3. **Toleranz erhöhen:** Bei schlechter Wellenlängen-Kalibrierung
4. **Custom Line-List:** Eigene Linien in LINE_LIST einfügen

### Performance
- **Große FITS:** Verwende `spec_stack.py` vor der Extraktion
- **Batch-Processing:** `--no-plot` und `--no-show` Flags nutzen
- **Pickles Cache:** Spektren werden in `pickles_cache/` gespeichert (nur 1x Download)

### Empfohlene Reihenfolge
1. Zuerst mehrere Roh-FITS mit `spec_stack.py` zu `in/science_spectrum.fits` stacken (standardmaessig Median)
2. Danach `spec_extsci.py` fuer die 1D-Extraktion ausfuehren
3. Anschliessend kalibrieren und plotten

---

## Fehlerbehebung

### "FITS nicht gefunden"
→ Prüfe Ordnerstruktur: `work_dir/in/` und `work_dir/out/` müssen existieren

### "Keine Wellenlösung gefunden"
→ Speichere zuerst mit "Apply & Save" in spec_calsci.py

### "Spektraltyp nicht in Pickles-Datenbank"
→ Verwende `--no-pickles` für synthetische Spektren oder wähle ähnlichen Typ

### "Pickles-Spektrum nicht verfügbar"
→ Automatischer Download fehlgeschlagen, siehe Konsole für manuelle Download-URL

### "Too many values to unpack"
→ FITS-Format nicht kompatibel, prüfe mit `astropy.io.fits.info(filename)`

### Linien werden nicht erkannt
→ SNR zu niedrig oder Kontinuums-Fit falsch, passe Parameter in Skript an

---

## Weiterführende Informationen

### Referenzen
- **Pickles Atlas:** Pickles, A. J. (1998), PASP, 110, 863
  - URL: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas
- **NIST Atomic Spectra Database:** https://physics.nist.gov/PhysRefData/ASD/lines_form.html
- **FITS Standard:** https://fits.gsfc.nasa.gov/

### Kontakt & Support
Bei Fragen oder Problemen:
1. Prüfe Konsolen-Ausgabe auf Fehlermeldungen
2. Validiere FITS-Dateien mit `astropy.io.fits.info()`
3. Überprüfe Ordnerstruktur und Dateinamen

---

**Ende der Dokumentation**
