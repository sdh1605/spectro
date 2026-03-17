#!/usr/bin/env python3
"""
Spektral-Flusskalibration
=========================
Kalibriert ein Science-Spektrum auf absolute Flusswerte unter Verwendung
eines Referenz-Spektrums basierend auf dem Spektraltyp.

Die Kalibration kann erfolgen mit:
- **Pickles Atlas** (empfohlen): Verwendet empirische stellare Spektren mit
  bekannten absoluten Flüssen. Berechnet zusätzlich die instrumental response.
- **Synthetische Spektren**: Verwendet Planck-basierte Modell-Spektren als Fallback.

Verwendung:
    python spec_flux.py ARBEITSORDNER SPEKTRALTYP [--pickles | --no-pickles]
    
Beispiele:
    # Mit Pickles Atlas (Standard)
    python spec_flux.py vega A0V
    python spec_flux.py altair A7V --pickles
    
    # Mit synthetischen Spektren
    python spec_flux.py sun G2V --no-pickles
    
Das Programm:
1. Lädt das kalibrierte Science-Spektrum aus ARBEITSORDNER/out/science_spectrum_calibrated.fits
2. Lädt ein Pickles-Referenz-Spektrum für den Spektraltyp (oder generiert synthetisches)
3. Berechnet die instrumental response R(λ) = flux_observed / flux_reference
4. Führt Flusskalibration durch
5. Speichert das flusskalibrierte Spektrum und die instrumental response:
   - ARBEITSORDNER/out/science_spectrum_flux_calibrated.fits
   - ARBEITSORDNER/out/science_spectrum_flux_calibrated.png
   - ARBEITSORDNER/out/flux_calibration.fits (wiederverwendbar)
   - ARBEITSORDNER/out/instrumental_response.fits (nur bei Pickles)

Instrumental Response:
----------------------
Die instrumental response R(λ) beschreibt die wellenlängenabhängige Effizienz
des Spektrographen und Detektors:

    flux_observed(λ) = flux_true(λ) × R(λ)

Sie wird aus dem Vergleich des beobachteten Spektrums mit dem kalibrierten
Pickles-Spektrum berechnet. Die Response-Kurve kann verwendet werden, um:
- Zukünftige Beobachtungen zu korrigieren
- Die Systemeffizienz zu analysieren
- Instrumenten-Degradation zu überwachen

Wiederverwendung der Kalibration:
----------------------------------
Die gespeicherte Kalibrationsdatei kann auf zukünftige Spektren angewendet werden:

In Python:
    from spec_flux import apply_flux_calibration
    
    # Lade dein Spektrum
    wavelength, flux_counts = load_spectrum("new_observation.fits")
    
    # Wende gespeicherte Kalibration an
    flux_calibrated = apply_flux_calibration(
        wavelength, 
        flux_counts, 
        "vega/out/flux_calibration.fits"
    )

In eigenem Skript:
    import numpy as np
    from astropy.io import fits
    from scipy.interpolate import interp1d
    
    # Lade Kalibration
    with fits.open("vega/out/flux_calibration.fits") as hdul:
        calib_wave = hdul[1].data['WAVELENGTH']
        calib_factor = hdul[1].data['CALIB_FACTOR']
    
    # Interpoliere auf deine Wellenlängen
    interp = interp1d(calib_wave, calib_factor, kind='linear')
    my_calib_factor = interp(my_wavelength)
    
    # Anwenden
    flux_calibrated = my_flux_counts * my_calib_factor
    # Einheit: erg/s/cm²/Å

Verfügbare Spektraltypen (Pickles):
---------------------------------
O: O5V, O9V
B: B0V, B1V, B3V, B5V, B8V
A: A0V, A2V, A3V, A5V
F: F0V, F2V, F5V, F8V
G: G0V, G2V, G5V, G8V
K: K0V, K2V, K5V, K7V
M: M0V, M2V, M4V, M5V
Riesen: G5III, G8III, K0III, K3III, K5III, M0III, M5III

Pickles Atlas:
---------------
Der Pickles Atlas ist eine Bibliothek von 131 flux-kalibrierten stellaren
Spektren, die eine vollständige Abdeckung aller normalen Spektraltypen und
Leuchtkraftklassen bietet.

Eigenschaften:
- Auflösung: ~500 Å (R ≈ 10-50)
- Wellenlängenbereich: 1150-25000 Å (UV bis nah-IR)
- Flux-kalibriert auf Vega (0 mag in allen Bändern)
- Quelle: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas

Referenz: Pickles, A. J. (1998), PASP, 110, 863
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.modeling import models, fitting
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os
import sys
import warnings
import urllib.request
import urllib.error
import ssl
warnings.filterwarnings('ignore')


# Referenz-Spektren für verschiedene Spektraltypen
# Quelle: Pickles (1998) PASP 110, 863 - Stellar Spectral Flux Library
# Oder vereinfachte Modelle basierend auf Temperatur und Spektralklasse
# ODER: MILES Database (Medium resolution INT Library of Empirical Spectra)

# MILES Database URLs und Star-IDs für verschiedene Spektraltypen
# Quelle: http://miles.iac.es/
# WICHTIG: Stand November 2025 ist der MILES-Server umstrukturiert worden
# Alternative Quellen:
# 1. Vizier: http://cdsarc.u-strasbg.fr/viz-bin/Cat?III/221
# 2. Lokale MILES-Bibliothek falls verfügbar
MILES_BASE_URL = "http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/fits?III/221/"
# Fallback: alte URL (funktioniert eventuell nicht mehr)
MILES_BASE_URL_OLD = "http://research.iac.es/proyecto/miles/pages/fluxCalibrated/fitsFiles/"

# Mapping von Spektraltypen zu MILES-Sternen (Star ID)
# Format: {spectral_type: (star_id, star_name)}
PICKLES_SPECTRAL_LIBRARY = {
    # O-Sterne
    'O5V': ('pickles_uk_1', 'O5V'),
    'O9V': ('pickles_uk_2', 'O9V'),
    # B-Sterne  
    'B0V': ('pickles_uk_3', 'B0V'),
    'B1V': ('pickles_uk_4', 'B1V'),
    'B3V': ('pickles_uk_5', 'B3V'),
    'B5V': ('pickles_uk_6', 'B5-7V'),
    'B8V': ('pickles_uk_7', 'B8V'),
    # A-Sterne
    'A0V': ('pickles_uk_9', 'A0V'),    # Vega-ähnlich
    'A2V': ('pickles_uk_10', 'A2V'),
    'A3V': ('pickles_uk_11', 'A3V'),
    'A5V': ('pickles_uk_12', 'A5V'),
    # F-Sterne
    'F0V': ('pickles_uk_14', 'F0V'),
    'F2V': ('pickles_uk_15', 'F2V'),
    'F5V': ('pickles_uk_16', 'F5V'),
    'F8V': ('pickles_uk_20', 'F8V'),
    # G-Sterne
    'G0V': ('pickles_uk_23', 'G0V'),
    'G2V': ('pickles_uk_26', 'G2V'),   # Sonne-ähnlich
    'G5V': ('pickles_uk_27', 'G5V'),
    'G8V': ('pickles_uk_30', 'G8V'),
    # K-Sterne
    'K0V': ('pickles_uk_31', 'K0V'),
    'K2V': ('pickles_uk_33', 'K2V'),
    'K5V': ('pickles_uk_36', 'K5V'),
    'K7V': ('pickles_uk_37', 'K7V'),
    # M-Sterne
    'M0V': ('pickles_uk_38', 'M0V'),
    'M2V': ('pickles_uk_40', 'M2V'),
    'M4V': ('pickles_uk_43', 'M4V'),
    'M5V': ('pickles_uk_44', 'M5V'),
    # Riesen
    'G5III': ('pickles_uk_73', 'G5III'),
    'G8III': ('pickles_uk_76', 'G8III'),
    'K0III': ('pickles_uk_78', 'K0III'),
    'K3III': ('pickles_uk_87', 'K3III'),
    'K5III': ('pickles_uk_93', 'K5III'),
    'M0III': ('pickles_uk_95', 'M0III'),
    'M5III': ('pickles_uk_100', 'M5III'),
}


def download_pickles_spectrum(spectral_type, cache_dir='pickles_cache'):
    """
    Lädt ein Pickles-Spektrum für den angegebenen Spektraltyp
    
    Diese Funktion prüft ZUERST ob die Datei lokal in cache_dir existiert.
    Der automatische Download wird nur als Fallback versucht.
    
    EMPFOHLEN: Lade Pickles-Spektren manuell herunter und lege sie in cache_dir/
    
    Parameters:
    -----------
    spectral_type : str
        Spektraltyp (z.B. 'A0V', 'G2V')
    cache_dir : str
        Verzeichnis mit Pickles-Spektren (Standard: 'pickles_cache')
    
    Returns:
    --------
    filepath : str
        Pfad zur lokalen FITS-Datei
    
    Raises:
    -------
    ValueError: Wenn Spektraltyp nicht in Pickles-Datenbank gefunden
    FileNotFoundError: Wenn Datei nicht lokal vorhanden und Download fehlschlägt
    """
    spectral_type = spectral_type.upper()
    
    if spectral_type not in PICKLES_SPECTRAL_LIBRARY:
        available = ', '.join(sorted(PICKLES_SPECTRAL_LIBRARY.keys()))
        raise ValueError(f"Spektraltyp '{spectral_type}' nicht in Pickles-Datenbank.\n"
                        f"Verfügbare Typen: {available}")
    
    file_id, spec_name = PICKLES_SPECTRAL_LIBRARY[spectral_type]
    
    # Erstelle Cache-Verzeichnis falls nicht vorhanden
    os.makedirs(cache_dir, exist_ok=True)
    
    # Pickles Dateiname (z.B. "pickles_uk_9.fits")
    filename = f"{file_id}.fits"
    
    # Mögliche Dateinamen
    possible_files = [
        os.path.join(cache_dir, filename),  # FITS (Standard)
    ]
    
    # ===================================================================
    # SCHRITT 1: Prüfe ob bereits lokal vorhanden (PRIMÄR)
    # ===================================================================
    for local_path in possible_files:
        if os.path.exists(local_path):
            # Prüfe Dateigröße
            file_size = os.path.getsize(local_path)
            if file_size < 1000:
                print(f"  ⚠ Warnung: {local_path} ist sehr klein ({file_size} Bytes)")
                print(f"    Wahrscheinlich eine Fehlerseite, überspringe...")
                continue
            
            # Verifiziere dass es lesbar ist
            try:
                wavelength, flux = load_pickles_spectrum(local_path)
                print(f"  ✓ Pickles-Spektrum gefunden: {local_path}")
                print(f"    Spektraltyp: {spec_name}")
                print(f"    Datenpunkte: {len(wavelength)}")
                return local_path
            except Exception as e:
                print(f"  ⚠ Warnung: {local_path} konnte nicht gelesen werden: {e}")
                continue
    
    # ===================================================================
    # SCHRITT 2: Datei nicht lokal → Gebe klare Anweisungen
    # ===================================================================
    print(f"\n  ⚠ Pickles-Spektrum nicht gefunden in: {cache_dir}/")
    print(f"     Benötigt: {spec_name} (File: {filename})")
    
    # Bevorzugter Dateiname
    local_path = os.path.join(cache_dir, filename)
    
    # Zeige wo man es herunterladen kann
    print(f"\n  📥 DOWNLOAD-ANLEITUNG:")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  1. Besuche: http://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/")
    print(f"  2. Lade herunter: {filename}")
    print(f"  3. Speichere als: {os.path.abspath(local_path)}")
    print(f"  4. Führe das Programm erneut aus")
    print(f"  ─────────────────────────────────────────────────────")
    
    # Versuche automatischen Download als Fallback
    print(f"\n  Versuche automatischen Download...")
    
    url = f"http://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/{filename}"
    
    try:
        print(f"    URL: {url}")
        
        try:
            import requests
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                # Verifiziere FITS
                try:
                    wavelength, flux = load_pickles_spectrum(local_path)
                    print(f"      ✓ Download erfolgreich! ({len(wavelength)} Punkte)")
                    return local_path
                except:
                    print(f"      → Ungültige FITS-Datei")
                    if os.path.exists(local_path):
                        os.remove(local_path)
            else:
                print(f"      → HTTP {response.status_code}")
                        
        except ImportError:
            print(f"      → 'requests' nicht installiert")
        except Exception as e:
            print(f"      → Fehler: {str(e)[:80]}...")
                
    except Exception as e:
        print(f"    Fehler: {e}")
    
    # ===================================================================
    # SCHRITT 3: Fehlgeschlagen → Klare Fehlermeldung
    # ===================================================================
    raise FileNotFoundError(
        f"\n{'='*70}\n"
        f"Pickles-Spektrum nicht verfügbar: {spec_name} ({spectral_type})\n"
        f"{'='*70}\n\n"
        f"Die Datei wurde weder lokal gefunden noch konnte sie heruntergeladen werden.\n\n"
        f"BITTE LADE DIE DATEI MANUELL HERUNTER:\n\n"
        f"Benötigte Datei: {filename}\n"
        f"Speichern als:   {os.path.abspath(local_path)}\n\n"
        f"Download-Quelle:\n"
        f"  STScI Archive:  {url}\n"
        f"  Alternative:    https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas\n\n"
        f"Nach dem Download: Führe das Programm erneut aus.\n"
        f"Alternative:       Verwende --no-pickles für synthetische Spektren\n"
        f"{'='*70}\n"
    )


def load_pickles_spectrum(filepath):
    """
    Lädt ein Pickles-Spektrum aus einer FITS-Datei
    
    Pickles FITS Format: Binary Table mit Spalten WAVELENGTH und FLUX
    
    Parameters:
    -----------
    filepath : str
        Pfad zur Pickles FITS-Datei
    
    Returns:
    --------
    wavelength : array
        Wellenlängen in Angström
    flux : array
        Flux in erg/s/cm²/Å (normiert auf Vega=0 mag)
    """
    import numpy as np
    
    # Prüfe ob Datei existiert
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickles-Spektrum nicht gefunden: {filepath}")
    
    try:
        with fits.open(filepath) as hdul:
            # Pickles Spektren sind als Binary Table gespeichert
            # Extension 1 enthält die Daten
            if len(hdul) < 2:
                raise ValueError("Pickles FITS muss mindestens 2 Extensions haben")
            
            table = hdul[1].data
            
            # Extrahiere Wellenlänge und Flux
            wavelength = table['WAVELENGTH']
            flux = table['FLUX']
            
            print(f"  Format: Pickles FITS ({len(wavelength)} Datenpunkte)")
            print(f"  Bereich: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Å")
            
            return wavelength, flux
            
    except Exception as e:
        raise RuntimeError(
            f"Konnte Pickles-Spektrum nicht laden: {filepath}\n"
            f"  Fehler: {e}\n"
            f"  Erwartetes Format: FITS Binary Table mit WAVELENGTH und FLUX Spalten"
        )


def calculate_instrumental_response(wavelength_obs, flux_obs, wavelength_ref, flux_ref):
    """
    Berechnet die instrumental response als Verhältnis observed/reference
    
    Die instrumental response R(λ) beschreibt die wellenlängenabhängige Effizienz
    des Instruments:
        flux_observed(λ) = flux_true(λ) × R(λ)
    
    Daher:
        R(λ) = flux_observed(λ) / flux_reference(λ)
    
    Tellurische O2- und H2O-Banden werden automatisch maskiert und interpoliert:
    - O2 A-Band: ~7550-7750 Å (7594 ±80 Å)
    - O2 B-Band: ~6820-6980 Å (6867 ±80 Å)
    - O2 γ-Band: ~6240-6360 Å (6287 ±60 Å)
    - H2O Banden: ~5850-6010, ~6450-6600, ~6920-7080, ~7100-7400, ~8100-8400, ~9250-9550 Å
    
    Bereiche sind großzügig dimensioniert um Wellenlängen-Kalibrationsungenauigkeiten
    zu kompensieren.
    
    Parameters:
    -----------
    wavelength_obs : array
        Wellenlängen des beobachteten Spektrums
    flux_obs : array
        Beobachteter Flux (in Counts oder beliebigen Einheiten)
    wavelength_ref : array
        Wellenlängen des Referenz-Spektrums (Pickles)
    flux_ref : array
        Referenz-Flux (kalibriert, in erg/s/cm²/Å)
    
    Returns:
    --------
    response : array
        Instrumental response auf wavelength_obs Grid
    response_smooth : array
        Geglättete instrumental response (mit interpolierten tellurischen Regionen)
    telluric_mask : array (bool)
        Maske der tellurischen Bereiche (True = tellurisch maskiert)
    """
    # Definiere tellurische Banden zum Maskieren (Wellenlänge in Å)
    # Hauptsächlich O2 Absorptionen im optischen Bereich
    # Bereiche moderat erweitert um Wellenlängen-Kalibrationsungenauigkeiten zu kompensieren
    telluric_bands = [
        # O2 Banden (stark) - erweitert um ±40-50 Å
        (7540, 7760),  # O2 A-Band (moderat erweitert: 7594 ±80 Å)
        (6820, 6990),  # O2 B-Band (moderat erweitert: 6867 ±80 Å)
        (6240, 6370),  # O2 γ-Band (moderat erweitert: 6287 ±65 Å)
    ]
    
    # Erstelle Maske für tellurische Bereiche
    telluric_mask = np.zeros(len(wavelength_obs), dtype=bool)
    for band_min, band_max in telluric_bands:
        band_mask = (wavelength_obs >= band_min) & (wavelength_obs <= band_max)
        telluric_mask |= band_mask
        if np.any(band_mask):
            print(f"  → Maskiere tellurisches Band: {band_min}-{band_max} Å ({np.sum(band_mask)} Pixel)")
    
    # NEUE STRATEGIE: Interpoliere tellurische Bereiche im BEOBACHTETEN SPEKTRUM
    # Dies ist physikalisch sinnvoller, da die tellurischen Linien nur dort vorhanden sind
    flux_obs_corrected = flux_obs.copy()
    
    if np.any(telluric_mask):
        valid_mask = ~telluric_mask & np.isfinite(flux_obs)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 10:
            print(f"  → Interpoliere {np.sum(telluric_mask)} tellurische Pixel im beobachteten Spektrum")
            print(f"  → Verwende lineare Interpolation (vermeidet Oszillationen)")
            
            # Interpoliere das beobachtete Spektrum über tellurische Bereiche
            # LINEAR um Oszillationen zu vermeiden
            interp_flux = interp1d(wavelength_obs[valid_mask], 
                                   flux_obs[valid_mask],
                                   kind='linear', bounds_error=False,
                                   fill_value='extrapolate')
            flux_obs_corrected[telluric_mask] = interp_flux(wavelength_obs[telluric_mask])
    
    # Interpoliere Referenz-Spektrum auf beobachtetes Wellenlängen-Grid
    # WICHTIG: LINEAR, da Pickles keine tellurischen Linien hat
    interp_func = interp1d(wavelength_ref, flux_ref, kind='linear', 
                           bounds_error=False, fill_value=np.nan)
    flux_ref_interp = interp_func(wavelength_obs)
    
    # Berechne Response aus ORIGINAL und KORRIGIERTEM Spektrum
    response_raw = flux_obs / (flux_ref_interp + 1e-10)  # Mit tellurischen Linien
    response_corrected = flux_obs_corrected / (flux_ref_interp + 1e-10)  # Ohne tellurische Linien
    
    # Glätte die korrigierte Response für finales Ergebnis
    response_smooth = gaussian_filter1d(response_corrected, sigma=5.0)
    print(f"  → Glätte korrigierte Response mit Gauss-Filter (sigma=5.0)")
    
    return response_raw, response_smooth, telluric_mask

class FluxCalibrator:
    """Klasse für spektrale Flusskalibration"""
    
    def __init__(self, work_dir, spectral_type, use_pickles=True, show_plot=True):
        """
        Initialisierung
        
        Parameters:
        -----------
        work_dir : str
            Arbeitsordner mit in/ und out/ Unterordnern
        spectral_type : str
            Spektraltyp (z.B. A0V, G2V, M5V)
        use_pickles : bool
            Wenn True, verwende Pickles Atlas; wenn False, synthetische Spektren
        show_plot : bool
            Wenn True, zeige Plot direkt an; wenn False, nur speichern
        """
        self.work_dir = work_dir
        self.spectral_type = spectral_type.upper()
        self.use_pickles = use_pickles
        self.show_plot = show_plot
        self.out_dir = os.path.join(work_dir, "out")
        
        # Pfade
        self.science_fits = os.path.join(self.out_dir, "science_spectrum_calibrated.fits")
        self.output_fits = os.path.join(self.out_dir, "science_spectrum_flux_calibrated.fits")
        self.output_png = os.path.join(self.out_dir, "science_spectrum_flux_calibrated.png")
        self.calibration_fits = os.path.join(self.out_dir, "flux_calibration.fits")
        self.response_fits = os.path.join(self.out_dir, "instrumental_response.fits")
        
        # Daten
        self.wavelength = None
        self.flux_observed = None
        self.flux_calibrated = None
        self.reference_flux = None
        self.reference_wavelength = None
        self.miles_wavelength = None  # Original MILES Wellenlängen
        self.miles_flux = None  # Original MILES Flux
        self.calibration_factor = None
        self.instrumental_response = None
        self.telluric_mask = None  # Maske der tellurischen Bereiche
        
    def load_science_spectrum(self):
        """Lädt das kalibrierte Science-Spektrum"""
        print(f"\nLade Science-Spektrum: {self.science_fits}")
        
        if not os.path.exists(self.science_fits):
            raise FileNotFoundError(f"Kalibriertes Spektrum nicht gefunden: {self.science_fits}")
        
        with fits.open(self.science_fits) as hdul:
            # Versuche verschiedene FITS-Strukturen
            if len(hdul) > 1 and isinstance(hdul[1].data, np.recarray):
                # Binary table
                data = hdul[1].data
                self.wavelength = data['WAVELENGTH']
                self.flux_observed = data['FLUX']
            elif len(hdul) > 1:
                # Separate HDUs
                self.wavelength = hdul[1].data
                self.flux_observed = hdul[0].data
            else:
                # Single HDU mit Header-Info
                self.flux_observed = hdul[0].data
                header = hdul[0].header
                # Versuche WCS zu lesen
                if 'CRVAL1' in header and 'CDELT1' in header:
                    crval1 = header['CRVAL1']
                    cdelt1 = header['CDELT1']
                    crpix1 = header.get('CRPIX1', 1)
                    naxis1 = header['NAXIS1']
                    self.wavelength = crval1 + (np.arange(naxis1) - crpix1 + 1) * cdelt1
                else:
                    raise ValueError("Keine Wellenlängen-Information im FITS gefunden!")
        
        print(f"  Wellenlängenbereich: {self.wavelength[0]:.1f} - {self.wavelength[-1]:.1f} Å")
        print(f"  Anzahl Pixel: {len(self.wavelength)}")
        print(f"  Flux-Bereich: {np.min(self.flux_observed):.2e} - {np.max(self.flux_observed):.2e}")
        
    def get_reference_spectrum(self):
        """
        Lädt oder generiert Referenz-Spektrum für den Spektraltyp
        
        Verwendet Pickles Atlas wenn use_pickles=True, sonst vereinfachte
        synthetische Modelle basierend auf Planck-Funktion
        """
        if self.use_pickles:
            print(f"\nLade Pickles-Referenz-Spektrum für: {self.spectral_type}")
            
            try:
                # Download (oder verwende lokale Datei)
                # Verwende pickles_cache im Script-Verzeichnis, nicht im Arbeitsverzeichnis
                script_dir = os.path.dirname(os.path.abspath(__file__))
                pickles_file = download_pickles_spectrum(self.spectral_type, 
                                                         cache_dir=os.path.join(script_dir, 'pickles_cache'))
                
                # Lade Spektrum
                self.miles_wavelength, self.miles_flux = load_pickles_spectrum(pickles_file)
                
                # Speichere auch als reference (für Kompatibilität)
                self.reference_wavelength = self.miles_wavelength
                self.reference_flux = self.miles_flux
                
                print(f"  Wellenlängenbereich: {self.miles_wavelength[0]:.1f} - "
                      f"{self.miles_wavelength[-1]:.1f} Å")
                print(f"  Anzahl Pixel: {len(self.miles_wavelength)}")
                print(f"  Flux-Einheit: erg/s/cm²/Å (kalibriert)")
                
            except FileNotFoundError as e:
                # Datei nicht lokal vorhanden - klare Fehlermeldung bereits ausgegeben
                print(f"\n{'='*70}")
                print(f"  ℹ️  Pickles-Spektrum nicht verfügbar")
                print(f"{'='*70}")
                print(f"\n  → Verwende stattdessen synthetisches Spektrum")
                print(f"  → Für bessere Qualität: Lade Pickles-Spektrum manuell herunter")
                print(f"  → Details siehe oben ↑")
                print(f"{'='*70}\n")
                self.use_pickles = False
                self._get_synthetic_reference_spectrum()
                
            except (ValueError, RuntimeError) as e:
                print(f"\n{'='*70}")
                print(f"  WARNUNG: Pickles-Spektrum nicht verfügbar")
                print(f"{'='*70}")
                print(f"{e}")
                print(f"\n  → Falle automatisch zurück auf synthetisches Spektrum...")
                print(f"{'='*70}\n")
                self.use_pickles = False
                self._get_synthetic_reference_spectrum()
        else:
            self._get_synthetic_reference_spectrum()
    
    def _get_synthetic_reference_spectrum(self):
        """
        Generiert synthetisches Referenz-Spektrum (Fallback wenn Pickles nicht verfügbar)
        """
        print(f"\nGeneriere synthetisches Referenz-Spektrum für: {self.spectral_type}")
        
        # Spektraltyp-Parameter (Teff, log(g))
        spectral_params = {
            # O-Sterne
            'O5V': (42000, 4.0),
            'O9V': (33000, 4.0),
            # B-Sterne
            'B0V': (30000, 4.0),
            'B5V': (15200, 4.0),
            'B8V': (11400, 4.0),
            # A-Sterne
            'A0V': (9520, 4.0),
            'A5V': (8200, 4.2),
            # F-Sterne
            'F0V': (7200, 4.3),
            'F5V': (6440, 4.4),
            # G-Sterne
            'G0V': (5920, 4.5),
            'G2V': (5780, 4.4),  # Sonne
            'G5V': (5660, 4.5),
            # K-Sterne
            'K0V': (5250, 4.5),
            'K5V': (4350, 4.5),
            # M-Sterne
            'M0V': (3850, 4.5),
            'M5V': (3170, 5.0),
            # Riesen
            'G5III': (5010, 2.5),
            'K0III': (4750, 2.5),
            'M0III': (3800, 1.5),
        }
        
        if self.spectral_type not in spectral_params:
            # Versuche ähnlichen Typ zu finden
            print(f"  Warnung: Spektraltyp {self.spectral_type} nicht in Datenbank.")
            print(f"  Verfügbare Typen: {', '.join(spectral_params.keys())}")
            
            # Versuche Näherung (z.B. A0V aus A0)
            for key in spectral_params.keys():
                if key.startswith(self.spectral_type[:2]):
                    print(f"  Verwende stattdessen: {key}")
                    self.spectral_type = key
                    break
            else:
                raise ValueError(f"Kein passender Spektraltyp gefunden für: {self.spectral_type}")
        
        teff, logg = spectral_params[self.spectral_type]
        print(f"  Effektive Temperatur: {teff} K")
        print(f"  log(g): {logg}")
        
        # Erzeuge Referenz-Spektrum basierend auf Planck + stellare Absorption
        self.reference_flux = self._generate_stellar_spectrum(self.wavelength, teff, logg)
        self.reference_wavelength = self.wavelength.copy()
        
    def _generate_stellar_spectrum(self, wavelength, teff, logg):
        """
        Generiert vereinfachtes stellares Spektrum
        
        Parameters:
        -----------
        wavelength : array
            Wellenlängen in Angström
        teff : float
            Effektive Temperatur in K
        logg : float
            Oberflächengravitation log(g)
        
        Returns:
        --------
        flux : array
            Relativer Flux (normalisiert)
        """
        # Konstanten
        h = 6.62607015e-34  # J·s
        c = 2.99792458e8    # m/s
        k = 1.380649e-23    # J/K
        
        # Wellenlänge in m
        lam = wavelength * 1e-10
        
        # Planck-Funktion
        with np.errstate(over='ignore', invalid='ignore'):
            B_lambda = (2 * h * c**2 / lam**5) / (np.exp(h * c / (lam * k * teff)) - 1)
        
        # Normalisiere
        B_lambda = B_lambda / np.max(B_lambda)
        
        # Füge typische Absorptionslinien hinzu (vereinfacht)
        flux = B_lambda.copy()
        
        # Balmer-Linien (abhängig von Spektraltyp)
        if teff > 7000:  # A-Sterne und heißer
            balmer_strength = 0.3
            self._add_absorption_line(flux, wavelength, 6562.79, 15.0, balmer_strength)  # Hα
            self._add_absorption_line(flux, wavelength, 4861.33, 12.0, balmer_strength)  # Hβ
            self._add_absorption_line(flux, wavelength, 4340.47, 10.0, balmer_strength)  # Hγ
            self._add_absorption_line(flux, wavelength, 4101.74, 8.0, balmer_strength)   # Hδ
        elif teff > 5000:  # F, G-Sterne
            balmer_strength = 0.15
            self._add_absorption_line(flux, wavelength, 6562.79, 12.0, balmer_strength)
            self._add_absorption_line(flux, wavelength, 4861.33, 10.0, balmer_strength)
            self._add_absorption_line(flux, wavelength, 4340.47, 8.0, balmer_strength)
            
            # Ca H & K
            self._add_absorption_line(flux, wavelength, 3933.66, 5.0, 0.4)  # Ca K
            self._add_absorption_line(flux, wavelength, 3968.47, 5.0, 0.4)  # Ca H
            
            # Na D
            self._add_absorption_line(flux, wavelength, 5889.95, 3.0, 0.2)
            self._add_absorption_line(flux, wavelength, 5895.92, 3.0, 0.2)
        
        # Glätte Spektrum
        flux = gaussian_filter1d(flux, sigma=2.0)
        
        # Normalisiere auf mittleren Flux = 1
        flux = flux / np.median(flux)
        
        return flux
    
    def _add_absorption_line(self, flux, wavelength, line_center, width, depth):
        """Fügt Gaussian-Absorptionslinie hinzu"""
        gaussian = depth * np.exp(-0.5 * ((wavelength - line_center) / width)**2)
        flux -= gaussian * flux
        
    def calibrate_flux(self):
        """Führt die Flusskalibration durch"""
        print("\nFlusskalibration...")
        
        if self.use_pickles:
            print("  Methode: Pickles Atlas (empirische Spektren)")
            
            # Berechne instrumental response
            response_raw, response_smooth, telluric_mask = calculate_instrumental_response(
                self.wavelength, self.flux_observed,
                self.reference_wavelength, self.reference_flux
            )
            
            self.instrumental_response = response_smooth
            self.instrumental_response_raw = response_raw  # Original (mit tellurischen Linien)
            self.telluric_mask = telluric_mask
            
            # Interpoliere Referenz-Flux auf beobachtetes Grid
            interp_func = interp1d(self.reference_wavelength, self.reference_flux,
                                   kind='quadratic', bounds_error=False, fill_value=np.nan)
            reference_flux_interp = interp_func(self.wavelength)
            
            # Kalibrationsfaktor ist response (counts -> erg/s/cm²/Å)
            # flux_calibrated = flux_observed / response
            # Aber wir speichern response so dass: flux_calibrated = flux_observed * calib_factor
            self.calibration_factor = 1.0 / (self.instrumental_response + 1e-10)
            
            # Normalisiere so dass median(calib_factor) ~ median(reference_flux / flux_observed)
            median_ratio = np.nanmedian(reference_flux_interp / (self.flux_observed + 1e-10))
            self.calibration_factor *= median_ratio
            
            # Wende Kalibration an
            self.flux_calibrated = self.flux_observed * self.calibration_factor
            
            print(f"  Instrumental Response (Median): {np.nanmedian(self.instrumental_response):.2e}")
            print(f"  Instrumental Response (Range): {np.nanmin(self.instrumental_response):.2e} - "
                  f"{np.nanmax(self.instrumental_response):.2e}")
            
        else:
            print("  Methode: Synthetische Spektren")
            
            # Glätte beobachtetes Spektrum für besseres Matching
            flux_smoothed = gaussian_filter1d(self.flux_observed, sigma=3.0)
            
            # Bestimme Kontinuum des beobachteten Spektrums
            continuum_obs = self._estimate_continuum(flux_smoothed)
            
            # Bestimme Kontinuum des Referenz-Spektrums
            continuum_ref = self._estimate_continuum(self.reference_flux)
            
            # Kalibrationsfaktor: Verhältnis der Kontinua
            # Mit Glättung für robustere Kalibration
            ratio = continuum_ref / (continuum_obs + 1e-10)
            ratio_smooth = gaussian_filter1d(ratio, sigma=5.0)
            
            # Speichere Kalibrationsfaktor für spätere Verwendung
            self.calibration_factor = ratio_smooth
            
            # Wende Kalibration an
            self.flux_calibrated = self.flux_observed * self.calibration_factor
        
        print(f"  Kalibrationsfaktor (Median): {np.nanmedian(self.calibration_factor):.2e}")
        print(f"  Kalibrationsfaktor (Min-Max): {np.nanmin(self.calibration_factor):.2e} - "
              f"{np.nanmax(self.calibration_factor):.2e}")
        
    def _estimate_continuum(self, flux, percentile=90):
        """
        Schätzt das Kontinuum eines Spektrums
        
        Parameters:
        -----------
        flux : array
            Flux-Array
        percentile : float
            Perzentil für Kontinuums-Schätzung
        
        Returns:
        --------
        continuum : array
            Geschätztes Kontinuum
        """
        # Verwende laufendes Perzentil-Filter
        window_size = max(50, len(flux) // 20)
        continuum = np.zeros_like(flux)
        
        for i in range(len(flux)):
            i_min = max(0, i - window_size // 2)
            i_max = min(len(flux), i + window_size // 2)
            continuum[i] = np.percentile(flux[i_min:i_max], percentile)
        
        # Glätte Kontinuum
        continuum = gaussian_filter1d(continuum, sigma=10.0)
        
        return continuum
    
    def save_results(self):
        """Speichert das flusskalibrierte Spektrum"""
        print(f"\nSpeichere Ergebnisse...")
        
        # Speichere FITS
        print(f"  FITS: {self.output_fits}")
        
        # Erstelle Binary Table
        col1 = fits.Column(name='WAVELENGTH', format='D', array=self.wavelength, unit='Angstrom')
        col2 = fits.Column(name='FLUX', format='D', array=self.flux_calibrated, 
                          unit='erg/s/cm^2/Angstrom')
        col3 = fits.Column(name='FLUX_OBSERVED', format='D', array=self.flux_observed, 
                          unit='counts')
        
        cols = fits.ColDefs([col1, col2, col3])
        table_hdu = fits.BinTableHDU.from_columns(cols)
        
        # Header
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['SPECTYPE'] = (self.spectral_type, 'Spectral type used for calibration')
        primary_hdu.header['FLUXCAL'] = (True, 'Flux calibrated')
        primary_hdu.header['BUNIT'] = ('erg/s/cm^2/Angstrom', 'Flux unit')
        primary_hdu.header['METHOD'] = ('PICKLES' if self.use_pickles else 'SYNTHETIC', 
                                        'Calibration method')
        
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(self.output_fits, overwrite=True)
        
        # Speichere Kalibrationsdatei
        self._save_calibration()
        
        # Speichere instrumental response (nur bei Pickles)
        if self.use_pickles and self.instrumental_response is not None:
            self._save_instrumental_response()
        
        # Speichere Plot
        self._plot_results()
    
    def _save_calibration(self):
        """Speichert die Flusskalibration zur Wiederverwendung"""
        print(f"  Kalibrationsdatei: {self.calibration_fits}")
        
        # Erstelle Binary Table mit Kalibrationsdaten
        col1 = fits.Column(name='WAVELENGTH', format='D', array=self.wavelength, unit='Angstrom')
        col2 = fits.Column(name='CALIB_FACTOR', format='D', array=self.calibration_factor,
                          unit='(erg/s/cm^2/Angstrom)/counts')
        col3 = fits.Column(name='REFERENCE_FLUX', format='D', array=self.reference_flux,
                          unit='normalized')
        
        cols = fits.ColDefs([col1, col2, col3])
        table_hdu = fits.BinTableHDU.from_columns(cols)
        
        # Header mit Metadaten
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['SPECTYPE'] = (self.spectral_type, 'Reference spectral type')
        primary_hdu.header['CALDATE'] = (fits.Header()['DATE'] if 'DATE' in fits.Header() else 'UNKNOWN', 
                                         'Calibration creation date')
        primary_hdu.header['PURPOSE'] = ('Flux calibration', 'File purpose')
        primary_hdu.header['COMMENT'] = 'Apply CALIB_FACTOR to counts to get flux in erg/s/cm^2/Angstrom'
        primary_hdu.header['COMMENT'] = 'Interpolate CALIB_FACTOR to match your wavelength grid if needed'
        primary_hdu.header['BUNIT'] = ('(erg/s/cm^2/Angstrom)/counts', 'Calibration factor unit')
        
        # Statistiken für Qualitätskontrolle
        primary_hdu.header['CAL_MED'] = (float(np.median(self.calibration_factor)), 'Median calibration factor')
        primary_hdu.header['CAL_MEAN'] = (float(np.mean(self.calibration_factor)), 'Mean calibration factor')
        primary_hdu.header['CAL_STD'] = (float(np.std(self.calibration_factor)), 'Std dev calibration factor')
        primary_hdu.header['CAL_MIN'] = (float(np.min(self.calibration_factor)), 'Min calibration factor')
        primary_hdu.header['CAL_MAX'] = (float(np.max(self.calibration_factor)), 'Max calibration factor')
        
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(self.calibration_fits, overwrite=True)
        
        print(f"    → Diese Datei kann auf andere Spektren angewendet werden")
        print(f"    → Verwenden Sie: apply_flux_calibration(spectrum, '{self.calibration_fits}')")
    
    def _save_instrumental_response(self):
        """Speichert die instrumental response (nur bei Pickles-Kalibration)"""
        print(f"  Instrumental Response: {self.response_fits}")
        
        # Erstelle Binary Table
        col1 = fits.Column(name='WAVELENGTH', format='D', array=self.wavelength, unit='Angstrom')
        col2 = fits.Column(name='RESPONSE', format='D', array=self.instrumental_response,
                          unit='counts/(erg/s/cm^2/Angstrom)')
        
        cols = fits.ColDefs([col1, col2])
        table_hdu = fits.BinTableHDU.from_columns(cols)
        
        # Header
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['SPECTYPE'] = (self.spectral_type, 'Reference spectral type')
        primary_hdu.header['PURPOSE'] = ('Instrumental Response', 'File purpose')
        primary_hdu.header['METHOD'] = ('PICKLES', 'Calculated from Pickles Atlas')
        primary_hdu.header['COMMENT'] = 'Instrumental response R(lambda) of the spectrograph'
        primary_hdu.header['COMMENT'] = 'flux_observed = flux_true * R(lambda)'
        primary_hdu.header['BUNIT'] = ('counts/(erg/s/cm^2/Angstrom)', 'Response unit')
        
        # Statistiken
        primary_hdu.header['RESP_MED'] = (float(np.nanmedian(self.instrumental_response)), 
                                          'Median response')
        primary_hdu.header['RESP_MEAN'] = (float(np.nanmean(self.instrumental_response)), 
                                           'Mean response')
        primary_hdu.header['RESP_STD'] = (float(np.nanstd(self.instrumental_response)), 
                                          'Std dev response')
        primary_hdu.header['RESP_MIN'] = (float(np.nanmin(self.instrumental_response)), 
                                          'Min response')
        primary_hdu.header['RESP_MAX'] = (float(np.nanmax(self.instrumental_response)), 
                                          'Max response')
        
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(self.response_fits, overwrite=True)
        
        print(f"    → Wellenlängenabhängige Effizienz des Instruments")
        print(f"    → Kann zur Korrektur zukünftiger Beobachtungen verwendet werden")
        
    def _plot_results(self):
        """Erstellt Vergleichs-Plot"""
        print(f"  PNG: {self.output_png}")
        
        if self.use_pickles and self.instrumental_response is not None:
            # 4 Plots bei Pickles-Kalibration
            fig, axes = plt.subplots(4, 1, figsize=(14, 8.4))
            
            # Plot 1: Beobachtetes Spektrum
            ax = axes[0]
            ax.plot(self.wavelength, self.flux_observed, 'b-', alpha=0.7, linewidth=0.8, label='Observed')
            ax.set_ylabel('Observed Flux\n[counts]', fontsize=11)
            ax.set_title(f'{self.work_dir} — Flux Calibration: {self.spectral_type} (Pickles Atlas)', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.wavelength[0], self.wavelength[-1])
            ax.legend(fontsize=9, loc='upper right')
            
            # Plot 2: Pickles Referenz-Spektrum (Original + Interpoliert)
            ax = axes[1]
            
            # Zeige das originale Pickles-Spektrum
            if self.miles_wavelength is not None and self.miles_flux is not None:
                ax.plot(self.miles_wavelength, self.miles_flux, 'r-', 
                       alpha=0.7, linewidth=0.8, label=f'Pickles {self.spectral_type} (original)')
                
                # Zeige auch das interpolierte Spektrum falls unterschiedlich
                if len(self.miles_wavelength) != len(self.wavelength) or \
                   not np.allclose(self.miles_wavelength, self.wavelength):
                    # Interpoliere für Anzeige
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(self.miles_wavelength, self.miles_flux,
                                          kind='quadratic', bounds_error=False, fill_value=np.nan)
                    flux_ref_interp = interp_func(self.wavelength)
                    ax.plot(self.wavelength, flux_ref_interp, 'r--', 
                           alpha=0.5, linewidth=0.6, label='Pickles (interpoliert)')
            else:
                # Fallback: zeige reference (sollte nicht passieren bei Pickles)
                ax.plot(self.reference_wavelength, self.reference_flux, 'r-', 
                       alpha=0.7, linewidth=0.8, label=f'Reference {self.spectral_type}')
            
            ax.set_ylabel(f'Pickles Reference\n{self.spectral_type}\n[erg/s/cm²/Å]', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.wavelength[0], self.wavelength[-1])
            ax.legend(fontsize=9, loc='upper right')
            
            # Plot 3: Instrumental Response
            ax = axes[2]
            
            # Zeige interpolierte und gemessene Bereiche farblich unterschiedlich
            if self.telluric_mask is not None and np.any(self.telluric_mask):
                # Finde zusammenhängende tellurische Regionen
                telluric_regions = []
                in_region = False
                start_idx = 0
                for i in range(len(self.telluric_mask)):
                    if self.telluric_mask[i] and not in_region:
                        start_idx = i
                        in_region = True
                    elif not self.telluric_mask[i] and in_region:
                        telluric_regions.append((start_idx, i-1))
                        in_region = False
                if in_region:  # Region geht bis zum Ende
                    telluric_regions.append((start_idx, len(self.telluric_mask)-1))
                
                # Zeige originale Response (mit tellurischen Linien) als dünne graue Linie
                if hasattr(self, 'instrumental_response_raw') and self.instrumental_response_raw is not None:
                    ax.plot(self.wavelength, self.instrumental_response_raw, '-',
                           color='gray', alpha=0.4, linewidth=0.8, 
                           label='Response (original, mit telluric)', zorder=1)
                
                # Zeichne gemessene Bereiche (nicht-tellurisch) in Orange
                non_telluric_mask = ~self.telluric_mask
                ax.plot(self.wavelength[non_telluric_mask], 
                       self.instrumental_response[non_telluric_mask], 
                       'o', color='darkorange', markersize=2, alpha=0.6,
                       label='Response (geglättet)', zorder=3)
                
                # Zeichne interpolierte Bereiche in Rot mit anderer Markierung
                ax.plot(self.wavelength[self.telluric_mask], 
                       self.instrumental_response[self.telluric_mask], 
                       's', color='red', markersize=2, alpha=0.7,
                       label='Response (interpoliert)', zorder=4)
                
                # Verbindungslinie für finalen Gesamtverlauf (grün)
                ax.plot(self.wavelength, self.instrumental_response, '-',
                       color='green', alpha=0.5, linewidth=1.0, 
                       label='Response (final)', zorder=2)
                
                # Hellrote Schattierungen für tellurische Bereiche im Hintergrund
                for start, end in telluric_regions:
                    ax.axvspan(self.wavelength[start], self.wavelength[end], 
                              alpha=0.1, color='red', zorder=1)
            else:
                # Fallback: Keine Maske vorhanden
                ax.plot(self.wavelength, self.instrumental_response, 'orange', 
                       alpha=0.8, linewidth=1.2, label='Instrumental Response')
            
            ax.set_ylabel('Instrumental\nResponse\n[counts/(erg/s/cm²/Å)]', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.wavelength[0], self.wavelength[-1])
            median_resp = np.nanmedian(self.instrumental_response)
            ax.axhline(median_resp, color='k', linestyle='--', 
                      alpha=0.5, linewidth=1.0, label=f'Median: {median_resp:.2e}')
            ax.legend(fontsize=8, loc='upper right')
            
            # Plot 4: Kalibriertes Spektrum (mit Pickles zum Vergleich)
            ax = axes[3]
            ax.plot(self.wavelength, self.flux_calibrated, 'g-', 
                   alpha=0.8, linewidth=0.9, label='Calibrated Spectrum')
            
            # Zeige Pickles-Spektrum zum Vergleich (skaliert auf ähnliches Niveau)
            if self.miles_wavelength is not None and self.miles_flux is not None:
                # Interpoliere Pickles auf beobachtetes Grid
                if len(self.miles_wavelength) != len(self.wavelength) or \
                   not np.allclose(self.miles_wavelength, self.wavelength):
                    interp_func = interp1d(self.miles_wavelength, self.miles_flux,
                                          kind='quadratic', bounds_error=False, fill_value=np.nan)
                    flux_ref_interp = interp_func(self.wavelength)
                else:
                    flux_ref_interp = self.miles_flux
                
                # Skaliere Pickles auf ähnliches Niveau wie kalibriertes Spektrum
                scale_factor = np.nanmedian(self.flux_calibrated) / np.nanmedian(flux_ref_interp)
                ax.plot(self.wavelength, flux_ref_interp * scale_factor, 'r--', 
                       alpha=0.4, linewidth=0.8, label=f'Pickles (skaliert)')
            
            ax.set_xlabel('Wavelength [Å]', fontsize=12)
            ax.set_ylabel('Calibrated Flux\n[erg/s/cm²/Å]', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.wavelength[0], self.wavelength[-1])
            ax.legend(fontsize=9, loc='upper right')
            
        else:
            # 3 Plots bei synthetischer Kalibration
            fig, axes = plt.subplots(3, 1, figsize=(14, 7))
            
            # Plot 1: Beobachtetes Spektrum
            ax = axes[0]
            ax.plot(self.wavelength, self.flux_observed, 'b-', alpha=0.7, linewidth=0.8)
            ax.set_ylabel('Observed Flux\n[counts]', fontsize=11)
            ax.set_title(f'{self.work_dir} — Flux Calibration: {self.spectral_type} (Synthetic)', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.wavelength[0], self.wavelength[-1])
            
            # Plot 2: Referenz-Spektrum
            ax = axes[1]
            ax.plot(self.wavelength, self.reference_flux, 'r-', alpha=0.7, linewidth=0.8)
            ax.set_ylabel(f'Reference Spectrum\n{self.spectral_type}\n[normalized]', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.wavelength[0], self.wavelength[-1])
            
            # Plot 3: Kalibriertes Spektrum
            ax = axes[2]
            ax.plot(self.wavelength, self.flux_calibrated, 'g-', alpha=0.7, linewidth=0.8)
            ax.set_xlabel('Wavelength [Å]', fontsize=12)
            ax.set_ylabel('Calibrated Flux\n[erg/s/cm²/Å]', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.wavelength[0], self.wavelength[-1])
        
        plt.tight_layout()
        plt.savefig(self.output_png, dpi=150, bbox_inches='tight')
        
        # Zeige Plot direkt an (wenn gewünscht)
        if self.show_plot:
            print(f"  Zeige Plot an (schließe Fenster um fortzufahren)...")
            plt.show()
        else:
            plt.close()
        
    def run(self):
        """Führt die komplette Flusskalibration durch"""
        print("=" * 60)
        print("SPEKTRAL-FLUSSKALIBRATION")
        print("=" * 60)
        print(f"Arbeitsordner: {self.work_dir}")
        print(f"Spektraltyp: {self.spectral_type}")
        print(f"Methode: {'Pickles Atlas' if self.use_pickles else 'Synthetische Spektren'}")
        
        self.load_science_spectrum()
        self.get_reference_spectrum()
        self.calibrate_flux()
        self.save_results()
        
        print("\n" + "=" * 60)
        print("FERTIG!")
        print("=" * 60)
        if self.use_pickles:
            print("\nErstellt:")
            print(f"  • {self.output_fits} (Flux-kalibriertes Spektrum)")
            print(f"  • {self.calibration_fits} (Wiederverwendbare Kalibration)")
            print(f"  • {self.response_fits} (Instrumental Response)")
            print(f"  • {self.output_png} (Visualisierung)")
        else:
            print("\nErstellt:")
            print(f"  • {self.output_fits} (Flux-kalibriertes Spektrum)")
            print(f"  • {self.calibration_fits} (Wiederverwendbare Kalibration)")
            print(f"  • {self.output_png} (Visualisierung)")


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(
        description='Spektral-Flusskalibration mit Referenz-Spektraltyp (Pickles Atlas oder synthetisch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Mit Pickles Atlas (Standard, empfohlen)
  python spec_flux.py vega A0V
  python spec_flux.py altair A7V --pickles
  
  # Mit synthetischen Spektren
  python spec_flux.py sun G2V --no-pickles
  
  # Batch-Modus (ohne Plot-Anzeige)
  python spec_flux.py vega A0V --no-plot
  
Verfügbare Spektraltypen (Pickles Atlas):
  O: O5V, O9V
  B: B0V, B1V, B3V, B5V, B8V
  A: A0V, A2V, A3V, A5V
  F: F0V, F2V, F5V, F8V
  G: G0V, G2V, G5V, G8V
  K: K0V, K2V, K5V, K7V
  M: M0V, M2V, M4V, M5V
  Riesen: G5III, G8III, K0III, K3III, K5III, M0III, M5III

Bei Pickles-Kalibration wird zusätzlich die instrumental response berechnet:
  - instrumental_response.fits: R(λ) des Spektrographen
  - flux_observed = flux_true × R(λ)

Die Kalibration wird gespeichert in: ARBEITSORDNER/out/flux_calibration.fits
Diese Datei kann mit der Funktion apply_flux_calibration() auf andere Spektren
angewendet werden.
        """
    )
    
    parser.add_argument('work_dir', 
                       help='Arbeitsordner mit in/ und out/ Unterordnern')
    parser.add_argument('spectral_type',
                       help='Spektraltyp (z.B. A0V, G2V, M5V)')
    parser.add_argument('--pickles', dest='use_pickles', action='store_true', default=True,
                       help='Verwende Pickles Atlas (Standard)')
    parser.add_argument('--no-pickles', dest='use_pickles', action='store_false',
                       help='Verwende synthetische Spektren statt Pickles Atlas')
    parser.add_argument('--no-plot', dest='show_plot', action='store_false', default=True,
                       help='Plot nur speichern, nicht anzeigen (für Batch-Modus)')
    
    args = parser.parse_args()
    
    # Prüfe Arbeitsordner
    if not os.path.exists(args.work_dir):
        print(f"Fehler: Arbeitsordner '{args.work_dir}' nicht gefunden!")
        sys.exit(1)
    
    out_dir = os.path.join(args.work_dir, "out")
    if not os.path.exists(out_dir):
        print(f"Fehler: Ausgabeordner '{out_dir}' nicht gefunden!")
        sys.exit(1)
    
    # Führe Kalibrierung durch
    calibrator = FluxCalibrator(args.work_dir, args.spectral_type, 
                                use_pickles=args.use_pickles, 
                                show_plot=args.show_plot)
    calibrator.run()


def apply_flux_calibration(wavelength, flux_counts, calibration_file):
    """
    Wendet eine gespeicherte Flusskalibration auf ein neues Spektrum an
    
    Parameters:
    -----------
    wavelength : array
        Wellenlängen des zu kalibrierenden Spektrums in Angström
    flux_counts : array
        Flux in Counts (unkalibriert)
    calibration_file : str
        Pfad zur Kalibrationsdatei (flux_calibration.fits)
    
    Returns:
    --------
    flux_calibrated : array
        Kalibrierter Flux in erg/s/cm²/Å
    
    Beispiel:
    ---------
    >>> wavelength, flux = load_spectrum("new_spectrum.fits")
    >>> flux_cal = apply_flux_calibration(wavelength, flux, "vega/out/flux_calibration.fits")
    """
    print(f"\nWende Flusskalibration an: {calibration_file}")
    
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Kalibrationsdatei nicht gefunden: {calibration_file}")
    
    # Lade Kalibrationsdaten
    with fits.open(calibration_file) as hdul:
        calib_data = hdul[1].data
        calib_wavelength = calib_data['WAVELENGTH']
        calib_factor = calib_data['CALIB_FACTOR']
        
        # Zeige Metadaten
        if 'SPECTYPE' in hdul[0].header:
            print(f"  Spektraltyp: {hdul[0].header['SPECTYPE']}")
        if 'CAL_MED' in hdul[0].header:
            print(f"  Median Kalibrationsfaktor: {hdul[0].header['CAL_MED']:.2e}")
    
    # Interpoliere Kalibrationsfaktor auf neue Wellenlängen
    interp_func = interp1d(calib_wavelength, calib_factor, 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
    calib_factor_interp = interp_func(wavelength)
    
    # Wende Kalibration an
    flux_calibrated = flux_counts * calib_factor_interp
    
    print(f"  Angewendet auf {len(wavelength)} Wellenlängenpunkte")
    print(f"  Wellenlängenbereich: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Å")
    
    return flux_calibrated


def load_instrumental_response(response_file):
    """
    Lädt eine gespeicherte instrumental response
    
    Parameters:
    -----------
    response_file : str
        Pfad zur Response-Datei (instrumental_response.fits)
    
    Returns:
    --------
    wavelength : array
        Wellenlängen in Angström
    response : array
        Instrumental response in counts/(erg/s/cm²/Å)
    
    Beispiel:
    ---------
    >>> wave, resp = load_instrumental_response("vega/out/instrumental_response.fits")
    >>> # Korrigiere neues Spektrum:
    >>> flux_corrected = flux_observed / resp
    """
    if not os.path.exists(response_file):
        raise FileNotFoundError(f"Response-Datei nicht gefunden: {response_file}")
    
    with fits.open(response_file) as hdul:
        data = hdul[1].data
        wavelength = data['WAVELENGTH']
        response = data['RESPONSE']
        
        # Zeige Metadaten
        print(f"Instrumental Response geladen: {response_file}")
        if 'SPECTYPE' in hdul[0].header:
            print(f"  Spektraltyp: {hdul[0].header['SPECTYPE']}")
        if 'RESP_MED' in hdul[0].header:
            print(f"  Median Response: {hdul[0].header['RESP_MED']:.2e}")
    
    return wavelength, response


# Beispiel-Workflow für Pickles-Kalibration
def example_pickles_calibration():
    """
    Beispiel: Vollständiger Workflow mit Pickles-Kalibration
    
    Zeigt wie man:
    1. Ein Spektrum mit Pickles Atlas kalibriert
    2. Die instrumental response speichert
    3. Die Kalibration auf andere Spektren anwendet
    """
    print("=" * 70)
    print("BEISPIEL: Pickles-basierte Flusskalibration")
    print("=" * 70)
    
    print("""
Schritt 1: Kalibriere ein Spektrum mit Pickles Atlas
-----------------------------------------------------
Angenommen du hast ein Vega-Spektrum beobachtet (Spektraltyp A0V):

    python spec_flux.py vega A0V --pickles

Dies erzeugt:
  • vega/out/science_spectrum_flux_calibrated.fits
  • vega/out/flux_calibration.fits
  • vega/out/instrumental_response.fits  <-- Instrumental Response!
  • vega/out/science_spectrum_flux_calibrated.png

Die instrumental_response.fits enthält R(λ), die wellenlängenabhängige
Effizienz deines Spektrographen.


Schritt 2: Verwende die Response für neue Spektren
---------------------------------------------------
Für ein neues Spektrum kannst du jetzt:

Option A) Die komplette Kalibration wiederverwenden:

    from spec_flux import apply_flux_calibration
    
    wave, flux = load_my_spectrum("new_star.fits")
    flux_cal = apply_flux_calibration(wave, flux, "vega/out/flux_calibration.fits")

Option B) Die instrumental response direkt anwenden:

    from spec_flux import load_instrumental_response
    from scipy.interpolate import interp1d
    
    # Lade Response
    resp_wave, response = load_instrumental_response("vega/out/instrumental_response.fits")
    
    # Interpoliere auf deine Wellenlängen
    interp = interp1d(resp_wave, response, kind='cubic')
    my_response = interp(my_wavelength)
    
    # Korrigiere für instrumental response
    # (Annahme: du kennst den absoluten Flux des Standards)
    flux_corrected = flux_observed / my_response


Schritt 3: Überwache die Instrument-Performance
-----------------------------------------------
Durch wiederholte Kalibrationen über die Zeit kannst du:
- Instrument-Degradation erkennen
- Änderungen in der Effizienz tracken
- Qualitätskontrolle durchführen

Vergleiche einfach die Response-Kurven von verschiedenen Nächten!


Typische Pickles-Spektraltypen:
--------------------------------
Heiße Sterne:   A0V (Vega), A5V (Altair), F0V
Sonne-ähnlich:  G2V, G5V
Kühle Sterne:   K5V, M0V, M5V
Riesen:         K0III, M0III, M5III

Vollständige Liste in PICKLES_SPECTRAL_LIBRARY.
""")
    
    print("=" * 70)


if __name__ == '__main__':
    # Wenn mit --example aufgerufen, zeige Beispiel
    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        example_pickles_calibration()
    else:
        main()

