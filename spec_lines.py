#!/usr/bin/env python3
"""
spec_lines.py - Vergleich von beobachtetem Spektrum mit Kurucz-Modellen

Lädt ein flusskalibiertes Spektrum und vergleicht es mit einem Kurucz-Modellspektrum
bei gegebenen stellaren Parametern (Teff, log g).

Usage:
    python spec_lines.py ARBEITSORDNER --line WAVELENGTH --teff TEFF --logg LOGG --resolution FWHM [OPTIONS]
    
Beispiele:
    # Mit Kreuzkorrelation (Standard)
    python spec_lines.py vega --line 6562.8 --teff 9600 --logg 3.9 --resolution 2.5
    
    # Mit größerem Intervall für Kreuzkorrelation (±1500 Å um Linie, empfohlen bei engem Plot)
    python spec_lines.py vega --line 6562.8 --teff 9600 --logg 3.9 --resolution 2.5 --xcorr-width 3000
    
    # Ohne Kreuzkorrelation
    python spec_lines.py vega --line 6562.8 --teff 9600 --logg 3.9 --resolution 2.5 --no-cross-correlate
"""

import os
import sys
import argparse
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import correlate
import matplotlib.pyplot as plt
import urllib.request
import gzip
import shutil
import ssl

def load_calibrated_spectrum(work_dir):
    """
    Lädt das flusskalibrierte Spektrum aus dem Arbeitsordner.
    
    Parameters:
    -----------
    work_dir : str
        Arbeitsordner mit out/ Unterordner
        
    Returns:
    --------
    wavelength : array
        Wellenlängen in Angström
    flux : array
        Kalibrierter Flux in erg/s/cm²/Å
    """
    out_dir = os.path.join(work_dir, "out")
    fits_file = os.path.join(out_dir, "science_spectrum_flux_calibrated.fits")
    
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"Kalibriertes Spektrum nicht gefunden: {fits_file}")
    
    print(f"Lade kalibriertes Spektrum: {fits_file}")
    hdul = fits.open(fits_file)
    data = hdul[1].data if len(hdul) > 1 else hdul[0].data
    hdul.close()
    
    wavelength = np.array(data['WAVELENGTH'], dtype=float).flatten()
    flux = np.array(data['FLUX'], dtype=float).flatten()
    
    # Entferne NaN und Inf
    valid = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength = wavelength[valid]
    flux = flux[valid]
    
    print(f"  Wellenlängenbereich: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Å")
    print(f"  {len(wavelength)} Datenpunkte")
    
    return wavelength, flux


def download_phoenix_model(teff, logg, model_dir="phoenix_models",
                          metallicity=0.0,
                          base_url="ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/"):
    """
    Lädt ein PHOENIX-Modellspektrum herunter (sehr hochaufgelöst).
    
    Parameters:
    -----------
    teff : float
        Effektivtemperatur in K
    logg : float
        Oberflächenschwere
    model_dir : str
        Zielverzeichnis
    metallicity : float
        Metallizität [Fe/H] in dex (default: 0.0)
    base_url : str
        Basis-URL für PHOENIX-Modelle
        
    Returns:
    --------
    filepath : str
        Pfad zur heruntergeladenen Datei
        
    Notes:
    ------
    PHOENIX Göttingen Spectral Library (ACES-AGSS-COND-2011):
    https://phoenix.astro.physik.uni-goettingen.de/
    
    Grid:
    - Teff: 2300-7000K (steps: 100K), 7000-12000K (steps: 200K)
    - log g: 0.0-6.0 (steps: 0.5)
    - [Fe/H]: -4.0 to +1.0
    - Sehr hoch aufgelöst: ~1.6 Mio Punkte, 500Å - 55000Å
    """
    # Runde Parameter
    if teff <= 7000:
        teff_rounded = round(teff / 100) * 100
    else:
        teff_rounded = round(teff / 200) * 200
    
    logg_rounded = round(logg / 0.5) * 0.5
    
    # Metallizität formatieren: 0.0 -> -0.0, -0.5 -> -0.5, +0.5 -> +0.5
    if metallicity >= 0:
        met_str = f"+{abs(metallicity):.1f}" if metallicity > 0 else "-0.0"
    else:
        met_str = f"{metallicity:.1f}"
    
    # Z-Verzeichnis: Z-0.0, Z-0.5, Z+0.5, etc.
    z_dir = f"Z{met_str}"
    
    # Dateiname: lte10000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    filename = f"lte{teff_rounded:05d}-{logg_rounded:.2f}{met_str}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    
    # URL
    url = f"{base_url}PHOENIX-ACES-AGSS-COND-2011/{z_dir}/{filename}"
    
    # Lokaler Dateiname
    local_filename = f"phoenix_t{teff_rounded}_g{logg_rounded:.1f}_z{met_str}.fits"
    filepath = os.path.join(model_dir, local_filename)
    
    print(f"  Download PHOENIX ACES:")
    print(f"    Teff={teff_rounded}K, log g={logg_rounded}, [Fe/H]={met_str}")
    print(f"    URL: {url}")
    print(f"    Speichere als: {filepath}")
    
    # Erstelle Verzeichnis
    os.makedirs(model_dir, exist_ok=True)
    
    # Download
    temp_file = filepath + ".temp"
    
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(temp_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        
        os.rename(temp_file, filepath)
        print(f"  Download erfolgreich!")
        return filepath
        
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise Exception(
            f"Download von PHOENIX fehlgeschlagen:\n"
            f"  URL: {url}\n"
            f"  Fehler: {e}\n"
            f"  Hinweis: PHOENIX Grid: 2300-12000K, log g=0-6, [Fe/H]=-4 to +1"
        )


def download_oats_kurucz_model(teff, logg, kurucz_dir="kurucz_models",
                              metallicity=0.0, vturb=2.0,
                              base_url="https://wwwuser.oats.inaf.it/fiorella.castelli/grids/"):
    """
    Lädt ein Castelli & Kurucz OATS-Modell herunter (hochaufgelöst).
    
    Parameters:
    -----------
    teff : float
        Effektivtemperatur in K
    logg : float
        Oberflächenschwere
    kurucz_dir : str
        Zielverzeichnis
    metallicity : float
        Metallizität [M/H] in dex (default: 0.0 solar)
    vturb : float
        Mikroturb in km/s (default: 2.0)
    base_url : str
        Basis-URL für OATS Kurucz-Modelle
        
    Returns:
    --------
    filepath : str
        Pfad zur heruntergeladenen Datei
        
    Notes:
    ------
    OATS Castelli & Kurucz Grid: https://wwwuser.oats.inaf.it/fiorella.castelli/grids.html
    Format: fp00t9500g40k2odfnew.dat
    - f = flux
    - p00 = [M/H]=0.0 (p=plus, m=minus), alternative: m05 = -0.5
    - t9500 = Teff=9500K
    - g40 = log g = 4.0
    - k2 = vturb = 2 km/s
    """
    # Runde Parameter auf verfügbare Grid-Punkte
    # OATS: Teff in 250K-Schritten (3500-6000K, dann variabler)
    if teff < 10000:
        teff_rounded = round(teff / 250) * 250
    elif teff < 13000:
        teff_rounded = round(teff / 500) * 500
    else:
        teff_rounded = round(teff / 1000) * 1000
    
    logg_rounded = round(logg / 0.5) * 0.5
    
    # Metallizität formatieren: 0.0 -> p00, -0.5 -> m05, +0.2 -> p02
    if metallicity >= 0:
        met_str = f"p{abs(int(metallicity*10)):02d}"
    else:
        met_str = f"m{abs(int(metallicity*10)):02d}"
    
    # Grid-Verzeichnis bestimmen (z.B. gridp00k2odfnew)
    vturb_int = int(vturb)
    grid_name = f"grid{met_str}k{vturb_int}odfnew"
    
    # Dateiname: fp00t9500g40k2odfnew.dat
    logg_str = f"{int(logg_rounded*10):02d}"  # 4.0 -> 40
    filename = f"f{met_str}t{teff_rounded}g{logg_str}k{vturb_int}odfnew.dat"
    
    # Vollständige URL
    url = f"{base_url}{grid_name}/{filename}"
    
    # Lokaler Dateiname
    local_filename = f"oats_ck_{met_str}_t{teff_rounded}_g{logg_rounded:.1f}_k{vturb_int}.dat"
    filepath = os.path.join(kurucz_dir, local_filename)
    
    print(f"  Download OATS Castelli & Kurucz:")
    print(f"    Teff={teff_rounded}K, log g={logg_rounded}, [M/H]={metallicity}, vturb={vturb_int}km/s")
    print(f"    URL: {url}")
    print(f"    Speichere als: {filepath}")
    
    # Erstelle Verzeichnis
    os.makedirs(kurucz_dir, exist_ok=True)
    
    # Download
    ssl_context = ssl._create_unverified_context()
    temp_file = filepath + ".temp"
    
    try:
        with urllib.request.urlopen(url, context=ssl_context, timeout=30) as response:
            with open(temp_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        
        os.rename(temp_file, filepath)
        print(f"  Download erfolgreich!")
        return filepath
        
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise Exception(
            f"Download von OATS fehlgeschlagen:\n"
            f"  URL: {url}\n"
            f"  Fehler: {e}\n"
            f"  Hinweis: Prüfe ob Parameter im Grid verfügbar sind:\n"
            f"           https://wwwuser.oats.inaf.it/fiorella.castelli/grids.html"
        )


def download_kurucz_model(teff, logg, kurucz_dir, base_url="https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/k93models/"):
    """
    Lädt Kurucz-Modell von STScI herunter.
    
    Parameters:
    -----------
    teff : float
        Effektivtemperatur in K
    logg : float
        Oberflächenschwere
    kurucz_dir : str
        Zielverzeichnis
    base_url : str
        Basis-URL für Kurucz-Modelle
        
    Returns:
    --------
    filepath : str
        Pfad zur heruntergeladenen Datei
    """
    # Kurucz-Namensschema bei STScI: kp00_9500.fits
    # Format: k{metallicity}_{logg}{teff}.fits
    # Beispiele: kp00_9500.fits, km01_7000.fits
    # p = plus (solar), m = minus, Zahl = dex
    # Die Teff wird auf 250K gerundet
    
    # Runde Teff auf 250K
    teff_rounded = round(teff / 250) * 250
    
    # Runde log g auf 0.5
    logg_rounded = round(logg / 0.5) * 0.5
    
    # STScI verwendet Format: k{met}_{logg}{teff}.fits
    # z.B. kp00_40009500.fits für log g=4.0, Teff=9500K, [Fe/H]=0.0
    # Das Format ist: {logg ohne Punkt}{teff mit leading zeros}
    # Metallizität: kp00 = solar, km01 = [Fe/H]=-0.1, etc.
    metallicity = "kp00"
    
    # STScI organisiert Dateien in Unterverzeichnissen: kp00/kp00_9500.fits
    possible_filenames = [
        f"{metallicity}/{metallicity}_{teff_rounded}.fits",      # kp00/kp00_9500.fits
        f"{metallicity}/{metallicity}_{teff_rounded:05d}.fits",  # kp00/kp00_09500.fits
    ]
    
    # Lokaler Dateiname (unkomprimiert)
    local_filename = f"kurucz_t{teff_rounded}g{logg_rounded:.1f}.fits"
    filepath = os.path.join(kurucz_dir, local_filename)
    
    print(f"  Speichere als: {filepath}")
    
    # Erstelle Verzeichnis falls nötig
    os.makedirs(kurucz_dir, exist_ok=True)
    
    # Versuche verschiedene URLs
    ssl_context = ssl._create_unverified_context()
    
    for filename in possible_filenames:
        url = base_url + filename
        print(f"  Versuche: {url}")
        
        temp_file = filepath + ".temp"
        try:
            # Versuche unkomprimiert
            with urllib.request.urlopen(url, context=ssl_context, timeout=10) as response:
                with open(temp_file, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            
            # Erfolgreich
            os.rename(temp_file, filepath)
            print(f"  Download erfolgreich!")
            return filepath
            
        except urllib.error.HTTPError as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            # Versuche komprimierte Version
            url_gz = url + ".gz"
            print(f"  Versuche komprimiert: {url_gz}")
            try:
                temp_gz = filepath + ".gz"
                with urllib.request.urlopen(url_gz, context=ssl_context, timeout=10) as response:
                    with open(temp_gz, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                
                # Entpacke
                with gzip.open(temp_gz, 'rb') as f_in:
                    with open(filepath, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                os.remove(temp_gz)
                print(f"  Download erfolgreich!")
                return filepath
                
            except Exception:
                if os.path.exists(temp_gz):
                    os.remove(temp_gz)
                continue
        except Exception:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            continue
    
    # Kein Download erfolgreich
    raise Exception(f"Keine passende Datei auf STScI gefunden. Versuche Dateinamen: {possible_filenames}")


def load_phoenix_spectrum(model_filepath, wave_filepath):
    """
    Lädt ein PHOENIX-Modellspektrum.
    
    Parameters:
    -----------
    model_filepath : str
        Pfad zur PHOENIX-Spektrum-FITS-Datei
    wave_filepath : str
        Pfad zur PHOENIX-Wellenlängen-FITS-Datei (WAVE_PHOENIX-ACES-AGSS-COND-2011.fits)
        
    Returns:
    --------
    wavelength : array
        Wellenlängen in Angström
    flux : array
        Flux (erg/s/cm²/Å)
    """
    # Lade Wellenlängen (in Angström)
    with fits.open(wave_filepath) as hdul:
        wavelength = hdul[0].data
    
    # Lade Flux
    with fits.open(model_filepath) as hdul:
        flux = hdul[0].data
    
    if len(wavelength) != len(flux):
        raise ValueError(f"Wavelength and flux arrays have different lengths: {len(wavelength)} vs {len(flux)}")
    
    return wavelength, flux


def load_oats_flux_file(filepath):
    """
    Liest ein OATS Castelli & Kurucz Flux-File (.dat).
    
    Format:
    -------
    FLUX    1     9.09        3.298047E+16   1.0000E-50   0.0000E+00   1.00000
           (index) (nm)        (frequency)    (Hnu)        (Hcont)      (ratio)
    
    Wellenlänge in nm (Spalte 3), Flux Hnu in Spalte 4
    Flambda = 4*Hnu*c/lambda^2
    
    Returns:
    --------
    wavelength : array
        Wellenlängen in Angström
    flux : array
        Flux Flambda (erg/s/cm²/Å/ster)
    """
    wavelength_nm = []
    hnu = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('FLUX'):
                parts = line.split()
                # Format: FLUX index wl_nm freq Hnu Hcont ratio
                # Manchmal fehlt der Index am Ende der Datei
                if len(parts) >= 5:
                    try:
                        # Versuche zweites Feld als Wellenlänge zu lesen
                        # Falls es ein Index ist (integer), ist die Wellenlänge in Feld 3
                        try:
                            int(parts[1])  # Test ob Index
                            wl_nm = float(parts[2])
                            h_nu = float(parts[3])
                        except ValueError:
                            # Kein Index, Wellenlänge direkt in Feld 2
                            wl_nm = float(parts[1])
                            h_nu = float(parts[2])
                        
                        # Filter: nur optischer/NIR Bereich (90nm - 20000nm = 900Å - 200000Å)
                        if 90 <= wl_nm <= 20000:
                            wavelength_nm.append(wl_nm)
                            hnu.append(h_nu)
                    except (ValueError, IndexError):
                        continue
    
    wavelength_nm = np.array(wavelength_nm)
    hnu = np.array(hnu)
    
    # Konvertiere zu Angström
    wavelength = wavelength_nm * 10.0
    
    # Konvertiere Hnu zu Flambda
    # Flambda = 4*Hnu*c/lambda^2
    # c = 2.99792458e10 cm/s
    # lambda in cm
    c = 2.99792458e10  # cm/s
    wavelength_cm = wavelength * 1e-8  # Å -> cm
    flambda = 4.0 * hnu * c / (wavelength_cm**2)
    
    return wavelength, flambda


def load_phoenix_model(teff, logg, model_dir="models", metallicity=0.0):
    """
    Lädt ein PHOENIX-Modellspektrum für gegebene Parameter.
    Lädt Modell automatisch herunter, falls nicht vorhanden.
    
    Parameters:
    -----------
    teff : float
        Effektivtemperatur in K (2300-12000K)
    logg : float
        Oberflächenschwere (log g in cgs, 0.0-6.0)
    model_dir : str
        Verzeichnis mit Modellen
    metallicity : float
        Metallizität [Fe/H] in dex (default: 0.0 solar, Range: -4.0 to +1.0)
        
    Returns:
    --------
    wavelength : array
        Wellenlängen in Angström (500-55000 Å)
    flux : array
        Modell-Flux (erg/s/cm²/Å)
        
    Notes:
    ------
    PHOENIX ACES-AGSS-COND-2011 (sehr hochaufgelöst, ~1.6 Mio Punkte):
    https://phoenix.astro.physik.uni-goettingen.de/
    
    Grid:
    - Teff: 2300-7000K (100K steps), 7000-12000K (200K steps)
    - log g: 0.0-6.0 (0.5 steps)
    - [Fe/H]: -4.0 to +1.0
    """
    logg_rounded = round(logg / 0.5) * 0.5
    
    # Prüfe ob Teff im Grid-Bereich liegt
    if not (2300 <= teff <= 12000):
        raise ValueError(
            f"Teff={teff}K außerhalb des PHOENIX-Grid (2300-12000K).\n"
            f"Für Teff>12000K sind keine hochaufgelösten Modelle verfügbar."
        )
    
    # PHOENIX Rundung
    if teff <= 7000:
        teff_phoenix = round(teff / 100) * 100
    else:
        teff_phoenix = round(teff / 200) * 200
    
    # Metallizität formatieren
    if metallicity >= 0:
        met_str = f"+{abs(metallicity):.1f}" if metallicity > 0 else "-0.0"
    else:
        met_str = f"{metallicity:.1f}"
    
    phoenix_filename = f"phoenix_t{teff_phoenix}_g{logg_rounded:.1f}_z{met_str}.fits"
    phoenix_filepath = os.path.join(model_dir, phoenix_filename)
    
    # Wellenlängen-Datei (wird einmal heruntergeladen und wiederverwendet)
    wave_filename = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    wave_filepath = os.path.join(model_dir, wave_filename)
    wave_url = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    
    # Erstelle Verzeichnis
    os.makedirs(model_dir, exist_ok=True)
    
    # Prüfe ob Spektrum lokal vorhanden
    if os.path.exists(phoenix_filepath):
        # Lade Wellenlängen-Datei falls nötig
        if not os.path.exists(wave_filepath):
            print(f"Lade PHOENIX Wellenlängen-Datei...")
            try:
                with urllib.request.urlopen(wave_url, timeout=60) as response:
                    with open(wave_filepath, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                print(f"  Wellenlängen-Datei heruntergeladen (12 MB).")
            except Exception as e:
                raise Exception(f"Wellenlängen-Download fehlgeschlagen: {e}")
        
        print(f"Lade PHOENIX ACES: {phoenix_filepath}")
        print(f"  Teff = {teff_phoenix} K, log g = {logg_rounded}, [Fe/H] = {met_str}")
        wavelength, flux = load_phoenix_spectrum(phoenix_filepath, wave_filepath)
        print(f"  Wellenlängenbereich: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Å")
        print(f"  {len(wavelength)} Datenpunkte (sehr hochaufgelöst)")
        return wavelength, flux
    
    # Nicht lokal -> versuche Download
    print(f"PHOENIX-Modell nicht lokal gefunden.")
    print(f"Versuche Download von PHOENIX...")
    
    # Lade Wellenlängen-Datei falls nötig
    if not os.path.exists(wave_filepath):
        print(f"  Lade Wellenlängen-Datei...")
        try:
            with urllib.request.urlopen(wave_url, timeout=60) as response:
                with open(wave_filepath, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            print(f"  Wellenlängen-Datei heruntergeladen (12 MB).")
        except Exception as e:
            raise Exception(f"Wellenlängen-Download fehlgeschlagen: {e}")
    
    # Download Spektrum
    phoenix_filepath = download_phoenix_model(teff_phoenix, logg_rounded, model_dir, metallicity)
    wavelength, flux = load_phoenix_spectrum(phoenix_filepath, wave_filepath)
    print(f"  Wellenlängenbereich: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Å")
    print(f"  {len(wavelength)} Datenpunkte (sehr hochaufgelöst)")
    return wavelength, flux


def convolve_to_resolution(wavelength, flux, fwhm_angstrom):
    """
    Faltet Spektrum mit Gauß-Kernel, um Auflösung zu simulieren.
    
    Parameters:
    -----------
    wavelength : array
        Wellenlängen in Angström
    flux : array
        Flux-Array
    fwhm_angstrom : float
        FWHM der instrumentellen Auflösung in Angström
        
    Returns:
    --------
    flux_convolved : array
        Gefalteter Flux
    """
    # Berechne mittleren Pixel-Abstand
    dwl = np.median(np.diff(wavelength))
    
    # Konvertiere FWHM zu sigma in Pixel
    # FWHM = 2.355 * sigma
    sigma_angstrom = fwhm_angstrom / 2.355
    sigma_pixel = sigma_angstrom / dwl
    
    print(f"Falte Modell auf Auflösung:")
    print(f"  FWHM = {fwhm_angstrom:.2f} Å")
    print(f"  Sigma = {sigma_angstrom:.2f} Å = {sigma_pixel:.2f} Pixel")
    
    flux_convolved = gaussian_filter1d(flux, sigma=sigma_pixel, mode='nearest')
    
    return flux_convolved


def normalize_continuum(wavelength, flux, window=100):
    """
    Normalisiert Spektrum auf Kontinuum = 1.
    
    Parameters:
    -----------
    wavelength : array
        Wellenlängen
    flux : array
        Flux
    window : int
        Fensterbreite für Kontinuumsschätzung (in Pixel)
        
    Returns:
    --------
    flux_normalized : array
        Normalisierter Flux
    continuum : array
        Geschätztes Kontinuum
    """
    # Glätte stark für Kontinuum
    continuum = gaussian_filter1d(flux, sigma=window)
    flux_normalized = flux / (continuum + 1e-10)
    
    return flux_normalized, continuum


def cross_correlate_spectra(wavelength_obs, flux_obs, wavelength_model, flux_model,
                           max_shift_aa=50.0, oversample=10, wl_range=None):
    """
    Bestimmt Wellenlängenverschiebung zwischen Beobachtung und Modell mittels Kreuzkorrelation
    
    Parameters:
    -----------
    wavelength_obs : array
        Wellenlängen des beobachteten Spektrums [Å]
    flux_obs : array
        Flux des beobachteten Spektrums
    wavelength_model : array
        Wellenlängen des Modell-Spektrums [Å]
    flux_model : array
        Flux des Modell-Spektrums
    max_shift_aa : float
        Maximale Verschiebung in Ångström (Standard: 50 Å)
    oversample : int
        Oversampling-Faktor für höhere Genauigkeit (Standard: 10)
    wl_range : tuple or None
        Wellenlängenbereich für Kreuzkorrelation (wl_min, wl_max) in Å
        Falls None, wird der gesamte überlappende Bereich verwendet
    
    Returns:
    --------
    shift_aa : float
        Bestimmte Wellenlängenverschiebung in Ångström
        Positiv: Beobachtung ist rotverschoben gegenüber Modell
    wavelength_corrected : array
        Korrigierte Wellenlängen des beobachteten Spektrums
    correlation : array
        Kreuzkorrelationsfunktion (für Plot)
    lags_aa : array
        Lags in Ångström (für Plot)
    """
    print("\n  → Kreuzkorrelation: Bestimme Wellenlängenverschiebung...")
    
    # Bestimme gemeinsamen Wellenlängenbereich
    if wl_range is not None:
        wl_min, wl_max = wl_range
        print(f"    Verwende vorgegebenen Bereich: {wl_min:.1f} - {wl_max:.1f} Å")
    else:
        wl_min = max(wavelength_obs.min(), wavelength_model.min())
        wl_max = min(wavelength_obs.max(), wavelength_model.max())
    
    # Berücksichtige max_shift
    wl_min += max_shift_aa
    wl_max -= max_shift_aa
    
    if wl_max <= wl_min:
        print("    ⚠ Warnung: Wellenlängenbereiche überlappen nicht ausreichend")
        return 0.0, wavelength_obs, None, None
    
    # Erstelle gemeinsames hochaufgelöstes Grid
    disp_obs = np.median(np.diff(wavelength_obs))
    disp_model = np.median(np.diff(wavelength_model))
    disp_common = min(disp_obs, disp_model) / oversample
    
    wl_common = np.arange(wl_min, wl_max, disp_common)
    
    # Interpoliere beide Spektren auf gemeinsames Grid
    interp_obs = interp1d(wavelength_obs, flux_obs, kind='linear', 
                         bounds_error=False, fill_value=0.0)
    interp_model = interp1d(wavelength_model, flux_model, kind='linear',
                         bounds_error=False, fill_value=0.0)
    
    flux_obs_common = interp_obs(wl_common)
    flux_model_common = interp_model(wl_common)
    
    # Normalisiere Spektren (wichtig für Kreuzkorrelation)
    flux_obs_norm = (flux_obs_common - np.mean(flux_obs_common)) / np.std(flux_obs_common)
    flux_model_norm = (flux_model_common - np.mean(flux_model_common)) / np.std(flux_model_common)
    
    # Berechne Kreuzkorrelation
    correlation = correlate(flux_obs_norm, flux_model_norm, mode='same', method='fft')
    
    # Lags in Pixel
    lags_pixels = np.arange(len(correlation)) - len(correlation) // 2
    
    # Konvertiere zu Wellenlängen-Shift
    lags_aa = lags_pixels * disp_common
    
    # Begrenze auf erlaubten Bereich
    mask = np.abs(lags_aa) <= max_shift_aa
    correlation_limited = correlation.copy()
    correlation_limited[~mask] = -np.inf
    
    # Finde Maximum der Kreuzkorrelation
    max_idx = np.argmax(correlation_limited)
    shift_aa = lags_aa[max_idx]
    
    # Korrelationspeak-Qualität
    corr_max = correlation[max_idx]
    corr_noise = np.std(correlation[mask])
    snr = corr_max / corr_noise if corr_noise > 0 else 0
    
    print(f"    Verschiebung: {shift_aa:+.2f} Å (SNR: {snr:.1f})")
    
    # Warne bei niedriger Qualität
    if snr < 3.0:
        print(f"    ⚠ Warnung: Niedrige Korrelations-SNR ({snr:.1f})")
        print(f"    → Wellenlängenverschiebung könnte unzuverlässig sein")
    
    # Korrigiere Wellenlängen
    wavelength_corrected = wavelength_obs - shift_aa
    
    print(f"    Wellenlängen korrigiert: {wavelength_obs[0]:.1f} → {wavelength_corrected[0]:.1f} Å")
    
    return shift_aa, wavelength_corrected, correlation, lags_aa


# Vordefinierte Linien-Sets
LINE_SETS = {
    'balmer': [
        (6562.79, 'Hα'),
        (4861.33, 'Hβ'),
        (4340.47, 'Hγ'),
    ],
    'ca_hk': [
        (3933.66, 'Ca II K'),
        (3968.47, 'Ca II H'),
    ],
    'na_d': [
        (5889.95, 'Na I D1'),
        (5895.92, 'Na I D2'),
    ],
    'mg_triplet': [
        (5167.32, 'Mg I b1'),
        (5172.68, 'Mg I b2'),
        (5183.60, 'Mg I b3'),
    ],
    'all_hydrogen': [
        (6562.79, 'Hα'),
        (4861.33, 'Hβ'),
        (4340.47, 'Hγ'),
        (4101.74, 'Hδ'),
        (3970.07, 'Hε'),
    ],
}


def extract_line_region(wavelength, flux, line_center, width=20):
    """
    Extrahiert Region um Spektrallinie.
    
    Parameters:
    -----------
    wavelength : array
        Wellenlängen
    flux : array
        Flux
    line_center : float
        Zentrale Wellenlänge der Linie
    width : float
        Breite der Region in Angström (±width/2)
        
    Returns:
    --------
    wl_region : array
        Wellenlängen in Region
    flux_region : array
        Flux in Region
    """
    mask = (wavelength >= line_center - width/2) & (wavelength <= line_center + width/2)
    
    if not np.any(mask):
        raise ValueError(f"Keine Daten im Bereich {line_center - width/2:.1f} - {line_center + width/2:.1f} Å")
    
    return wavelength[mask], flux[mask]


def plot_multi_line_comparison(obs_wl, obs_flux, model_wl, model_flux, lines_list,
                              teff, logg, work_dir, width=20, 
                              cross_correlate=True, max_shift=50.0, xcorr_width=None,
                              line_set_name=None):
    """
    Erstellt Grid-Plot für mehrere Spektrallinien gleichzeitig
    
    Parameters:
    -----------
    obs_wl, obs_flux : arrays
        Beobachtetes Spektrum (noch NICHT korrigiert)
    model_wl, model_flux : arrays
        Modellspektrum
    lines_list : list of tuples
        Liste von (wavelength, label) Tupeln
    teff, logg : float
        Stellare Parameter
    work_dir : str
        Arbeitsordner
    width : float
        Plot-Breite in Angström pro Linie
    cross_correlate : bool
        Ob Kreuzkorrelation durchgeführt werden soll
    max_shift : float
        Maximale Verschiebung für Kreuzkorrelations-Plot
    xcorr_width : float or None
        Breite des Intervalls für Kreuzkorrelation
    line_set_name : str or None
        Name des Linien-Sets für Dateinamen
    """
    n_lines = len(lines_list)
    
    # Bestimme Grid-Layout - alle Linien nebeneinander in einer Zeile
    ncols = n_lines
    nrows = 1
    
    # Führe Kreuzkorrelation für jede Linie einzeln durch
    shifts = []
    correlations = []
    lags_list = []
    
    if cross_correlate:
        print(f"  Führe Kreuzkorrelation für {n_lines} Linien durch...")
        for line_center, line_label in lines_list:
            # Berechne Wellenlängenbereich für diese Linie
            # Verwende größeren Standardbereich wenn xcorr_width nicht angegeben
            if xcorr_width:
                half_width = xcorr_width / 2.0
            else:
                # Standardmäßig: ±200 Å um Linie für Multi-Line (breiter als width für Plot)
                half_width = max(200.0, width * 2.0)
            
            xcorr_range = (line_center - half_width, line_center + half_width)
            print(f"    {line_label}: Korrelationsbereich {line_center - half_width:.1f} - {line_center + half_width:.1f} Å (±{half_width:.0f} Å)")
            
            try:
                shift, obs_wl_temp, corr, lags = cross_correlate_spectra(
                    obs_wl, obs_flux,
                    model_wl, model_flux,
                    max_shift_aa=max_shift,
                    wl_range=xcorr_range
                )
                shifts.append(shift)
                correlations.append(corr)
                lags_list.append(lags)
                # Debug: Zeige ob Korrelation None ist
                if corr is None:
                    print(f"    {line_label}: ⚠ Keine Korrelation berechnet (Bereich zu klein?)")
            except Exception as e:
                print(f"    {line_label}: Kreuzkorrelation fehlgeschlagen - {e}")
                import traceback
                traceback.print_exc()
                shifts.append(0.0)
                correlations.append(None)
                lags_list.append(None)
        
        # Berechne mittlere Verschiebung (nur von erfolgreichen Korrelationen)
        valid_shifts = [s for s in shifts if s != 0.0]
        mean_shift = np.mean(valid_shifts) if valid_shifts else 0.0
        print(f"  Mittlere Verschiebung: {mean_shift:+.2f} Å")
        
        # Wende Verschiebung auf Beobachtung an (subtrahieren, um zu korrigieren!)
        obs_wl = obs_wl - mean_shift
    else:
        mean_shift = 0.0
    
    # Interpoliere Modell auf (korrigiertes) Beobachtungs-Grid
    print(f"  Interpoliere Modell auf korrigiertes Beobachtungs-Grid...")
    interp_func = interp1d(model_wl, model_flux, kind='linear',
                          bounds_error=False, fill_value=np.nan)
    model_flux_interp = interp_func(obs_wl)
    
    # Erstelle Figure - alle Linien nebeneinander
    if cross_correlate and any(c is not None for c in correlations):
        # Mit Kreuzkorrelations-Subplots
        print(f"  Erstelle Multi-Line Plot mit {n_lines} Linien + Kreuzkorrelationen")
        fig = plt.figure(figsize=(5*ncols, 8))
        gs = fig.add_gridspec(3, ncols, height_ratios=[4, 2, 0.5], 
                             hspace=0.5, wspace=0.25, top=0.86, bottom=0.07)
    else:
        print(f"  Erstelle Multi-Line Plot mit {n_lines} Linien")
        fig = plt.figure(figsize=(5*ncols, 4))
        gs = fig.add_gridspec(1, ncols, hspace=0.35, wspace=0.25, top=0.85, bottom=0.12)
    
    # Plotte jede Linie
    for idx, (line_center, line_label) in enumerate(lines_list):
        col = idx
        ax = fig.add_subplot(gs[0, col])
        
        try:
            # Region extrahieren (beide auf demselben korrigierten Wellenlängen-Grid)
            obs_wl_region, obs_flux_region = extract_line_region(obs_wl, obs_flux, line_center, width)
            model_wl_region, model_flux_region = extract_line_region(obs_wl, model_flux_interp, line_center, width)
            
            # Skalierung
            scale_factor = np.median(obs_flux_region) / np.median(model_flux_region)
            
            # Plot (beide mit denselben Wellenlängen)
            ax.plot(obs_wl_region, obs_flux_region, 'b-', linewidth=1.0, 
                   label='Beobachtung', alpha=0.8)
            ax.plot(model_wl_region, model_flux_region * scale_factor, 'r-', 
                   linewidth=0.8, label='PHOENIX', alpha=0.7)
            ax.axvline(line_center, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
            
            # Zeige individuelle Verschiebung im Titel
            title = f'{line_label} ({line_center:.2f} Å)'
            if cross_correlate and idx < len(shifts):
                title += f'\nΔλ = {shifts[idx]:+.2f} Å'
            
            # Labels
            ax.set_xlabel('Wellenlänge [Å]', fontsize=9)
            ax.set_ylabel('Flux [erg/s/cm²/Å]', fontsize=9)
            ax.set_title(title, fontsize=9, fontweight='bold', pad=8)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
        except ValueError as e:
            # Keine Daten in diesem Bereich
            ax.text(0.5, 0.5, f'{line_label}\n{line_center:.2f} Å\n\nKeine Daten', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{line_label} ({line_center:.2f} Å)', fontsize=9, fontweight='bold', pad=8)
        
        # Kreuzkorrelations-Subplot für diese Linie
        if cross_correlate and idx < len(correlations) and correlations[idx] is not None:
            ax_corr = fig.add_subplot(gs[1, col])
            ax_corr.plot(lags_list[idx], correlations[idx], 'b-', lw=1.0)
            ax_corr.axvline(x=shifts[idx], color='red', linestyle='--', lw=1.5)
            ax_corr.axvline(x=0, color='gray', linestyle=':', lw=0.8, alpha=0.5)
            ax_corr.set_xlabel('Δλ [Å]', fontsize=8)
            ax_corr.set_ylabel('Korrelation', fontsize=8)
            ax_corr.grid(True, alpha=0.3)
            ax_corr.set_xlim(-max_shift, max_shift)
            ax_corr.tick_params(labelsize=7)
    
    # Haupt-Titel (weit oben, um Überlappung mit Subplot-Titeln zu vermeiden)
    title_text = f'PHOENIX Multi-Line Vergleich (Teff={teff}K, log g={logg})'
    if cross_correlate and mean_shift != 0.0:
        title_text += f' — Mittlere Δλ: {mean_shift:+.2f} Å'
    fig.suptitle(title_text, fontsize=11, fontweight='bold', y=0.985)
    
    # Speichern
    out_dir = os.path.join(work_dir, "out")
    if line_set_name:
        output_png = os.path.join(out_dir, f"line_comparison_{line_set_name}.png")
    else:
        output_png = os.path.join(out_dir, f"multi_line_comparison.png")
    
    try:
        fig.savefig(output_png, dpi=200, bbox_inches='tight')
        print(f"\nMulti-Line Vergleichsplot gespeichert: {output_png}")
    except Exception as e:
        print(f"Fehler beim Speichern des Plots: {e}")
    
    plt.show()


def plot_comparison(obs_wl, obs_flux, model_wl, model_flux, line_center, 
                   teff, logg, work_dir, width=20, 
                   shift_aa=0.0, correlation=None, lags_aa=None, max_shift=50.0):
    """
    Erstellt Vergleichsplot von Beobachtung und Modell.
    
    Parameters:
    -----------
    obs_wl, obs_flux : arrays
        Beobachtetes Spektrum
    model_wl, model_flux : arrays
        Modellspektrum
    line_center : float
        Zentrale Wellenlänge
    teff, logg : float
        Stellare Parameter
    work_dir : str
        Arbeitsordner
    width : float
        Plot-Breite in Angström
    shift_aa : float
        Wellenlängenverschiebung aus Kreuzkorrelation
    correlation : array or None
        Kreuzkorrelationsfunktion (für Subplot)
    lags_aa : array or None
        Lags in Ångström (für Subplot)
    max_shift : float
        Maximale Verschiebung für Kreuzkorrelations-Plot
    """
    # Erstelle Figure mit Subplots wenn Kreuzkorrelation vorhanden
    if correlation is not None and lags_aa is not None:
        print(f"  Erstelle Plot mit Kreuzkorrelations-Subplot ({len(correlation)} Punkte)")
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        ax_spec = fig.add_subplot(gs[0])
        ax_corr = fig.add_subplot(gs[1])
    else:
        print(f"  Erstelle Plot ohne Kreuzkorrelations-Subplot (correlation={correlation is not None}, lags={lags_aa is not None})")
        fig, ax_spec = plt.subplots(1, 1, figsize=(12, 6))
        ax_corr = None
    
    # Region extrahieren
    obs_wl_region, obs_flux_region = extract_line_region(obs_wl, obs_flux, line_center, width)
    model_wl_region, model_flux_region = extract_line_region(model_wl, model_flux, line_center, width)
    
    # Berechne Skalierungsfaktor: Median des Modells auf Median der Beobachtung
    scale_factor = np.median(obs_flux_region) / np.median(model_flux_region)
    
    # Plot: Absolute Flüsse
    ax_spec.plot(obs_wl_region, obs_flux_region, 'b-', linewidth=1.2, label='Beobachtung', alpha=0.8)
    model_label = f'PHOENIX (Teff={teff}K, log g={logg})'
    ax_spec.plot(model_wl_region, model_flux_region * scale_factor, 'r-', 
            linewidth=1.0, label=model_label, alpha=0.7)
    
    ax_spec.axvline(line_center, color='gray', linestyle=':', alpha=0.5, label=f'Linie @ {line_center:.2f} Å')
    ax_spec.set_xlabel('Wellenlänge [Å]', fontsize=11)
    ax_spec.set_ylabel('Flux [erg/s/cm²/Å]', fontsize=11)
    
    title_text = f'Spektrallinien-Vergleich: {line_center:.2f} Å'
    if shift_aa != 0.0:
        title_text += f'\n(Wellenlängen-Korrektur: {shift_aa:+.2f} Å)'
    ax_spec.set_title(title_text, fontsize=13, fontweight='bold')
    ax_spec.legend(fontsize=9, loc='best')
    ax_spec.grid(True, alpha=0.3)
    
    # Plot Kreuzkorrelationsfunktion
    if ax_corr is not None and correlation is not None and lags_aa is not None:
        ax_corr.plot(lags_aa, correlation, 'b-', lw=1.0)
        ax_corr.axvline(x=shift_aa, color='red', linestyle='--', lw=1.5, 
                       label=f'Best match: {shift_aa:+.2f} Å')
        ax_corr.axvline(x=0, color='gray', linestyle=':', lw=0.8, alpha=0.5)
        ax_corr.set_xlabel('Wellenlängenverschiebung [Å]', fontsize=11)
        ax_corr.set_ylabel('Kreuzkorrelation', fontsize=10)
        ax_corr.grid(True, alpha=0.3)
        ax_corr.legend(loc='upper right', fontsize=8)
        ax_corr.set_xlim(-max_shift, max_shift)
    
    # Speichern
    out_dir = os.path.join(work_dir, "out")
    output_png = os.path.join(out_dir, f"line_comparison_{line_center:.1f}A.png")
    
    try:
        fig.tight_layout()
        fig.savefig(output_png, dpi=200)
        print(f"\nVergleichsplot gespeichert: {output_png}")
    except Exception as e:
        print(f"Fehler beim Speichern des Plots: {e}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Vergleich von beobachtetem Spektrum mit Kurucz-Modell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # H-alpha Linie bei Vega (mit Kreuzkorrelation, Standard)
  python spec_lines.py vega --line 6562.8 --teff 9600 --logg 3.9 --resolution 2.5
  
  # Ca II K Linie
  python spec_lines.py star1 --line 3933.7 --teff 5800 --logg 4.4 --resolution 1.5
  
  # Ohne Kreuzkorrelation
  python spec_lines.py vega --line 6562.8 --teff 9600 --logg 3.9 --resolution 2.5 --no-cross-correlate
  
  # Mit größerer erlaubter Verschiebung
  python spec_lines.py vega --line 6562.8 --teff 9600 --logg 3.9 --resolution 2.5 --max-shift 100
  
  # Mit größerem Intervall für Kreuzkorrelation (±1500 Å um Linie)
  python spec_lines.py vega --line 6562.8 --teff 9600 --logg 3.9 --resolution 2.5 --xcorr-width 3000
  
  # Multi-Line Plots: alle Balmer-Linien gleichzeitig
  python spec_lines.py vega --lines balmer --teff 9600 --logg 3.9 --resolution 2.5
  
  # Ca II H & K Doublet
  python spec_lines.py star1 --lines ca_hk --teff 5800 --logg 4.4 --resolution 1.5
        """
    )
    
    parser.add_argument('work_dir', help='Arbeitsordner mit in/ und out/ Unterordnern')
    
    # Linien-Auswahl (entweder --line ODER --lines)
    line_group = parser.add_mutually_exclusive_group(required=True)
    line_group.add_argument('--line', type=float,
                       help='Zentrale Wellenlänge einer einzelnen Linie (Å)')
    line_group.add_argument('--lines', choices=list(LINE_SETS.keys()),
                       help=f'Vordefiniertes Linien-Set: {", ".join(LINE_SETS.keys())}')
    
    parser.add_argument('--teff', type=float, required=True,
                       help='Effektivtemperatur des Sterns (K)')
    parser.add_argument('--logg', type=float, required=True,
                       help='Oberflächenschwere log g (cgs)')
    parser.add_argument('--resolution', type=float, required=True,
                       help='Instrumentelle Auflösung FWHM (Å)')
    parser.add_argument('--width', type=float, default=100.0,
                       help='Breite der Plotregion um Linie (Å, default: 100)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Verzeichnis mit Atmosphärenmodellen (default: models)')
    parser.add_argument('--metallicity', type=float, default=0.0,
                       help='Metallizität [Fe/H] in dex (default: 0.0 solar)')
    parser.add_argument('--cross-correlate', dest='cross_correlate', action='store_true',
                       default=True,
                       help='Korrigiere Wellenlängen mittels Kreuzkorrelation (Standard)')
    parser.add_argument('--no-cross-correlate', dest='cross_correlate', action='store_false',
                       help='Deaktiviere Wellenlängen-Kreuzkorrelation')
    parser.add_argument('--max-shift', type=float, default=50.0,
                       help='Maximale Wellenlängenverschiebung in Å (Standard: 50)')
    parser.add_argument('--xcorr-width', type=float, 
                       help='Breite des Intervalls um Linienmitte für Kreuzkorrelation in Å (z.B. --xcorr-width 3000 für ±1500 Å)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("spec_lines.py - Spektrallinien-Vergleich mit PHOENIX-Modellen")
    print("=" * 60)
    print(f"Arbeitsordner: {args.work_dir}")
    
    # Bestimme Linien-Liste
    if args.lines:
        lines_list = LINE_SETS[args.lines]
        print(f"Linien-Set: {args.lines} ({len(lines_list)} Linien)")
        for wl, label in lines_list:
            print(f"  - {label}: {wl:.2f} Å")
        # Zentrale Linie für Kreuzkorrelation: Mittelwert
        center_line = np.mean([wl for wl, _ in lines_list])
    else:
        lines_list = [(args.line, f'{args.line:.2f} Å')]
        center_line = args.line
        print(f"Linie: {args.line:.2f} Å")
    
    print(f"Stellare Parameter: Teff = {args.teff} K, log g = {args.logg}")
    print(f"Metallizität: [Fe/H] = {args.metallicity}")
    print(f"Instrumentelle Auflösung: FWHM = {args.resolution:.2f} Å")
    print(f"Kreuzkorrelation: {'Aktiviert' if args.cross_correlate else 'Deaktiviert'}", end="")
    if args.cross_correlate:
        print(f" (max. ±{args.max_shift} Å)")
        if args.xcorr_width:
            half_width = args.xcorr_width / 2.0
            wl_min = center_line - half_width
            wl_max = center_line + half_width
            print(f"  Wellenlängenbereich: {wl_min:.1f} - {wl_max:.1f} Å (±{half_width:.1f} Å um {center_line:.1f} Å)")
    else:
        print()
    print("=" * 60)
    
    # Prüfe Arbeitsordner
    if not os.path.exists(args.work_dir):
        print(f"Fehler: Arbeitsordner '{args.work_dir}' existiert nicht!")
        sys.exit(1)
    
    out_dir = os.path.join(args.work_dir, "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    try:
        # Lade beobachtetes Spektrum
        print("\n1. Lade beobachtetes Spektrum...")
        obs_wl, obs_flux = load_calibrated_spectrum(args.work_dir)
        
        # Lade PHOENIX-Modell
        print("\n2. Lade PHOENIX-Modell...")
        model_wl, model_flux = load_phoenix_model(args.teff, args.logg, args.model_dir,
                                                   args.metallicity)
        
        # Falte Modell auf instrumentelle Auflösung
        print("\n3. Falte Modell auf instrumentelle Auflösung...")
        model_flux_convolved = convolve_to_resolution(model_wl, model_flux, args.resolution)
        
        # Unterscheide: Single-line oder Multi-line Modus
        if args.lines:
            # Multi-line Modus: Plot-Funktion macht Kreuzkorrelation für jede Linie
            print("\n4. Erstelle Multi-Line Vergleichsplot...")
            plot_multi_line_comparison(
                obs_wl, obs_flux, model_wl, model_flux_convolved,
                lines_list, args.teff, args.logg, args.work_dir, args.width,
                cross_correlate=args.cross_correlate,
                max_shift=args.max_shift,
                xcorr_width=args.xcorr_width,
                line_set_name=args.lines
            )
        else:
            # Single-line Modus: Kreuzkorrelation einmal, dann interpolieren
            shift_aa = 0.0
            correlation = None
            lags_aa = None
            obs_wl_corrected = obs_wl
            
            if args.cross_correlate:
                print("\n4. Führe Kreuzkorrelation durch...")
                # Verwende gefaltetes Modell für Kreuzkorrelation
                # Berechne Wellenlängenbereich aus Linienmitte und Breite
                xcorr_range = None
                if args.xcorr_width:
                    half_width = args.xcorr_width / 2.0
                    xcorr_range = (center_line - half_width, center_line + half_width)
                
                shift_aa, obs_wl_corrected, correlation, lags_aa = cross_correlate_spectra(
                    obs_wl, obs_flux,
                    model_wl, model_flux_convolved,
                    max_shift_aa=args.max_shift,
                    wl_range=xcorr_range
                )
                obs_wl = obs_wl_corrected
            
            # Interpoliere Modell auf (korrigiertes) Beobachtungs-Grid
            step_num = 5 if args.cross_correlate else 4
            print(f"\n{step_num}. Interpoliere Modell auf Beobachtungs-Wellenlängen...")
            interp_func = interp1d(model_wl, model_flux_convolved, kind='linear',
                                  bounds_error=False, fill_value=np.nan)
            model_flux_interp = interp_func(obs_wl)
            
            # Erstelle Vergleichsplot
            step_num += 1
            print(f"\n{step_num}. Erstelle Vergleichsplot...")
            plot_comparison(obs_wl, obs_flux, obs_wl, model_flux_interp,
                           args.line, args.teff, args.logg, args.work_dir, args.width,
                           shift_aa=shift_aa, correlation=correlation, lags_aa=lags_aa,
                           max_shift=args.max_shift)
        
        print("\n" + "=" * 60)
        print("Fertig!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nFehler: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
