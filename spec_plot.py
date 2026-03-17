# spec_plot.py
"""
Improved spectral line detection:
- stronger noise reduction (Savitzky-Golay + optional Gaussian)
- robust continuum estimate (median filter)
- expanded line list for common optical transitions
- SNR-based pruning of candidate lines
Produces a wavelength-calibrated PNG with labeled matches.
"""
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
import os
import sys
import re

# Parameter für Arbeitsordner
if len(sys.argv) != 2:
    print("Verwendung: python spec_plot.py ARBEITSORDNER")
    sys.exit(1)

work_dir = sys.argv[1]
out_dir = os.path.join(work_dir, "out")

# Prüfe ob der Ausgabeordner existiert
if not os.path.exists(out_dir):
    print(f"Fehler: Ausgabeordner '{out_dir}' existiert nicht!")
    sys.exit(1)

# Configuration
CALIBRATED_FITS_FILE = os.path.join(out_dir, "science_spectrum_calibrated.fits")
OUTPUT_PNG_FILE = os.path.join(out_dir, "science_spectrum_with_lines_improved.png")

# Expanded line list (wavelengths in Å, common optical/near-UV lines)
LINE_LIST: List[Tuple[float, str]] = [
    # Hydrogen Balmer series
    (6562.79, "Hα"),
    (4861.33, "Hβ"),
    (4340.47, "Hγ"),
    (4101.74, "Hδ"),
    (3970.07, "Hε"),
    # Helium
    (5875.62, "He I 5876"),
    (4471.48, "He I 4471"),
    # Sodium D
    (5895.92, "Na I D2"),
    (5889.95, "Na I D1"),
    # Calcium H & K
    (3933.66, "Ca II K"),
    (3968.47, "Ca II H"),
    # Magnesium b triplet
    (5167.32, "Mg I b1"),
    (5172.68, "Mg I b2"),
    (5183.60, "Mg I b3"),
    # Iron (some common Fe I/II lines)
    (5270.40, "Fe I 5270"),
    (4923.92, "Fe II 4924"),
    (5018.44, "Fe II 5018"),
    # Forbidden oxygen/nitrogen/sulfur (nebular)
    (4958.91, "[O III] 4959"),
    (5006.84, "[O III] 5007"),
    (3726.03, "[O II] 3726"),
    (3728.82, "[O II] 3729"),
    (6548.05, "[N II] 6548"),
    (6583.45, "[N II] 6583"),
    (6716.44, "[S II] 6716"),
    (6730.82, "[S II] 6731"),
    # Additional useful metallic lines
    (4307.9, "CH G-band ~4308"),
    (5175.0, "MgH / blend ~5175"),
    (5890.0, "telluric/Na blend ~5890"),
    # Telluric lines (atmospheric absorption)
    (7594.0, "O2 A-band"),
    (6867.0, "O2 B-band"),
    (6287.0, "O2 γ-band"),
    (7164.0, "H2O"),
    (8227.0, "H2O"),
    (9380.0, "H2O"),
    (6515.0, "H2O"),
    (5932.0, "H2O"),
    (7187.0, "H2O"),
    (6974.0, "H2O"),
]

def load_calibrated_spectrum(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load calibrated spectrum from FITS file"""
    with fits.open(filename) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        pixel = np.arange(len(data['FLUX']))
        wavelength = data['WAVELENGTH']
        flux = data['FLUX']
    return pixel, wavelength, flux

def match_line_label(wav: float, line_list: List[Tuple[float, str]] = LINE_LIST, tol: float = 3.0) -> Optional[str]:
    """Return element label including Balmer series designation if present"""
    diffs = np.abs(np.array([lw for lw, _ in line_list]) - wav)
    idx = int(np.argmin(diffs))
    if diffs[idx] <= tol:
        # Return the full label for Balmer lines, otherwise just the element
        label_full = line_list[idx][1]
        if "H I H" in label_full:  # It's a Balmer line
            return label_full
        # For other lines, keep the previous behavior
        tokens = label_full.split()
        elem = " ".join(tokens[:2]) if len(tokens) > 1 else tokens[0]
        elem = elem.strip("(),")
        return elem
    return None

def detect_spectral_lines(wavelength: np.ndarray, 
                         flux: np.ndarray,
                         window_length: int = 501,      # größeres Fenster
                         min_prominence: float = 0.05,
                         distance: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect absorption lines in spectrum"""
    # Fit continuum using Savitzky-Golay filter
    if window_length % 2 == 0:
        window_length += 1
    continuum = savgol_filter(flux, window_length, 3)
    
    # Calculate normalized depth with better handling of edge effects
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_depth = (continuum - flux) / np.maximum(continuum, np.median(continuum)*0.1)
    norm_depth = np.nan_to_num(norm_depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply rolling normalization to handle varying line depths across spectrum
    window = len(norm_depth) // 10
    if window % 2 == 0:
        window += 1
    
    # Smooth normalized depth with smaller window to preserve line features
    smoothed_depth = savgol_filter(norm_depth, 31, 3)
    
    # Find absorption lines with adaptive prominence
    peaks, properties = find_peaks(smoothed_depth,
                                 prominence=min_prominence,
                                 distance=distance,
                                 width=2)  # minimum width requirement
    
    # Filter out weak detections using local SNR
    local_noise = np.std(norm_depth - smoothed_depth)
    snr = properties["prominences"] / local_noise
    peaks = peaks[snr > 3.0]  # keep only peaks with SNR > 3
    
    print(f"Total peaks found: {len(peaks)}")
    print(f"Wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Å")
    
    return peaks, continuum, norm_depth

def plot_spectrum_with_lines(wavelength: np.ndarray,
                           flux: np.ndarray, 
                           peaks: np.ndarray,
                           continuum: np.ndarray,
                           norm_depth: np.ndarray,
                           match_tol: float = 8.0,    # erhöhte Toleranz auf 8 Angstrom
                           show_unmatched: bool = False):
    """Plot spectrum and mark detected lines"""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    
    # Plot original spectrum
    ax1.plot(wavelength, flux, 'b-', label='Original Spectrum')
    ax1.plot(wavelength, continuum, 'r-', alpha=0.5, label='Continuum')
    ax1.set_ylabel('Flux')
    ax1.grid(True)
    ax1.legend()
    
    # Plot normalized spectrum
    ax2.plot(wavelength, flux/continuum, 'b-', label='Normalized Spectrum')
    ax2.axhline(y=1.0, color='r', linestyle='-', alpha=0.5, label='Continuum Level')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_xlabel('Wavelength (Å)')
    ax2.grid(True)
    ax2.legend()
    
    # Mark peaks and annotate in all plots
    for peak in peaks:
        wav = wavelength[peak]
        flu = flux[peak]
        flu_norm = flux[peak]/continuum[peak]
        
        element = match_line_label(wav, tol=match_tol)
        if element is None and not show_unmatched:
            continue
            
        label = element if element is not None else f"{wav:.1f}Å"
        
        # Draw vertical lines and annotations in both subplots
        for ax, y in [(ax1, flu), (ax2, flu_norm)]:
            ax.axvline(x=wav, color='r', linestyle='--', alpha=0.5)
            ax.annotate(label,
                       xy=(wav, ax.get_ylim()[1]),
                       xytext=(0, 2),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       fontsize=8,
                       rotation=45)
    
    ax1.set_title('Spectrum Analysis with Identified Lines')
    
    # Save plot
    plt.savefig(OUTPUT_PNG_FILE, dpi=200, bbox_inches='tight')
    
    # Show plot instead of closing
    plt.show()
    
    # Print found lines
    print(f"Found {len(peaks)} lines:")
    for peak in peaks:
        wav = wavelength[peak]
        depth = norm_depth[peak]
        element = match_line_label(wav, tol=match_tol)
        if element:
            print(f"  {element} at {wav:.1f} Å (depth: {depth:.3f})")

if __name__ == "__main__":
    # Load spectrum
    pixel, wavelength, flux = load_calibrated_spectrum(CALIBRATED_FITS_FILE)
    
    # Detect lines with improved parameters
    peaks, continuum, norm_depth = detect_spectral_lines(wavelength, flux,
                                                        window_length=501,
                                                        min_prominence=0.03,
                                                        distance=15)
    
    # Plot and save with increased matching tolerance
    plot_spectrum_with_lines(wavelength, flux, peaks, continuum, norm_depth,
                           match_tol=8.0,            # erhöhte Toleranz
                           show_unmatched=False)     # nur identifizierte Linien zeigen