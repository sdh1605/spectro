#!/usr/bin/env python3
"""
Plot flux-calibrated spectrum

Usage:
    python3 spec_plot_fluxcal.py WORK_DIR [options]
    # or point directly to a FITS file
    python3 spec_plot_fluxcal.py /path/to/science_spectrum_flux_calibrated.fits [options]

Defaults:
- Expects file at WORK_DIR/out/science_spectrum_flux_calibrated.fits
- Automatically saves a PNG next to the FITS (default name: <fits_basename>_plot.png)
- Shows an interactive plot; use --no-show to skip the window

Options:
    --file PATH         Explicit FITS file path to plot
    --save PATH         Save plot to this PNG path (overrides default)
    --no-save           Do not save a PNG (disables default auto-save)
    --show              Force showing the window
    --no-show           Do not show window (useful in batch)
    --lines             Enable automatic spectral line detection and annotation
    --smooth SIGMA      Gaussian smoothing sigma in pixels (float, default 0)
    --xlim XMIN XMAX    X-range (Å)
    --ylim YMIN YMAX    Y-range (erg/s/cm^2/Å)
    --title TEXT        Custom title

Output:
- A single panel: Flux vs Wavelength (Å), labeled with units
- Title includes work_dir and SPECTYPE from FITS header when available
- With --lines: detects and labels absorption/emission lines
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple, List, Optional


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


def match_line_label(wav: float, line_list: List[Tuple[float, str]] = LINE_LIST, tol: float = 3.0) -> Optional[str]:
    """Return element label if wavelength matches a known line within tolerance"""
    diffs = np.abs(np.array([lw for lw, _ in line_list]) - wav)
    idx = int(np.argmin(diffs))
    if diffs[idx] <= tol:
        return line_list[idx][1]
    return None


def detect_spectral_lines(wavelength: np.ndarray, 
                         flux: np.ndarray,
                         window_length: int = 501,
                         min_prominence: float = 0.05,
                         distance: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect absorption lines in spectrum
    
    Returns:
        peaks: indices of detected line centers
        continuum: estimated continuum flux
        norm_depth: normalized line depth (continuum - flux) / continuum
    """
    # Fit continuum using Savitzky-Golay filter
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(flux):
        window_length = len(flux) if len(flux) % 2 == 1 else len(flux) - 1
    if window_length < 5:
        window_length = 5
        
    continuum = savgol_filter(flux, window_length, 3)
    
    # Calculate normalized depth
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_depth = (continuum - flux) / np.maximum(continuum, np.median(continuum)*0.1)
    norm_depth = np.nan_to_num(norm_depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Smooth normalized depth to preserve line features
    smooth_window = min(31, len(norm_depth) if len(norm_depth) % 2 == 1 else len(norm_depth) - 1)
    if smooth_window < 5:
        smooth_window = 5
    smoothed_depth = savgol_filter(norm_depth, smooth_window, 3)
    
    # Find absorption lines
    peaks, properties = find_peaks(smoothed_depth,
                                 prominence=min_prominence,
                                 distance=distance,
                                 width=2)
    
    # Filter using local SNR
    if len(peaks) > 0:
        local_noise = np.std(norm_depth - smoothed_depth)
        if local_noise > 0:
            snr = properties["prominences"] / local_noise
            peaks = peaks[snr > 3.0]
    
    return peaks, continuum, norm_depth


def resolve_input_path(work_or_file: str, explicit_file: str | None) -> tuple[str, str]:
    """Return (work_dir, fits_file) from user input.
    If explicit_file is provided, use it; otherwise, if work_or_file ends with .fits, use directly.
    Else assume work_or_file is a work directory containing out/science_spectrum_flux_calibrated.fits
    """
    if explicit_file:
        fits_path = os.path.abspath(explicit_file)
        work_dir = os.path.dirname(os.path.dirname(fits_path)) if os.path.basename(os.path.dirname(fits_path)) == 'out' else os.path.dirname(fits_path)
        return work_dir, fits_path
    if work_or_file.lower().endswith('.fits'):
        fits_path = os.path.abspath(work_or_file)
        work_dir = os.path.dirname(os.path.dirname(fits_path)) if os.path.basename(os.path.dirname(fits_path)) == 'out' else os.path.dirname(fits_path)
        return work_dir, fits_path
    work_dir = os.path.abspath(work_or_file)
    fits_path = os.path.join(work_dir, 'out', 'science_spectrum_flux_calibrated.fits')
    return work_dir, fits_path


def load_flux_calibrated_fits(path: str):
    """Load wavelength [Å] and flux [erg/s/cm^2/Å] from a flux-calibrated FITS.
    Supports the format written by spec_flux.py (binary table with columns WAVELENGTH, FLUX).
    Returns: wavelength, flux, meta(dict)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FITS nicht gefunden: {path}")

    with fits.open(path) as hdul:
        meta = {}
        # Primary header meta
        hdr0 = hdul[0].header if len(hdul) > 0 else None
        if hdr0 is not None:
            meta['SPECTYPE'] = hdr0.get('SPECTYPE')
            meta['METHOD'] = hdr0.get('METHOD')
            meta['FLUXCAL'] = hdr0.get('FLUXCAL')
        # Binary table expected in extension 1
        if len(hdul) > 1 and hasattr(hdul[1], 'data') and hdul[1].data is not None:
            data = hdul[1].data
            # Column names may vary in case; make robust
            cols = {c.lower(): c for c in data.columns.names}
            wl_col = cols.get('wavelength') or cols.get('lambda')
            fx_col = cols.get('flux')
            if not wl_col or not fx_col:
                raise KeyError(f"Erwarte Spalten 'WAVELENGTH' und 'FLUX', gefunden: {data.columns.names}")
            wavelength = np.array(data[wl_col], dtype=float)
            flux = np.array(data[fx_col], dtype=float)
            return wavelength, flux, meta
        # Fallback: 1-HDU image-like with WCS
        if len(hdul) == 1 and hdul[0].data is not None:
            arr = np.array(hdul[0].data, dtype=float)
            header = hdul[0].header
            if 'CRVAL1' in header and 'CDELT1' in header and 'NAXIS1' in header:
                crval1 = header['CRVAL1']
                cdelt1 = header['CDELT1']
                crpix1 = header.get('CRPIX1', 1)
                n = header['NAXIS1']
                wavelength = crval1 + (np.arange(n) - crpix1 + 1) * cdelt1
                flux = arr
                return wavelength, flux, meta
        raise ValueError("Unbekanntes FITS-Format. Erwarte Binär-Tabelle oder 1D-Image mit WCS.")


def make_title(work_dir: str, meta: dict, custom: str | None) -> str:
    if custom:
        return custom
    spectype = meta.get('SPECTYPE')
    method = meta.get('METHOD')
    left = os.path.basename(os.path.abspath(work_dir))
    right = 'Flux-Calibrated Spectrum'
    if spectype:
        right += f' ({spectype})'
    if method:
        src = 'Pickles Atlas' if str(method).upper() in ('PICKLES', 'MILES') else str(method)
        # Note: method in existing files might still say 'MILES' though we use Pickles
        right += f' — {src}'
    return f"{left} — {right}"


def main():
    p = argparse.ArgumentParser(description='Plot a flux-calibrated spectrum (FITS).')
    p.add_argument('work_or_file', help='Work directory or direct path to FITS')
    p.add_argument('--file', dest='file', help='Explicit FITS path to plot')
    p.add_argument('--save', dest='save', help='Save PNG to this path (overrides default)')
    p.add_argument('--no-save', dest='no_save', action='store_true', help='Disable default auto-save')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--show', dest='show', action='store_true', help='Show window')
    g.add_argument('--no-show', dest='show', action='store_false', help='Do not show window')
    p.set_defaults(show=None)
    p.add_argument('--smooth', type=float, default=0.0, help='Gaussian sigma in pixels')
    p.add_argument('--lines', action='store_true', help='Detect and annotate spectral lines')
    p.add_argument('--xlim', nargs=2, type=float, help='X-range Å: XMIN XMAX')
    p.add_argument('--ylim', nargs=2, type=float, help='Y-range erg/s/cm^2/Å: YMIN YMAX')
    p.add_argument('--title', help='Custom title')

    args = p.parse_args()

    work_dir, fits_path = resolve_input_path(args.work_or_file, args.file)
    if not os.path.exists(fits_path):
        sys.exit(f"✗ Datei nicht gefunden: {fits_path}")

    print('='*70)
    print('Flux-Calibrated Spectrum Plotter')
    print('='*70)
    print(f"Work dir: {work_dir}")
    print(f"FITS:     {fits_path}")

    wavelength, flux, meta = load_flux_calibrated_fits(fits_path)

    if args.smooth and args.smooth > 0:
        flux_plot = gaussian_filter1d(flux, sigma=float(args.smooth))
    else:
        flux_plot = flux

    # Detect spectral lines if requested
    peaks = []
    continuum = None
    if args.lines:
        print('\nDetecting spectral lines...')
        peaks, continuum, norm_depth = detect_spectral_lines(wavelength, flux_plot,
                                                             window_length=501,
                                                             min_prominence=0.03,
                                                             distance=15)
        print(f'  Found {len(peaks)} candidate lines')
        matched = 0
        for peak in peaks:
            wav = wavelength[peak]
            label = match_line_label(wav, tol=8.0)
            if label:
                matched += 1
        print(f'  Matched {matched} known lines')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wavelength, flux_plot, color='tab:green', lw=0.8, label='Flux (calibrated)')
    
    # Annotate detected lines
    if args.lines and len(peaks) > 0:
        ylim_current = ax.get_ylim()
        y_range = ylim_current[1] - ylim_current[0]
        # Place labels at 90% of plot height (below top)
        label_y = ylim_current[1] - 0.1 * y_range
        
        for peak in peaks:
            wav = wavelength[peak]
            label = match_line_label(wav, tol=8.0)
            if label:
                ax.axvline(x=wav, color='red', linestyle=':', alpha=0.6, lw=0.8)
                # Position label horizontally, slightly to the right of the line
                ax.annotate(label,
                           xy=(wav, label_y),
                           xytext=(3, 0),  # 3 points to the right
                           textcoords='offset points',
                           ha='left',
                           va='center',
                           fontsize=7,
                           rotation=0,  # horizontal text
                           alpha=0.8)
    
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('Flux [erg/s/cm²/Å]')
    ax.grid(True, alpha=0.3)
    if args.xlim:
        ax.set_xlim(args.xlim[0], args.xlim[1])
    else:
        ax.set_xlim(wavelength[0], wavelength[-1])
    if args.ylim:
        ax.set_ylim(args.ylim[0], args.ylim[1])

    title = make_title(work_dir, meta, args.title)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # Decide PNG output path (default auto-save unless --no-save)
    default_png_dir = os.path.dirname(fits_path)
    base = os.path.splitext(os.path.basename(fits_path))[0]
    default_png_name = f"{base}_plot.png"
    default_png_path = os.path.join(default_png_dir, default_png_name)

    png_path = None
    if not args.no_save:
        png_path = os.path.abspath(args.save) if args.save else default_png_path
        os.makedirs(os.path.dirname(png_path) or '.', exist_ok=True)
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"✓ PNG gespeichert: {png_path}")

    # Show logic: default to showing unless explicitly disabled
    show = True if args.show is None else args.show

    if show:
        print('Zeige Fenster... (Schließen zum Beenden)')
        plt.show()
    else:
        plt.close(fig)

    print('Fertig.')


if __name__ == '__main__':
    main()
