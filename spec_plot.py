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
from matplotlib.ticker import MultipleLocator
from astropy.io import fits
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
import os
import sys
import math
import re

# Parameter für Arbeitsordner
args = sys.argv[1:]
ylog_plot = False
raw_only_plot = False
auto_peaks = False
peaks_out_file = None
wave_min = None
wave_max = None
work_dir = None

i = 0
while i < len(args):
    a = args[i]
    if a == "--ylog":
        ylog_plot = True
        i += 1
    elif a == "--raw-only":
        raw_only_plot = True
        i += 1
    elif a == "--auto-peaks":
        auto_peaks = True
        i += 1
    elif a == "--peaks-out":
        if i + 1 >= len(args):
            print("Fehler: Nach --peaks-out fehlt ein Dateipfad.")
            sys.exit(1)
        peaks_out_file = args[i + 1]
        i += 2
    elif a == "--wmin":
        if i + 1 >= len(args):
            print("Fehler: Nach --wmin fehlt ein Wert.")
            sys.exit(1)
        try:
            wave_min = float(args[i + 1])
        except ValueError:
            print(f"Fehler: Ungültiger --wmin Wert: {args[i + 1]}")
            sys.exit(1)
        i += 2
    elif a == "--wmax":
        if i + 1 >= len(args):
            print("Fehler: Nach --wmax fehlt ein Wert.")
            sys.exit(1)
        try:
            wave_max = float(args[i + 1])
        except ValueError:
            print(f"Fehler: Ungültiger --wmax Wert: {args[i + 1]}")
            sys.exit(1)
        i += 2
    elif a.startswith("--"):
        print(f"Unbekannte Option: {a}")
        print("Verwendung: python spec_plot.py ARBEITSORDNER [--ylog] [--raw-only] [--auto-peaks] [--peaks-out DATEI] [--wmin A] [--wmax B]")
        sys.exit(1)
    else:
        if work_dir is not None:
            print("Fehler: Mehrere Arbeitsordner angegeben.")
            print("Verwendung: python spec_plot.py ARBEITSORDNER [--ylog] [--raw-only] [--auto-peaks] [--peaks-out DATEI] [--wmin A] [--wmax B]")
            sys.exit(1)
        work_dir = a
        i += 1

if work_dir is None:
    print("Verwendung: python spec_plot.py ARBEITSORDNER [--ylog] [--raw-only] [--auto-peaks] [--peaks-out DATEI] [--wmin A] [--wmax B]")
    sys.exit(1)

if wave_min is not None and wave_max is not None and wave_min >= wave_max:
    print(f"Fehler: --wmin ({wave_min}) muss kleiner als --wmax ({wave_max}) sein.")
    sys.exit(1)
out_dir = os.path.join(work_dir, "out")

# Prüfe ob der Ausgabeordner existiert
if not os.path.exists(out_dir):
    print(f"Fehler: Ausgabeordner '{out_dir}' existiert nicht!")
    sys.exit(1)

# Configuration
CALIBRATED_FITS_FILE = os.path.join(out_dir, "science_spectrum_calibrated.fits")
OUTPUT_PNG_FILE = os.path.join(out_dir, "science_spectrum_with_lines_improved.png")
if peaks_out_file is None:
    peaks_out_file = os.path.join(out_dir, "detected_peaks.txt")
elif not os.path.isabs(peaks_out_file):
    peaks_out_file = os.path.join(work_dir, peaks_out_file)

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
                         distance: int = 15,
                         use_log_flux: bool = False,
                         snr_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect absorption lines in spectrum"""
    flux_work = np.array(flux, dtype=float)
    if use_log_flux:
        positive = flux_work > 0
        if np.any(positive):
            # Für hohe Dynamikbereiche liefert Log-Flux robustere Linienkontraste.
            floor = np.percentile(flux_work[positive], 1.0)
            floor = max(floor, np.min(flux_work[positive]))
            flux_work = np.log10(np.clip(flux_work, floor, None))
            print("Peak-Detektion im Log-Flux-Modus aktiv (--ylog).")
        else:
            print("Warnung: Keine positiven Flux-Werte für Log-Detektion; wechsle auf lineare Detektion.")

    # Fit continuum using Savitzky-Golay filter
    if window_length % 2 == 0:
        window_length += 1
    continuum = savgol_filter(flux_work, window_length, 3)
    
    # Calculate normalized depth with better handling of edge effects
    with np.errstate(divide='ignore', invalid='ignore'):
        scale_ref = np.maximum(np.abs(continuum), np.median(np.abs(continuum))*0.1 + 1e-12)
        norm_depth = (continuum - flux_work) / scale_ref
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
    if local_noise <= 0 or not np.isfinite(local_noise):
        local_noise = 1e-12
    snr = properties["prominences"] / local_noise
    peaks = peaks[snr > snr_threshold]
    
    print(f"Total peaks found: {len(peaks)}")
    print(f"Wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} Å")
    
    return peaks, continuum, norm_depth

def detect_auto_flux_peaks(wavelength: np.ndarray,
                           flux: np.ndarray,
                           use_log_flux: bool = False,
                           wmin: Optional[float] = None,
                           wmax: Optional[float] = None,
                           distance: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Detect broad rounded maxima in the shown spectrum region."""
    n = len(flux)
    if n < 5:
        return np.array([], dtype=int), np.zeros(n, dtype=float)

    wmin_eff = float(np.nanmin(wavelength)) if wmin is None else float(wmin)
    wmax_eff = float(np.nanmax(wavelength)) if wmax is None else float(wmax)
    region_mask = (wavelength >= wmin_eff) & (wavelength <= wmax_eff)

    signal = np.array(flux, dtype=float)
    if use_log_flux:
        positive = signal > 0
        if np.any(positive):
            floor = np.percentile(signal[positive], 1.0)
            signal = np.log10(np.clip(signal, max(floor, 1e-12), None))

    def _odd_at_least(v: int) -> int:
        vv = max(5, int(v))
        if vv % 2 == 0:
            vv += 1
        return vv

    smooth_win = min(_odd_at_least(n // 40), _odd_at_least(n - 1))
    base_win = min(_odd_at_least(n // 10), _odd_at_least(n - 1))

    smooth = savgol_filter(signal, smooth_win, 3)
    baseline = medfilt(smooth, kernel_size=base_win)
    profile = smooth - baseline

    noise = np.std(profile[region_mask]) if np.any(region_mask) else np.std(profile)
    if not np.isfinite(noise) or noise <= 0:
        noise = 1e-12

    prominence = max(0.5 * noise, 1e-6)
    width_guess = max(2, int(n / 400))

    peaks, props = find_peaks(
        profile,
        prominence=prominence,
        distance=max(3, int(distance)),
        width=width_guess,
    )

    # Lokale Nachsuche: spaltet breite/enge Doppellinien in benachbarte Maxima auf.
    split_candidates = set(int(p) for p in peaks)
    for p in peaks:
        lo = max(0, int(p) - 12)
        hi = min(n - 1, int(p) + 12)
        if hi - lo < 5:
            continue
        sub = profile[lo:hi + 1]
        sub_noise = np.std(sub - savgol_filter(sub, 5 if len(sub) >= 5 else 3, 2 if len(sub) >= 5 else 1))
        if not np.isfinite(sub_noise) or sub_noise <= 0:
            sub_noise = noise
        sub_prom = max(0.25 * sub_noise, 1e-7)
        sub_peaks, _ = find_peaks(sub, prominence=sub_prom, distance=2, width=1)
        for sp in sub_peaks:
            split_candidates.add(lo + int(sp))

    if split_candidates:
        peaks = np.array(sorted(split_candidates), dtype=int)

        # Dedupliziere sehr nahe Kandidaten, behalte jeweils den höheren.
        dedup = []
        for p in peaks:
            if not dedup:
                dedup.append(int(p))
                continue
            if p - dedup[-1] <= 1:
                if profile[p] > profile[dedup[-1]]:
                    dedup[-1] = int(p)
            else:
                dedup.append(int(p))
        peaks = np.array(dedup, dtype=int)

    if peaks.size:
        peaks = peaks[(wavelength[peaks] >= wmin_eff) & (wavelength[peaks] <= wmax_eff)]

    print(f"Auto-Peaks gefunden (Maxima): {len(peaks)} im Bereich [{wmin_eff:.1f}, {wmax_eff:.1f}] Å")
    return peaks.astype(int), profile

def refine_peak_wavelengths(wavelength: np.ndarray,
                            profile: np.ndarray,
                            peaks: np.ndarray,
                            fit_radius: int = 3) -> np.ndarray:
    """Refine peak centers for broad rounded lines using local quadratic fit."""
    n = len(wavelength)
    idx = np.arange(n, dtype=float)
    refined = np.zeros(len(peaks), dtype=float)

    for k, p in enumerate(peaks):
        p = int(p)
        lo = max(0, p - fit_radius)
        hi = min(n - 1, p + fit_radius)
        if hi - lo < 2:
            refined[k] = float(wavelength[p])
            continue

        local_x = np.arange(lo, hi + 1)
        local_y = profile[lo:hi + 1]
        p_local = int(local_x[np.argmax(local_y)])

        qlo = max(lo, p_local - 1)
        qhi = min(hi, p_local + 1)
        qx = idx[qlo:qhi + 1]
        qy = profile[qlo:qhi + 1]

        if qx.size >= 3:
            try:
                a, b, c = np.polyfit(qx, qy, 2)
                if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-12:
                    xv = -b / (2.0 * a)
                    if qlo <= xv <= qhi:
                        refined[k] = float(np.interp(xv, idx, wavelength))
                        continue
            except Exception:
                pass

        refined[k] = float(wavelength[p_local])

    return refined

def export_peak_wavelengths(peaks: np.ndarray,
                            peak_wavelengths: np.ndarray,
                            wavelength: np.ndarray,
                            flux: np.ndarray,
                            continuum: np.ndarray,
                            norm_depth: np.ndarray,
                            wmin: float,
                            wmax: float,
                            output_file: str) -> int:
    records = []
    for peak, wav_refined in zip(peaks, peak_wavelengths):
        wav = float(wav_refined)
        if wav < wmin or wav > wmax:
            continue
        line_id = match_line_label(wav, tol=8.0) or "-"
        peak_idx = int(np.clip(peak, 0, len(wavelength) - 1))
        depth = float(norm_depth[peak_idx])
        flux_val = float(flux[peak_idx])
        cont_val = float(continuum[peak_idx])
        records.append((wav, flux_val, cont_val, depth, line_id))

    records.sort(key=lambda r: r[0])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("# Automatisch bestimmte Peak-Wellenlaengen\n")
        f.write("# columns: wavelength_A flux continuum norm_depth match\n")
        for wav, flux_val, cont_val, depth, line_id in records:
            f.write(f"{wav:.6f}\t{flux_val:.8g}\t{cont_val:.8g}\t{depth:.6f}\t{line_id}\n")

    print(f"Peak-Liste gespeichert: {output_file}")
    print(f"Anzahl Peaks im Bereich [{wmin:.2f}, {wmax:.2f}] Å: {len(records)}")
    for wav, _, _, depth, line_id in records:
        if line_id == "-":
            print(f"  {wav:.3f} Å (depth={depth:.3f})")
        else:
            print(f"  {wav:.3f} Å (depth={depth:.3f}, match={line_id})")

    return len(records)

def plot_spectrum_with_lines(wavelength: np.ndarray,
                           flux: np.ndarray, 
                           peaks: np.ndarray,
                           peak_wavelengths: np.ndarray,
                           continuum: np.ndarray,
                           norm_depth: np.ndarray,
                           match_tol: float = 8.0,    # erhöhte Toleranz auf 8 Angstrom
                           show_unmatched: bool = False,
                           use_ylog: bool = False,
                           raw_only: bool = False,
                           show_auto_peak_lines: bool = False,
                           wave_min: Optional[float] = None,
                           wave_max: Optional[float] = None):
    """Plot spectrum and mark detected lines"""

    wmin = float(np.nanmin(wavelength)) if wave_min is None else float(wave_min)
    wmax = float(np.nanmax(wavelength)) if wave_max is None else float(wave_max)
    if wmin >= wmax:
        raise ValueError(f"Ungültiger Wellenlängenbereich: [{wmin}, {wmax}]")
    visible = (wavelength >= wmin) & (wavelength <= wmax)
    if not np.any(visible):
        raise ValueError(f"Kein Datenpunkt im Wellenlängenbereich [{wmin}, {wmax}].")

    def apply_wavelength_grid(ax):
        span = max(wmax - wmin, 1.0)
        raw_step = span / 10.0
        base = 10 ** math.floor(math.log10(raw_step))
        major_step = 10 * base
        for m in (1, 2, 2.5, 5, 10):
            step = m * base
            if step >= raw_step:
                major_step = step
                break
        ax.set_xlim(wmin, wmax)
        ax.xaxis.set_major_locator(MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(MultipleLocator(max(major_step / 5.0, 0.1)))
        ax.grid(True, which='major', alpha=0.5)
        ax.grid(True, which='minor', alpha=0.2)
    
    # Create figure
    if raw_only:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.subplots_adjust(hspace=0.3)
    
    # Plot original spectrum
    if use_ylog:
        positive_mask = flux > 0
        if np.any(positive_mask):
            ax1.plot(wavelength, np.where(positive_mask, flux, np.nan), 'b-', label='Original Spectrum')
            ax1.set_yscale('log')
        else:
            print("Warnung: Keine positiven Flux-Werte für --ylog gefunden. Es wird linear geplottet.")
            ax1.plot(wavelength, flux, 'b-', label='Original Spectrum')
    else:
        ax1.plot(wavelength, flux, 'b-', label='Original Spectrum')
    if not raw_only:
        ax1.plot(wavelength, continuum, 'r-', alpha=0.5, label='Continuum')
    ax1.set_ylabel('Flux (log)' if use_ylog else 'Flux')
    apply_wavelength_grid(ax1)
    ax1.legend()
    
    # Plot normalized spectrum
    if not raw_only and ax2 is not None:
        ax2.plot(wavelength, flux/continuum, 'b-', label='Normalized Spectrum')
        ax2.axhline(y=1.0, color='r', linestyle='-', alpha=0.5, label='Continuum Level')
        ax2.set_ylabel('Normalized Flux')
        ax2.set_xlabel('Wavelength (Å)')
        apply_wavelength_grid(ax2)
        ax2.legend()
    else:
        ax1.set_xlabel('Wavelength (Å)')

    if show_auto_peak_lines:
        axis_items = [ax1] if raw_only or ax2 is None else [ax1, ax2]
        for wav in peak_wavelengths:
            if wav < wmin or wav > wmax:
                continue
            for ax in axis_items:
                ax.axvline(x=wav, color='#ff00aa', linestyle='--', alpha=0.9, lw=1.8, zorder=4)
    
    # Mark peaks and annotate in all plots
    plotted_lines = 0
    for peak, wav in zip(peaks, peak_wavelengths):
        if wav < wmin or wav > wmax:
            continue
        peak_idx = int(np.clip(peak, 0, len(wavelength) - 1))
        flu = flux[peak_idx]
        flu_norm = flux[peak_idx]/continuum[peak_idx] if continuum[peak_idx] != 0 else np.nan
        
        element = match_line_label(wav, tol=match_tol)
        if element is None and not show_unmatched:
            continue
        plotted_lines += 1
            
        label = element if element is not None else f"{wav:.1f}Å"
        
        # Draw vertical lines and annotations in both subplots
        axis_items = [(ax1, flu)] if raw_only or ax2 is None else [(ax1, flu), (ax2, flu_norm)]
        for ax, y in axis_items:
            ax.axvline(x=wav, color='r', linestyle='--', alpha=0.5)
            ax.annotate(label,
                       xy=(wav, ax.get_ylim()[1]),
                       xytext=(0, 2),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       fontsize=8,
                       rotation=45)
    
    if raw_only:
        ax1.set_title('Unnormalized Spectrum (Selected Wavelength Range)')
    else:
        ax1.set_title('Spectrum Analysis with Identified Lines')
    
    # Save plot
    plt.savefig(OUTPUT_PNG_FILE, dpi=200, bbox_inches='tight')
    
    # Show plot instead of closing
    plt.show()
    
    # Print found lines
    print(f"Found {plotted_lines} lines in displayed range:")
    for peak, wav in zip(peaks, peak_wavelengths):
        if wav < wmin or wav > wmax:
            continue
        peak_idx = int(np.clip(peak, 0, len(wavelength) - 1))
        depth = norm_depth[peak_idx]
        element = match_line_label(wav, tol=match_tol)
        if element:
            print(f"  {element} at {wav:.1f} Å (depth: {depth:.3f})")

if __name__ == "__main__":
    # Load spectrum
    pixel, wavelength, flux = load_calibrated_spectrum(CALIBRATED_FITS_FILE)

    wmin = float(np.nanmin(wavelength)) if wave_min is None else float(wave_min)
    wmax = float(np.nanmax(wavelength)) if wave_max is None else float(wave_max)
    if wmin >= wmax:
        raise ValueError(f"Ungültiger Wellenlängenbereich: [{wmin}, {wmax}]")
    
    # Detect lines with improved parameters
    peaks_abs, continuum, norm_depth = detect_spectral_lines(wavelength, flux,
                                                              window_length=501,
                                                              min_prominence=0.015 if ylog_plot else 0.03,
                                                              distance=15,
                                                              use_log_flux=ylog_plot,
                                                              snr_threshold=2.0 if ylog_plot else 3.0)

    if auto_peaks:
        peaks, peak_profile = detect_auto_flux_peaks(
            wavelength=wavelength,
            flux=flux,
            use_log_flux=ylog_plot,
            wmin=wmin,
            wmax=wmax,
            distance=6,
        )
        if len(peaks) == 0:
            print("Hinweis: Keine Auto-Peaks gefunden, fallback auf Linien-Dips.")
            peaks = peaks_abs
            peak_profile = norm_depth
    else:
        peaks = peaks_abs
        peak_profile = norm_depth

    peak_wavelengths = refine_peak_wavelengths(wavelength, peak_profile, peaks, fit_radius=2)

    if auto_peaks:
        export_peak_wavelengths(
            peaks=peaks,
            peak_wavelengths=peak_wavelengths,
            wavelength=wavelength,
            flux=flux,
            continuum=continuum,
            norm_depth=norm_depth,
            wmin=wmin,
            wmax=wmax,
            output_file=peaks_out_file,
        )
    
    # Plot and save with increased matching tolerance
    plot_spectrum_with_lines(wavelength, flux, peaks, peak_wavelengths, continuum, norm_depth,
                           match_tol=8.0,            # erhöhte Toleranz
                           show_unmatched=False,     # nur identifizierte Linien zeigen
                           use_ylog=ylog_plot,
                           raw_only=raw_only_plot,
                           show_auto_peak_lines=auto_peaks,
                           wave_min=wave_min,
                           wave_max=wave_max)