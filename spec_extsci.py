import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter1d
import os
import sys

# -------------------------
# Parameter
# -------------------------
if len(sys.argv) != 2:
    print("Verwendung: python spec_extsci.py ARBEITSORDNER")
    sys.exit(1)

work_dir = sys.argv[1]
in_dir = os.path.join(work_dir, "in")
out_dir = os.path.join(work_dir, "out")

# Prüfe ob die Ordner existieren
if not os.path.exists(in_dir):
    print(f"Fehler: Eingabeordner '{in_dir}' existiert nicht!")
    sys.exit(1)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fits_datei = os.path.join(in_dir, "science_spectrum.fits")
spektrum_half_width_init = 10
sky_offset_init = 40
sky_width_init = 10
snr_threshold = 5.0

# -------------------------
# FITS laden und x-spiegeln
# -------------------------
hdul = fits.open(fits_datei)
daten = hdul[0].data.copy()
header = hdul[0].header
hdul.close()

if daten is None or daten.ndim != 2:
    raise ValueError("Dieses Skript erwartet eine 2D-FITS-Datei.")

daten = np.fliplr(daten)
ny, nx = daten.shape

# -------------------------
# Hilfsfunktionen
# -------------------------
def clamp(val, a, b):
    return max(a, min(b, int(val)))

def clamp_range(r, ny):
    return (max(0, int(r[0])), min(ny, int(r[1])))

def compute_trace(daten, y_center, half_width, sky_offset, sky_width, snr_thresh):
    ny, nx = daten.shape
    x = np.arange(nx)

    # Sky-Fenster definieren
    sky1 = clamp_range((y_center - sky_offset - sky_width, y_center - sky_offset), ny)
    sky2 = clamp_range((y_center + sky_offset, y_center + sky_offset + sky_width), ny)

    # Sky-Median pro Spalte
    sky_parts = []
    if sky1[1] > sky1[0]:
        sky_parts.append(daten[sky1[0]:sky1[1], :])
    if sky2[1] > sky2[0]:
        sky_parts.append(daten[sky2[0]:sky2[1], :])
    sky_median_per_col = np.median(np.vstack(sky_parts), axis=0) if sky_parts else np.median(daten, axis=0)

    # Zentroiden berechnen
    win_rad = max(3 * half_width, half_width + 5)
    y_min_w = clamp(y_center - win_rad, 0, ny-1)
    y_max_w = clamp(y_center + win_rad, 0, ny-1)
    y_inds = np.arange(y_min_w, y_max_w)

    centroids = np.full(nx, np.nan)
    valid = np.zeros(nx, dtype=bool)

    for i in range(nx):
        col = daten[y_min_w:y_max_w, i].astype(float) - sky_median_per_col[i]
        mask_pos = col > 0
        flux = np.sum(col[mask_pos]) if np.any(mask_pos) else 0.0
        noise = np.std(col) if np.std(col) > 0 else 1.0
        snr = flux / (noise * np.sqrt(col.size))
        if flux > 0 and snr >= snr_thresh and np.any(mask_pos):
            y_pos = y_inds[mask_pos]
            f = col[mask_pos]
            centroids[i] = np.sum(y_pos * f) / np.sum(f)
            valid[i] = True

    # Interpolation oder fallback auf y_center
    valid_idx = np.where(valid)[0]
    y_fit = np.full(nx, y_center)
    if len(valid_idx) >= 2:
        y_fit_interp = np.interp(x, x[valid_idx], centroids[valid_idx])
        y_fit = gaussian_filter1d(y_fit_interp, sigma=3)

    return {
        "x": x,
        "y_fit": np.clip(y_fit, 0, ny-1),
        "sky1": sky1,
        "sky2": sky2,
        "sky_median_per_col": sky_median_per_col
    }

def extract_1d_spectrum(daten, y_fit, half_width, sky_median_per_col):
    ny, nx = daten.shape
    flux = np.zeros(nx)
    for i in range(nx):
        yi = y_fit[i] if np.isfinite(y_fit[i]) else ny//2
        ylo = clamp(np.floor(yi - half_width), 0, ny-1)
        yhi = clamp(np.ceil(yi + half_width), 0, ny-1)
        if yhi < ylo:
            ylo, yhi = yhi, ylo
        ap_pixels = daten[ylo:yhi+1, i].astype(float)
        npix = ap_pixels.size
        flux[i] = np.nansum(ap_pixels) - sky_median_per_col[i] * npix
    return flux

# -------------------------
# Grobe Lokalisierung
# -------------------------
profil = np.sum(daten, axis=1)
profil_smooth = gaussian_filter1d(profil, sigma=5)
y_peak = int(np.argmax(profil_smooth))

trace_info = compute_trace(daten, y_peak, spektrum_half_width_init, sky_offset_init, sky_width_init, snr_threshold)
spektrum_1d = extract_1d_spectrum(daten, trace_info["y_fit"], spektrum_half_width_init, trace_info["sky_median_per_col"])

# -------------------------
# Plot 2D + 1D + Slider
# -------------------------
fig, (ax_img, ax_spec) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios':[2,1]})
plt.subplots_adjust(left=0.1, right=0.98, bottom=0.30, hspace=0.3)

vmin_init = np.percentile(daten,5)
vmax_init = np.percentile(daten,95)

img = ax_img.imshow(daten, cmap='inferno', origin='lower', aspect='auto', vmin=vmin_init, vmax=vmax_init)
ax_img.set_title("2D-Spektrum (x gespiegelt)")
ax_img.set_xlabel("Dispersion (x)")
ax_img.set_ylabel("Spalt (y)")
fig.colorbar(img, ax=ax_img, pad=0.01)

# Trace und Sky markieren
trace_line, = ax_img.plot(trace_info["x"], trace_info["y_fit"], color='yellow', lw=1.5)
ap_upper = ax_img.plot(trace_info["x"], trace_info["y_fit"]+spektrum_half_width_init, color='lime', lw=0.8)[0]
ap_lower = ax_img.plot(trace_info["x"], trace_info["y_fit"]-spektrum_half_width_init, color='lime', lw=0.8)[0]
sky1_span = ax_img.axhspan(*trace_info["sky1"], color='cyan', alpha=0.2)
sky2_span = ax_img.axhspan(*trace_info["sky2"], color='cyan', alpha=0.2)

# 1D Spektrum
spec_line, = ax_spec.plot(trace_info["x"], spektrum_1d, color='blue')
ax_spec.set_title("Extrahiertes 1D-Spektrum (Sky-subtrahiert)")
ax_spec.set_xlabel("Dispersion (x)")
ax_spec.set_ylabel("Flux")
ax_spec.grid(True)

# Slider-Achsen
ax_vmin = plt.axes([0.15,0.12,0.65,0.03])
ax_vmax = plt.axes([0.15,0.08,0.65,0.03])
ax_hw   = plt.axes([0.15,0.04,0.35,0.03])
ax_offset = plt.axes([0.55,0.04,0.25,0.03])
ax_skyw = plt.axes([0.55,0.00,0.25,0.03])

s_vmin = Slider(ax_vmin,'Min', float(np.min(daten)), float(np.max(daten)), valinit=vmin_init)
s_vmax = Slider(ax_vmax,'Max', float(np.min(daten)), float(np.max(daten)), valinit=vmax_init)
s_hw   = Slider(ax_hw,'HalfW', 1, max(1, ny//8), valinit=spektrum_half_width_init, valstep=1)
s_offset = Slider(ax_offset,'SkyOffset', 1, ny//2, valinit=sky_offset_init, valstep=1)
s_skyw = Slider(ax_skyw, 'SkyWidth', 1, max(2, ny//6), valinit=sky_width_init, valstep=1)

# -------------------------
# Update-Funktion
# -------------------------
def update(val):
    global sky1_span, sky2_span, spektrum_1d

    vmin = s_vmin.val; vmax = s_vmax.val
    if vmax>vmin: img.set_clim(vmin=vmin,vmax=vmax)

    half_w = int(s_hw.val)
    sky_off = int(s_offset.val)
    sky_w = int(s_skyw.val)

    new_trace = compute_trace(daten, y_peak, half_w, sky_off, sky_w, snr_threshold)
    new_flux = extract_1d_spectrum(daten, new_trace["y_fit"], half_w, new_trace["sky_median_per_col"])

    trace_line.set_ydata(new_trace["y_fit"])
    ap_upper.set_ydata(new_trace["y_fit"]+half_w)
    ap_lower.set_ydata(new_trace["y_fit"]-half_w)

    # Sky Spans aktualisieren
    try:
        sky1_span.remove()
        sky2_span.remove()
    except Exception:
        pass

    sky1_span = ax_img.axhspan(*new_trace["sky1"], color='cyan', alpha=0.2)
    sky2_span = ax_img.axhspan(*new_trace["sky2"], color='cyan', alpha=0.2)

    spektrum_1d = new_flux
    spec_line.set_ydata(new_flux)
    ax_spec.relim(); ax_spec.autoscale_view()

    fig.canvas.draw_idle()

s_vmin.on_changed(update)
s_vmax.on_changed(update)
s_hw.on_changed(update)
s_offset.on_changed(update)
s_skyw.on_changed(update)

plt.show()

# 1D-Spektrum als FITS-Tabelle speichern
t = Table([np.arange(nx), spektrum_1d], names=('PIXEL', 'FLUX'))
output_file = os.path.join(out_dir, "science_spectrum_1d.fits")
t.write(output_file, overwrite=True)
print(f"Extrahiertes 1D-Spektrum gespeichert: {output_file}")
