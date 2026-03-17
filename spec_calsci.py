import os
import sys
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

# -------------------------
# Parameter
# -------------------------
if len(sys.argv) != 2:
    print("Verwendung: python spec_calsci.py ARBEITSORDNER")
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

# Dateien
science_1d_fits = os.path.join(out_dir, "science_spectrum_1d.fits")  # gespeichertes 1D-Spektrum
calib_2d_fits   = os.path.join(in_dir, "calibration_lamp.fits")    # Kalibrations-FITS
solution_fname_default = os.path.join(out_dir, "wavelength_solution.txt")

# -------------------------
# Hilfsfunktionen für Wellenlösungs-Datei
# -------------------------
def save_solution(coeffs, filename=solution_fname_default):
    """Speichere Polynom-Koeffizienten in eine Textdatei."""
    # Stelle sicher, dass der Zielordner existiert
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, "w") as f:
            f.write("# Wellenlöse-Polynom\n")
            f.write("# Format: eine Zeile 'deg N' und eine Zeile 'coeffs c0 c1 ...' (höchster Grad zuerst)\n")
            deg = len(coeffs) - 1
            f.write(f"deg {deg}\n")
            f.write("coeffs " + " ".join(f"{c:.12g}" for c in np.atleast_1d(coeffs)) + "\n")
        print(f"Wellenlösung gespeichert: {filename}")
    except Exception as e:
        print(f"Fehler beim Speichern der Wellenlösungs-Datei: {e}")

def load_solution(filename):
    """Lade Koeffizienten aus einer Textdatei. Erwartet eine Zeile mit 'coeffs' oder reine Zahlen."""
    try:
        with open(filename) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        # Suche Zeile mit 'coeffs'
        for ln in lines:
            if ln.lower().startswith("coeffs"):
                parts = ln.split()[1:]
                return np.array([float(p) for p in parts], dtype=float)
        # ansonsten: erste nicht-kommentar-zeile als Zahlenreihe interpretieren
        if lines:
            parts = lines[0].split()
            return np.array([float(p) for p in parts], dtype=float)
    except Exception as e:
        raise ValueError(f"Fehler beim Laden der Lösung '{filename}': {e}")
    raise ValueError(f"Keine Koeffizienten in Datei '{filename}' gefunden.")

# -------------------------
# 1D Science-Spektrum einlesen
# -------------------------
hdul_sci = fits.open(science_1d_fits)
data_sci_table = hdul_sci[1].data if len(hdul_sci) > 1 else hdul_sci[0].data
hdul_sci.close()

pixel_sci = np.array(data_sci_table['PIXEL'], dtype=float).flatten()
flux_sci  = np.array(data_sci_table['FLUX'], dtype=float).flatten()
print(f"Science-Spektrum eingelesen: {len(flux_sci)} Pixel")

# -------------------------
# 2D Kalibrationsspektrum einlesen & x-spiegeln
# -------------------------
hdul_cal = fits.open(calib_2d_fits)
data_cal = np.fliplr(hdul_cal[0].data.copy())
hdul_cal.close()
ny, nx = data_cal.shape
print(f"Kalibrationsbild geladen: {nx} x {ny} Pixel")

# -------------------------
# Hilfsfunktionen für 1D-Extraktion
# -------------------------
def clamp(val, a, b):
    return max(a, min(b, int(val)))

def clamp_range(r, ny):
    return (max(0, int(r[0])), min(ny, int(r[1])))

def compute_trace(daten, y_center, half_width=5, sky_offset=10, sky_width=5, snr_thresh=2.0):
    ny, nx = daten.shape
    x = np.arange(nx)

    # Sky-Fenster
    sky1 = clamp_range((y_center - sky_offset - sky_width, y_center - sky_offset), ny)
    sky2 = clamp_range((y_center + sky_offset, y_center + sky_offset + sky_width), ny)
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

    valid_idx = np.where(valid)[0]
    y_fit = np.full(nx, y_center)
    if len(valid_idx) >= 2:
        y_fit_interp = np.interp(x, x[valid_idx], centroids[valid_idx])
        y_fit = gaussian_filter1d(y_fit_interp, sigma=3)

    return {"x": x, "y_fit": np.clip(y_fit, 0, ny-1), "sky_median_per_col": sky_median_per_col}

def extract_1d_spectrum(daten, y_fit, half_width=5, sky_median_per_col=None):
    ny, nx = daten.shape
    flux = np.zeros(nx)
    for i in range(nx):
        yi = y_fit[i] if np.isfinite(y_fit[i]) else ny//2
        ylo = clamp(np.floor(yi - half_width), 0, ny-1)
        yhi = clamp(np.ceil(yi + half_width), 0, ny-1)
        if yhi < ylo: ylo, yhi = yhi, ylo
        ap_pixels = daten[ylo:yhi+1, i].astype(float)
        npix = ap_pixels.size
        # Sky optional abziehen
        if sky_median_per_col is None:
            flux[i] = np.nansum(ap_pixels)
        else:
            flux[i] = np.nansum(ap_pixels) - sky_median_per_col[i] * npix
    return flux

# -------------------------
# Kalibrationsspektrum extrahieren
# -------------------------
profil_cal = np.sum(data_cal, axis=1)
profil_cal_smooth = gaussian_filter1d(profil_cal, sigma=5)
y_peak_cal = int(np.argmax(profil_cal_smooth))

trace_cal = compute_trace(
    data_cal, y_peak_cal, half_width=8, sky_offset=15, sky_width=5, snr_thresh=1.0
)
flux_cal = extract_1d_spectrum(
    data_cal, trace_cal["y_fit"], half_width=8, sky_median_per_col=trace_cal["sky_median_per_col"]
)

# -------------------------
# Zwei separate Plots
# -------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)

# Science Spektrum
ax1.plot(pixel_sci, flux_sci, color='blue')
ax1.set_title("Science 1D-Spektrum (unkalibriert)")
ax1.set_ylabel("Flux")
ax1.grid(True)

# Kalibrationsspektrum
ax2.plot(trace_cal["x"], flux_cal, color='red')
ax2.set_title("Kalibrations-Spektrum (1D, Sky-subtrahiert) - bitte Linien anklicken")
ax2.set_xlabel("Pixel")
ax2.set_ylabel("Flux")
ax2.grid(True)

# -------------------------
# Option: Wellenlösung aus Datei verwenden?
# -------------------------
use_solution = False
solution_file = None

# CLI: --solution <file>
if len(sys.argv) >= 3 and sys.argv[1] in ("--solution", "--apply"):
    solution_file = sys.argv[2]
    if not os.path.exists(solution_file):
        print(f"Lösungsdatei '{solution_file}' nicht gefunden.")
        solution_file = None
    else:
        use_solution = True
# sonst: falls Standarddatei vorhanden, nachfragen
elif os.path.exists(solution_fname_default):
    ans = input(f"Wellenlösung '{solution_fname_default}' gefunden. Aus Datei verwenden? (j = ja, Enter = nein): ").strip().lower()
    if ans == "j":
        solution_file = solution_fname_default
        use_solution = True

# Wenn Wellenlösung aus Datei verwendet werden soll: direkt anwenden und beenden (keine interaktive Kalibrierung)
if use_solution and solution_file is not None:
    try:
        coeffs = load_solution(solution_file)
        print(f"Wellenlösung geladen aus '{solution_file}': Koeffizienten (höchster Grad zuerst): {coeffs}")
    except Exception as e:
        print(f"Fehler beim Laden der Wellenlösung: {e}")
        use_solution = False

if use_solution:
    # Wellenlängen für Science-Pixel berechnen
    wavelength_sci = np.polyval(coeffs, pixel_sci)

    # Plot und Speichern (PNG + FITS)
    fig_sci = plt.figure(figsize=(10,5))
    plt.plot(wavelength_sci, flux_sci, color='blue')
    plt.title("Science-Spektrum (wellenlängenkalibriert, aus Datei)")
    plt.xlabel("Wellenlänge (Å)")
    plt.ylabel("Flux")
    plt.grid(True)

    outpng = os.path.join(out_dir, "science_spectrum_calibrated.png")
    try:
        fig_sci.tight_layout()
        fig_sci.savefig(outpng, dpi=200)
        print(f"Kalibriertes Science-Spektrum als PNG gespeichert: {outpng}")
    except Exception as e:
        print(f"Fehler beim Speichern der PNG-Datei: {e}")

    # FITS speichern
    outname = os.path.join(out_dir, "science_spectrum_calibrated.fits")
    try:
        col_pixel = fits.Column(name='PIXEL', format='E', array=pixel_sci.astype(np.float32))
        col_wave  = fits.Column(name='WAVELENGTH', format='E', array=wavelength_sci.astype(np.float32))
        col_flux  = fits.Column(name='FLUX', format='E', array=flux_sci.astype(np.float32))
        cols = fits.ColDefs([col_pixel, col_wave, col_flux])
        hdu_table = fits.BinTableHDU.from_columns(cols)
        hdul_out = fits.HDUList([fits.PrimaryHDU(), hdu_table])
        hdul_out.writeto(outname, overwrite=True)
        print(f"Kalibriertes Science-Spektrum gespeichert: {outname}")
    except Exception as e:
        print(f"Fehler beim Schreiben der FITS-Datei: {e}")

    # Wellenlösung (Wellenlänge vs. Pixel) plotten
    pix_full = np.arange(len(flux_cal))
    wav_solution = np.polyval(coeffs, pix_full)
    plt.figure(figsize=(10,5))
    plt.plot(pix_full, wav_solution, color='navy', lw=1.5, label='Wellenlösung')
    plt.title("Wellenlöse-Kurve (aus Datei)")
    plt.xlabel("Pixel (Kalibrations-1D)")
    plt.ylabel("Wellenlänge (Å)")
    plt.grid(True)
    plt.show()

    sys.exit(0)

# -------------------------
# konfigurierbare Suche / Gauß-Fit-Funktion (muss VOR dem interaktiven Callback stehen)
def gaussian(x, A, mu, sigma, c):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

def find_peak_subpixel(flux_arr, x_click, radius=5, do_fit=True):
    """
    Suche lokales Maximum um x_click ± radius.
    Falls do_fit True, passe einen Gauß an und gib das gefittete mu (float) zurück.
    Bei Fehlern wird der einfache lokale Maximalpixel zurückgegeben.
    """
    nx = len(flux_arr)
    x0 = max(0, int(x_click - radius))
    x1 = min(nx - 1, int(x_click + radius))
    x = np.arange(x0, x1 + 1)
    y = flux_arr[x0:x1 + 1]
    if y.size == 0:
        return float(x_click)

    # Startwerte für Fit
    A0 = float(np.max(y) - np.min(y))
    mu0 = float(x[np.argmax(y)])
    sigma0 = max(1.0, (x1 - x0) / 4.0)
    c0 = float(np.min(y))

    if do_fit:
        try:
            popt, _ = curve_fit(
                gaussian, x, y,
                p0=[max(A0, 0.0), mu0, sigma0, c0],
                bounds=([0.0, x0, 0.1, -np.inf], [np.inf, x1, np.inf, np.inf]),
                maxfev=10000
            )
            mu_fit = float(popt[1])
            if mu_fit < x0 - 1e-6 or mu_fit > x1 + 1e-6 or not np.isfinite(mu_fit):
                return float(mu0)
            return mu_fit
        except Exception:
            return float(mu0)
    else:
        return float(mu0)

# Suchradius global (anpassbar)
search_radius = 5

# -------------------------
# Interaktive Linienwahl + Wellenlängen-Input (GUI-TextBox)
# -------------------------
selected_pixels = []
selected_waves = []
current_xpeak = None
current_flux_at_xpeak = None

def onclick(event):
    global current_xpeak, current_flux_at_xpeak
    # nur left-click in Kalib.-Achse akzeptieren
    if event.inaxes != ax2 or event.button != 1:
        return
    if event.xdata is None:
        return
    xpix_click = int(round(event.xdata))
    xpix_click = max(0, min(len(flux_cal)-1, xpix_click))

    # sub-pixel Peak via Gauß-Fit suchen
    xpeak = find_peak_subpixel(flux_cal, xpix_click, radius=search_radius, do_fit=True)
    flux_at_xpeak = float(np.interp(xpeak, np.arange(len(flux_cal)), flux_cal))

    current_xpeak = xpeak
    current_flux_at_xpeak = flux_at_xpeak

    # Markiere Klick und temporären Peak
    ax2.scatter([xpix_click], [flux_cal[xpix_click]], color='yellow', s=40, zorder=6, marker='x')
    # temporärer Marker (cyan) für gefittetes Peak
    ax2.scatter([xpeak], [flux_at_xpeak], color='cyan', s=120, zorder=7)
    if 'info_text' in globals():
        info_text.set_text(f"Gefundener Peak: x = {xpeak:.3f}  — Wellenlänge eingeben und 'Bestätigen' drücken.")
    fig.canvas.draw()

def confirm_wavelength(text):
    global current_xpeak, current_flux_at_xpeak
    if current_xpeak is None:
        print("Kein Peak ausgewählt. Bitte zuerst im Kalibrationsplot klicken.")
        return
    txt = text.strip()
    if txt == "":
        print("Leere Eingabe, verworfen.")
        return
    try:
        wav = float(txt)
    except ValueError:
        print("Ungültige Zahl, verworfen.")
        return

    # speichern
    selected_pixels.append(current_xpeak)
    selected_waves.append(wav)

    # dauerhafte Markierung + Annotation
    ax2.scatter([current_xpeak], [current_flux_at_xpeak], color='lime', s=100, zorder=8)
    ax2.annotate(f"{wav:.2f}", (current_xpeak, current_flux_at_xpeak),
                 textcoords="offset points", xytext=(0,10), ha='center', color='lime')
    if 'info_text' in globals():
        info_text.set_text(f"Referenz hinzugefügt: x={current_xpeak:.3f} → {wav:.2f} Å")
    current_xpeak = None
    current_flux_at_xpeak = None
    text_box.set_val("")  # Textfeld leeren
    fig.canvas.draw()

def done(event):
    plt.close(fig)

# Info-Text oberhalb der Plots (kleines Panel)
info_ax = fig.add_axes([0.02, 0.94, 0.96, 0.04])
info_ax.axis('off')
info_text = info_ax.text(0.01, 0.5,
                        "Klicken Sie eine Linie im unteren Plot, geben Sie die Wellenlänge ins Feld ein und drücken 'Bestätigen' (oder Enter). 'Fertig' schließt.",
                        va='center', fontsize=9)

# TextBox und Buttons unten im Figure
axbox = fig.add_axes([0.15, 0.02, 0.35, 0.04])
text_box = TextBox(axbox, "Wellenlänge [Å]:", initial="")
text_box.on_submit(confirm_wavelength)

btn_confirm_ax = fig.add_axes([0.52, 0.02, 0.12, 0.04])
btn_confirm = Button(btn_confirm_ax, "Bestätigen")
btn_confirm.on_clicked(lambda event: confirm_wavelength(text_box.text))

btn_done_ax = fig.add_axes([0.65, 0.02, 0.12, 0.04])
btn_done = Button(btn_done_ax, "Fertig")
btn_done.on_clicked(done)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

print("Interaktive Auswahl gestartet: Klicken Sie Linien im Kalibrationsspektrum an und benutzen Sie das Textfeld.")
plt.show()  # blockiert bis Fenster geschlossen

fig.canvas.mpl_disconnect(cid)

# -------------------------
# Wellenlösungs-fit und Anwendung auf Science
# -------------------------
if len(selected_pixels) < 2:
    print("Nicht genügend Referenzlinien (mindestens 2 benötigt). Keine Kalibrierung durchgeführt.")
else:
    deg = 1 if len(selected_pixels) == 2 else min(2, len(selected_pixels)-1)
    coeffs = np.polyfit(selected_pixels, selected_waves, deg=deg)
    print(f"Wellenlöse-Polynom (grad {deg}): {coeffs}")

    # Speichere Wellenlösung automatisch
    save_solution(coeffs, solution_fname_default)

    # Wellenlängen für Science-Pixel berechnen
    wavelength_sci = np.polyval(coeffs, pixel_sci)

    # Plot kalibriertes Science-Spektrum (als PNG speichern)
    fig_sci = plt.figure(figsize=(10,5))
    plt.plot(wavelength_sci, flux_sci, color='blue')
    plt.title("Science-Spektrum (wellenlängenkalibriert)")
    plt.xlabel("Wellenlänge (Å)")
    plt.ylabel("Flux")
    plt.grid(True)

    # Markiere kalibrierte Referenzpunkte (vertikale Linien)
    for px, wv in zip(selected_pixels, selected_waves):
        plt.axvline(wv, color='orange', linestyle='--', alpha=0.7)

    outpng = os.path.join(out_dir, "science_spectrum_calibrated.png")
    try:
        fig_sci.tight_layout()
        fig_sci.savefig(outpng, dpi=200)
        print(f"Kalibriertes Science-Spektrum als PNG gespeichert: {outpng}")
    except Exception as e:
        print(f"Fehler beim Speichern der PNG-Datei: {e}")

    plt.show()

    # -------------------------
    # Wellenlösung: Wellenlänge vs. Pixel (grafisch)
    # -------------------------
    pix_full = np.arange(len(flux_cal))
    wav_solution = np.polyval(coeffs, pix_full)

    plt.figure(figsize=(10,5))
    plt.plot(pix_full, wav_solution, color='navy', lw=1.5, label='Wellenlösung')
    # Referenzpunkte überlagern
    plt.scatter(selected_pixels, selected_waves, color='red', zorder=5, label='Referenzlinien')
    plt.title("Wellenlöse-Kurve (Wellenlänge vs. Pixel)")
    plt.xlabel("Pixel (Kalibrations-1D)")
    plt.ylabel("Wellenlänge (Å)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------
    # Kalibriertes Science-Spektrum in FITS speichern
    # -------------------------
    outname = os.path.join(out_dir, "science_spectrum_calibrated.fits")
    try:
        col_pixel = fits.Column(name='PIXEL', format='E', array=pixel_sci.astype(np.float32))
        col_wave  = fits.Column(name='WAVELENGTH', format='E', array=wavelength_sci.astype(np.float32))
        col_flux  = fits.Column(name='FLUX', format='E', array=flux_sci.astype(np.float32))
        cols = fits.ColDefs([col_pixel, col_wave, col_flux])
        hdu_table = fits.BinTableHDU.from_columns(cols)
        hdul_out = fits.HDUList([fits.PrimaryHDU(), hdu_table])
        hdul_out.writeto(outname, overwrite=True)
        print(f"Kalibriertes Science-Spektrum gespeichert: {outname}")
    except Exception as e:
        print(f"Fehler beim Schreiben der FITS-Datei: {e}")