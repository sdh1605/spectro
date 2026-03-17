#!/usr/bin/env python3
"""
Stack multiple FITS files with the same prefix.
Usage: python spec_stack.py WORK_DIR FILE_PREFIX
Example: python spec_stack.py /path/to/work science_spectrum
"""

import os
import sys
import numpy as np
from astropy.io import fits
from glob import glob

def stack_fits_files(files, output_file):
    """
    Stack (average) multiple FITS files.
    
    Args:
        files: List of input FITS file paths
        output_file: Path where to save the stacked FITS
    """
    if not files:
        raise ValueError("Keine Dateien zum Stacken gefunden!")

    # Erste Datei lesen um die Dimensionen zu bestimmen
    with fits.open(files[0]) as hdul:
        first_data = hdul[0].data.copy()
        header = hdul[0].header.copy()

    # Prüfe ob alle Dateien die gleichen Dimensionen haben
    stack = np.zeros_like(first_data, dtype=np.float64)
    n_files = 0

    for file in files:
        try:
            with fits.open(file) as hdul:
                data = hdul[0].data
                if data.shape != first_data.shape:
                    print(f"Warnung: {file} hat abweichende Dimensionen und wird übersprungen!")
                    continue
                stack += data
                n_files += 1
                print(f"Datei hinzugefügt: {file}")
        except Exception as e:
            print(f"Fehler beim Lesen von {file}: {e}")
            continue

    if n_files == 0:
        raise ValueError("Keine gültigen Dateien zum Stacken gefunden!")

    # Mittelwert berechnen
    stack /= n_files

    # Header aktualisieren
    header['NCOMBINE'] = (n_files, 'Number of combined images')
    header.add_comment(f'Stacked from {n_files} files with prefix {os.path.basename(files[0]).split("_")[0]}')
    header.add_comment(f'Stacking performed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Gestacktes Bild speichern
    hdu = fits.PrimaryHDU(data=stack, header=header)
    hdu.writeto(output_file, overwrite=True)
    print(f"\nErfolgreich {n_files} Dateien gestackt und gespeichert als: {output_file}")
    print(f"Dimensionen: {stack.shape}")

def main():
    # Kommandozeilenargumente prüfen
    if len(sys.argv) != 3:
        print("Verwendung: python spec_stack.py ARBEITSORDNER DATEI_PREFIX")
        print("Beispiel: python spec_stack.py /path/to/work science_spectrum")
        sys.exit(1)

    work_dir = sys.argv[1]
    file_prefix = sys.argv[2]

    # Ordnerstruktur überprüfen
    in_dir = os.path.join(work_dir, "in")
    if not os.path.exists(in_dir):
        print(f"Fehler: Eingabeordner '{in_dir}' existiert nicht!")
        sys.exit(1)

    # FITS-Dateien mit dem gegebenen Präfix suchen
    search_pattern = os.path.join(in_dir, f"{file_prefix}*.fits")
    input_files = sorted(glob(search_pattern))

    if not input_files:
        print(f"Keine FITS-Dateien mit Präfix '{file_prefix}' in {in_dir} gefunden!")
        sys.exit(1)

    print(f"Gefundene Dateien zum Stacken: {len(input_files)}")
    for f in input_files:
        print(f"  {os.path.basename(f)}")

    # Ausgabedatei definieren
    output_file = os.path.join(in_dir, "science_spectrum.fits")

    try:
        # Stack durchführen
        stack_fits_files(input_files, output_file)
    except Exception as e:
        print(f"Fehler beim Stacken: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()