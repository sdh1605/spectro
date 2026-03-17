#!/usr/bin/env python3
"""
Stack multiple FITS files with the same prefix.
Usage: python3 spec_stack.py WORK_DIR FILE_PREFIX [--name OUTPUT_NAME] [--arith]
Example: python3 spec_stack.py /path/to/work science_spectrum --name calibration_lamp
"""

import argparse
import os
import sys
import numpy as np
from astropy.io import fits
from glob import glob

def stack_fits_files(files, output_file, use_arith=False):
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
    stack_frames = []
    n_files = 0

    for file in files:
        try:
            with fits.open(file) as hdul:
                data = hdul[0].data
                if data.shape != first_data.shape:
                    print(f"Warnung: {file} hat abweichende Dimensionen und wird übersprungen!")
                    continue
                stack_frames.append(data.astype(np.float64))
                n_files += 1
                print(f"Datei hinzugefügt: {file}")
        except Exception as e:
            print(f"Fehler beim Lesen von {file}: {e}")
            continue

    if n_files == 0:
        raise ValueError("Keine gültigen Dateien zum Stacken gefunden!")

    # Median ist Standard; optional kann arithmetisches Mittel verwendet werden.
    stack_cube = np.stack(stack_frames, axis=0)
    if use_arith:
        stack = np.nanmean(stack_cube, axis=0)
        combine_method = "MEAN"
    else:
        stack = np.nanmedian(stack_cube, axis=0)
        combine_method = "MEDIAN"

    # Header aktualisieren
    header['NCOMBINE'] = (n_files, 'Number of combined images')
    header['COMBMETH'] = (combine_method, 'Combination method: MEDIAN or MEAN')
    header.add_comment(f'Stacked from {n_files} files with prefix {os.path.basename(files[0]).split("_")[0]}')
    header.add_comment(f'Stacking performed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Gestacktes Bild speichern
    hdu = fits.PrimaryHDU(data=stack, header=header)
    hdu.writeto(output_file, overwrite=True)
    print(f"\nErfolgreich {n_files} Dateien gestackt ({combine_method}) und gespeichert als: {output_file}")
    print(f"Dimensionen: {stack.shape}")

def main():
    parser = argparse.ArgumentParser(
        description="Stackt mehrere FITS-Dateien mit gleichem Prefix (Standard: Median)."
    )
    parser.add_argument("work_dir", help="Arbeitsordner mit in/-Unterordner")
    parser.add_argument("file_prefix", help="Prefix der zu stackenden Dateien in in/")
    parser.add_argument(
        "--name",
        dest="output_name",
        default="science_spectrum",
        help="Ausgabedateiname ohne oder mit .fits (Standard: science_spectrum)",
    )
    parser.add_argument(
        "--arith",
        action="store_true",
        help="Verwendet arithmetisches Mittel statt Median",
    )
    args = parser.parse_args()

    work_dir = args.work_dir
    file_prefix = args.file_prefix

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
    output_name = args.output_name
    if not output_name.lower().endswith(".fits"):
        output_name = f"{output_name}.fits"
    output_file = os.path.join(in_dir, output_name)

    try:
        # Stack durchführen
        stack_fits_files(input_files, output_file, use_arith=args.arith)
    except Exception as e:
        print(f"Fehler beim Stacken: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()