#!/usr/bin/env python3

"""
Generate a CSV file listing WSIs and their MPP for Trident.

Assumptions:
- All slides are scanned at 20x.
- Converted TIFFs are named as: {prep_lims_id}.tiff
- TIFFs live in the same converted_wsis dir you used in convert_wsi_trident.py

Output CSV columns:
- wsi: filename of the slide (relative to --wsi_dir)
- mpp: microns per pixel (float)

You will pass this CSV to Trident via:
    --custom_list_of_wsis /path/to/my_wsis_20x.csv
"""

import pandas as pd
from pathlib import Path

# ---- Paths: adjust if needed ----
METADATA_PATH = Path("/home/scai/scratch/E0141_CUP/data/E0141_combined_meta.xlsx")
WSI_DIR = Path("/home/scai/scratch/E0141_CUP/data/trident_processing/converted_wsis")
CSV_OUT = Path("/home/scai/scratch/E0141_CUP/data/trident_processing/my_wsis_20x.csv")

# ---- 20x MPP (in microns per pixel) ----
# Many scanners use ~0.50 µm/px at 20x. If you know the exact value for your scanner,
# change this to that number (e.g. 0.46, 0.497, etc.).
MPP_20X = 0.50


def main():
    print(f"Loading metadata from: {METADATA_PATH}")
    meta = pd.read_excel(METADATA_PATH)

    if "prep_lims_id" not in meta.columns:
        raise ValueError("prep_lims_id column not found in metadata!")

    # Build WSI filenames and paths
    meta["wsi"] = meta["prep_lims_id"].astype(str) + ".tiff"
    meta["wsi_path"] = meta["wsi"].apply(lambda name: WSI_DIR / name)

    # Keep only rows where the TIFF actually exists
    meta["has_tiff"] = meta["wsi_path"].apply(lambda p: p.exists())
    df = meta[meta["has_tiff"]].copy()

    if df.empty:
        raise RuntimeError(f"No TIFFs found in {WSI_DIR}. Did you run the converter?")

    # Set constant MPP for all 20x slides
    df["mpp"] = MPP_20X

    # Trident only needs these columns
    out_df = df[["wsi", "mpp"]]

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(CSV_OUT, index=False)

    print(f"Total metadata rows: {len(meta)}")
    print(f"Rows with existing TIFFs: {len(df)}")
    print(f"Saved Trident WSI list to: {CSV_OUT}")


if __name__ == "__main__":
    main()
