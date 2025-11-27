#!/usr/bin/env python3
"""
Convert CZI files to TIFF format for Trident preprocessing.

This script converts CZI whole slide images to pyramidal TIFF format,
naming them by prep_lims_id for use with Trident pipeline.

Usage:
    python convert_czi_trident.py [--metadata PATH] [--image_dir PATH] [--output_dir PATH]
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Try to import pyvips for efficient TIFF conversion
try:
    import pyvips

    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False
    print("Warning: pyvips not available. Will try alternative methods.")

# Try to import aicsimageio as fallback
try:
    from aicsimageio import AICSImage
    import tifffile
    import numpy as np

    AICSIMAGE_AVAILABLE = True
except ImportError:
    AICSIMAGE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_METADATA_PATH = "/home/scai/scratch/E0141_CUP/data/E0141_combined_meta.xlsx"
DEFAULT_IMAGE_DIR = "/mnt/histopathology/E0141/P01"
DEFAULT_OUTPUT_DIR = (
    "/home/scai/scratch/E0141_CUP/data/trident_processing/converted_wsis"
)


def convert_czi_with_pyvips(input_path, output_path):
    """
    Convert CZI to pyramidal TIFF using pyvips.

    Args:
        input_path: Path to input CZI file
        output_path: Path to output TIFF file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the CZI file
        image = pyvips.Image.new_from_file(str(input_path))

        # Save as pyramidal TIFF with compression
        image.tiffsave(
            str(output_path),
            compression="jpeg",
            Q=90,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
            bigtiff=True,
        )
        return True
    except Exception as e:
        logger.error(f"pyvips conversion failed: {e}")
        return False


def convert_czi_with_aicsimage(input_path, output_path):
    """
    Convert CZI to TIFF using AICSImage and tifffile.

    Args:
        input_path: Path to input CZI file
        output_path: Path to output TIFF file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the CZI file
        img_obj = AICSImage(str(input_path))

        # Get image data in YXC format
        if img_obj.dims.order == "TCZYXS" and img_obj.shape[-1] == 3:
            # Last dimension is S=3, those are your RGB channels
            other_dims = img_obj.data.shape[:-3]
            img = img_obj.data[tuple([0] * len(other_dims)) + (Ellipsis,)]  # H x W x C
        else:
            # Use normal method
            img = img_obj.get_image_data("YXC", T=0, Z=0, S=0)

        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            # Normalize to 0-255 range
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        # Save as TIFF with compression
        tifffile.imwrite(
            str(output_path),
            img,
            compression="jpeg",
            photometric="rgb",
            tile=(256, 256),
        )
        return True
    except Exception as e:
        logger.error(f"AICSImage conversion failed: {e}")
        return False


def convert_single_czi(input_path, output_path, force=False):
    """
    Convert a single CZI file to TIFF.

    Args:
        input_path: Path to input CZI file
        output_path: Path to output TIFF file
        force: If True, overwrite existing files

    Returns:
        dict: Status information about the conversion
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Check if input exists
    if not input_path.exists():
        return {
            "status": "error",
            "reason": "input_not_found",
            "message": f"Input file not found: {input_path}",
        }

    # Check if output already exists
    if output_path.exists() and not force:
        file_size = output_path.stat().st_size / (1024**3)  # Size in GB
        return {
            "status": "skipped",
            "reason": "already_exists",
            "message": f"Output already exists ({file_size:.2f} GB)",
        }

    # Try conversion with available methods
    success = False

    if PYVIPS_AVAILABLE:
        logger.debug(f"Attempting conversion with pyvips: {input_path.name}")
        success = convert_czi_with_pyvips(input_path, output_path)

    if not success and AICSIMAGE_AVAILABLE:
        logger.debug(f"Attempting conversion with AICSImage: {input_path.name}")
        success = convert_czi_with_aicsimage(input_path, output_path)

    if success:
        file_size = output_path.stat().st_size / (1024**3)  # Size in GB
        return {
            "status": "success",
            "reason": "converted",
            "message": f"Converted successfully ({file_size:.2f} GB)",
        }
    else:
        return {
            "status": "error",
            "reason": "conversion_failed",
            "message": "All conversion methods failed",
        }


def load_metadata(metadata_path):
    """
    Load metadata from Excel file.

    Args:
        metadata_path: Path to Excel metadata file

    Returns:
        pd.DataFrame: Metadata dataframe
    """
    logger.info(f"Loading metadata from: {metadata_path}")

    try:
        df = pd.read_excel(metadata_path)
        logger.info(f"Loaded {len(df)} records from metadata")

        # Check required columns
        required_cols = ["Image Label", "prep_lims_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Convert CZI files to TIFF format for Trident preprocessing"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=DEFAULT_METADATA_PATH,
        help="Path to metadata Excel file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing CZI files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save converted TIFF files",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite of existing files"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: only convert first file"
    )

    args = parser.parse_args()

    # Check available conversion methods
    if not PYVIPS_AVAILABLE and not AICSIMAGE_AVAILABLE:
        logger.error(
            "No conversion libraries available. Install pyvips or aicsimageio+tifffile."
        )
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("CZI to TIFF Converter for Trident Preprocessing")
    logger.info("=" * 70)
    logger.info(f"Metadata: {args.metadata}")
    logger.info(f"Image directory: {args.image_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Force overwrite: {args.force}")
    logger.info(f"Test mode: {args.test}")
    logger.info(f"pyvips available: {PYVIPS_AVAILABLE}")
    logger.info(f"AICSImage available: {AICSIMAGE_AVAILABLE}")
    logger.info("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")

    # Load metadata
    meta_df = load_metadata(args.metadata)

    # Filter for CZI files
    image_dir = Path(args.image_dir)
    czi_records = []

    logger.info("Scanning for CZI files...")
    for idx, row in meta_df.iterrows():
        img_label = row["Image Label"]
        lims_id = row["prep_lims_id"]

        czi_path = image_dir / f"{img_label}.czi"
        if czi_path.exists():
            czi_records.append(
                {
                    "image_label": img_label,
                    "prep_lims_id": lims_id,
                    "input_path": czi_path,
                    "output_path": output_dir / f"{lims_id}.tiff",
                }
            )

    logger.info(f"Found {len(czi_records)} CZI files to convert")

    if len(czi_records) == 0:
        logger.warning("No CZI files found. Exiting.")
        sys.exit(0)

    # Test mode: only convert first file
    if args.test:
        logger.info("TEST MODE: Converting only the first file")
        czi_records = czi_records[:1]

    # Convert files
    logger.info(f"\nConverting {len(czi_records)} CZI files...")
    results = []

    for record in tqdm(czi_records, desc="Converting CZI files"):
        logger.info(
            f"\nProcessing: {record['image_label']} -> {record['prep_lims_id']}"
        )

        result = convert_single_czi(
            record["input_path"], record["output_path"], force=args.force
        )

        result.update(
            {
                "image_label": record["image_label"],
                "prep_lims_id": record["prep_lims_id"],
            }
        )
        results.append(result)

        logger.info(f"  Status: {result['status']} - {result['message']}")

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 70)

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")

    logger.info(f"Total files processed: {len(results)}")
    logger.info(f"Successfully converted: {success_count}")
    logger.info(f"Skipped (already exist): {skipped_count}")
    logger.info(f"Errors: {error_count}")

    if error_count > 0:
        logger.info("\nFailed conversions:")
        for r in results:
            if r["status"] == "error":
                logger.info(
                    f"  - {r['image_label']} ({r['prep_lims_id']}): {r['message']}"
                )

    logger.info("=" * 70)
    logger.info("Conversion complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
