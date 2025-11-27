#!/usr/bin/env python3
"""
Convert CZI and VSI files to TIFF format using Trident's AnyToTiffConverter.

This script converts whole slide images (CZI and VSI formats) to TIFF format,
naming them by prep_lims_id for use with Trident pipeline.

Note: Configure FILE_FORMATS_TO_PROCESS to control which formats to process
      ("czi", "vsi", or "both"). Currently defaults to "czi" only due to
      Trident technical issues with VSI files.

Usage:
    python convert_wsi_trident.py [--metadata PATH] [--image_dir PATH] [--output_dir PATH]
"""

import os

os.environ["VIPS_WARNING"] = "0"

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Import Trident converter
try:
    from trident.Converter import AnyToTiffConverter

    TRIDENT_AVAILABLE = True
except ImportError:
    TRIDENT_AVAILABLE = False
    print("ERROR: Trident not available. Please install trident.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============ Configuration ============
# Default paths
DEFAULT_METADATA_PATH = "/home/scai/scratch/E0141_CUP/data/E0141_combined_meta.xlsx"
DEFAULT_IMAGE_DIR = "/mnt/histopathology/E0141/P01"
DEFAULT_OUTPUT_DIR = (
    "/home/scai/scratch/E0141_CUP/data/trident_processing/converted_wsis"
)


# ============ Performance Configuration ============
# Toggle multiprocessing
ENABLE_MULTIPROCESSING = True  # Parallel file conversion with joblib
NUM_WORKERS = 16  # Trident uses CPU (not GPU) for conversion

# File format processing
# Options: "czi", "vsi", or "both"
# Note: Trident currently has technical issues with VSI files
FILE_FORMATS_TO_PROCESS = "czi"  # Default to CZI only due to Trident VSI issues


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


def convert_single_wsi(
    input_path, output_path, output_dir, mpp=0.25, zoom=1.0, force=False, bigtiff=False
):
    """
    Convert a single WSI file to TIFF using Trident's AnyToTiffConverter.

    Args:
        input_path: Path to input WSI file (CZI or VSI)
        output_path: Path to output TIFF file (used for naming after conversion)
        output_dir: Directory for converter initialization
        mpp: Microns per pixel (default 0.25 for 40x magnification)
        zoom: Zoom factor for resizing (default 1.0 for no resize)
        force: If True, overwrite existing files
        bigtiff: If True, use BigTIFF format for large files

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

    try:
        # Initialize converter for this worker process
        converter = AnyToTiffConverter(job_dir=str(output_dir), bigtiff=bigtiff)

        # Convert using Trident's AnyToTiffConverter
        # Note: process_file saves with the input filename (without extension)
        # We'll need to rename it to match our desired output name (prep_lims_id)
        logger.debug(f"Converting {input_path.name} to TIFF...")

        # The converter will save as {job_dir}/{input_basename}.tiff
        input_basename = input_path.stem  # filename without extension
        temp_output = Path(converter.job_dir) / f"{input_basename}.tiff"

        # Remove temp output if it exists and we're forcing
        if temp_output.exists() and force:
            temp_output.unlink()

        # Process the file
        converter.process_file(str(input_path), mpp=mpp, zoom=zoom)

        # Rename to desired output name if needed
        if temp_output.exists():
            if temp_output != output_path:
                # Rename to prep_lims_id.tiff
                temp_output.rename(output_path)

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
                "message": "Conversion completed but output file not found",
            }

    except Exception as e:
        return {
            "status": "error",
            "reason": "conversion_failed",
            "message": f"Conversion failed: {str(e)}",
        }


def process_single_wsi_record(record, output_dir, mpp, zoom, force, bigtiff):
    """
    Process a single WSI record for parallel execution.

    Args:
        record: Dictionary containing WSI record information
        output_dir: Directory for output files
        mpp: Microns per pixel
        zoom: Zoom factor
        force: Force overwrite flag
        bigtiff: BigTIFF format flag

    Returns:
        dict: Processing result with metadata
    """
    result = convert_single_wsi(
        record["input_path"],
        record["output_path"],
        output_dir,
        mpp=mpp,
        zoom=zoom,
        force=force,
        bigtiff=bigtiff,
    )
    result.update(
        {
            "image_label": record["image_label"],
            "prep_lims_id": record["prep_lims_id"],
            "file_format": record["file_format"],
        }
    )
    return result


def process_wsi_files(
    meta_df,
    image_dir,
    output_dir,
    mpp=0.25,
    zoom=1.0,
    force=False,
    test_mode=False,
    bigtiff=False,
):
    """
    Process all WSI files from metadata.

    Args:
        meta_df: Metadata dataframe
        image_dir: Directory containing WSI files
        output_dir: Directory to save converted TIFF files
        mpp: Microns per pixel (default 0.25 for 40x magnification)
        zoom: Zoom factor for resizing (default 1.0 for no resize)
        force: If True, overwrite existing files
        test_mode: If True, only process first file
        bigtiff: If True, use BigTIFF format for large files

    Returns:
        list: List of result dictionaries
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # Scan for WSI files based on configuration
    wsi_records = []

    # Determine which formats to process
    formats_to_check = []
    if FILE_FORMATS_TO_PROCESS == "both":
        formats_to_check = ["czi", "vsi"]
        logger.info("Scanning for CZI and VSI files...")
    elif FILE_FORMATS_TO_PROCESS == "czi":
        formats_to_check = ["czi"]
        logger.info("Scanning for CZI files only...")
    elif FILE_FORMATS_TO_PROCESS == "vsi":
        formats_to_check = ["vsi"]
        logger.info("Scanning for VSI files only...")
    else:
        raise ValueError(
            f"Invalid FILE_FORMATS_TO_PROCESS: {FILE_FORMATS_TO_PROCESS}. "
            f"Must be 'czi', 'vsi', or 'both'."
        )

    for idx, row in meta_df.iterrows():
        img_label = row["Image Label"]
        lims_id = row["prep_lims_id"]

        input_path = None
        file_format = None

        # Check for files based on configured formats
        for fmt in formats_to_check:
            file_path = image_dir / f"{img_label}.{fmt}"
            if file_path.exists():
                input_path = file_path
                file_format = fmt
                break  # Use first matching format

        if input_path is not None:
            wsi_records.append(
                {
                    "image_label": img_label,
                    "prep_lims_id": lims_id,
                    "input_path": input_path,
                    "output_path": output_dir / f"{lims_id}.tiff",
                    "file_format": file_format,
                }
            )

    logger.info(f"Found {len(wsi_records)} WSI files to convert")

    if len(wsi_records) == 0:
        logger.warning("No WSI files found. Exiting.")
        return []

    # Test mode: only process first file
    if test_mode:
        logger.info("TEST MODE: Converting only the first file")
        wsi_records = wsi_records[:1]

    # Convert files with parallel or sequential processing
    if ENABLE_MULTIPROCESSING and len(wsi_records) > 1:
        logger.info(
            f"\nConverting {len(wsi_records)} files in parallel with {NUM_WORKERS} workers..."
        )
        results = Parallel(n_jobs=NUM_WORKERS, backend="loky")(
            delayed(process_single_wsi_record)(
                record, output_dir, mpp, zoom, force, bigtiff
            )
            for record in tqdm(wsi_records, desc="Converting WSI files")
        )
    else:
        logger.info(f"\nConverting {len(wsi_records)} files sequentially...")
        results = []
        for record in tqdm(wsi_records, desc="Converting WSI files"):
            result = process_single_wsi_record(
                record, output_dir, mpp, zoom, force, bigtiff
            )
            results.append(result)
            logger.info(
                f"  {record['image_label']} ({record['file_format']}) -> {record['prep_lims_id']}: "
                f"{result['status']} - {result['message']}"
            )

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Convert CZI and VSI files to TIFF format using Trident"
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
        help="Directory containing WSI files (CZI and VSI)",
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
    parser.add_argument(
        "--bigtiff", action="store_true", help="Use BigTIFF format for large files"
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=0.25,
        help="Microns per pixel (default 0.25 for 40x magnification)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom factor for resizing (default 1.0 for no resize, 0.5 for half size)",
    )

    args = parser.parse_args()

    # Check Trident availability
    if not TRIDENT_AVAILABLE:
        logger.error("Trident not available. Install with: pip install trident")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("WSI to TIFF Converter using Trident")
    logger.info("=" * 70)
    logger.info(f"Metadata: {args.metadata}")
    logger.info(f"Image directory: {args.image_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Force overwrite: {args.force}")
    logger.info(f"Test mode: {args.test}")
    logger.info(f"BigTIFF format: {args.bigtiff}")
    logger.info(f"Microns per pixel (mpp): {args.mpp}")
    logger.info(f"Zoom factor: {args.zoom}")
    logger.info(
        f"Multiprocessing: {ENABLE_MULTIPROCESSING} (workers={NUM_WORKERS if ENABLE_MULTIPROCESSING else 1})"
    )
    logger.info(f"File formats to process: {FILE_FORMATS_TO_PROCESS}")
    logger.info("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")

    # Load metadata
    meta_df = load_metadata(args.metadata)

    # Process WSI files
    results = process_wsi_files(
        meta_df,
        args.image_dir,
        output_dir,
        mpp=args.mpp,
        zoom=args.zoom,
        force=args.force,
        test_mode=args.test,
        bigtiff=args.bigtiff,
    )

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
