#!/usr/bin/env python3
"""
Convert VSI files to TIFF format for Trident preprocessing.

This script converts VSI whole slide images to pyramidal TIFF format,
naming them by prep_lims_id for use with Trident pipeline.
Uses AICSImage (bioformats) for VSI reading.

Usage:
    python convert_vsi_aicsimage.py [--metadata PATH] [--image_dir PATH] [--output_dir PATH]
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress SLF4J warnings from Java-based image readers (bioformats)
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.slf4j.simpleLogger.defaultLogLevel=off"

# Import AICSImage for VSI reading
try:
    from aicsimageio import AICSImage

    AICSIMAGE_AVAILABLE = True
except ImportError:
    AICSIMAGE_AVAILABLE = False
    print("ERROR: aicsimageio not available. Please install it.")
    sys.exit(1)

# Try to import tifffile for TIFF writing
try:
    import tifffile

    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

# Try pyvips as alternative for TIFF writing
try:
    import pyvips

    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False

# Try bfio for robust tile-based reading (fallback for large images)
try:
    from bfio import BioReader

    BFIO_AVAILABLE = True
except ImportError:
    BFIO_AVAILABLE = False

# Try OpenSlide for whole slide image reading (another fallback)
try:
    import openslide

    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

# Try jpype for direct Bio-Formats tile reading
try:
    import jpype
    import jpype.imports

    JPYPE_AVAILABLE = True
except ImportError:
    JPYPE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_METADATA_PATH = "/home/scai/scratch/E0141_CUP/data/E0141_combined_meta.xlsx"
DEFAULT_IMAGE_DIR = "/mnt/histopathology/E0141/P01"
DEFAULT_OUTPUT_DIR = (
    "/home/scai/scratch/E0141_CUP/data/trident_processing/converted_wsis"
)


def load_vsi_with_bfio(input_path, tile_size=2048):
    """
    Load VSI file using bfio with true tile-based reading.
    This bypasses the Bio-Formats 2GB limit by using the native tile API.

    Args:
        input_path: Path to input VSI file
        tile_size: Size of tiles to read (default 2048x2048)

    Returns:
        numpy.ndarray: Image data in YXC format (Height x Width x Channels)
    """
    logger.info(f"Loading VSI with bfio (tile size: {tile_size})...")

    with BioReader(str(input_path)) as reader:
        # Get image dimensions
        height = reader.Y
        width = reader.X
        channels = reader.C if reader.C else 3

        logger.info(f"Image dimensions: {height}x{width}x{channels}")

        # Get dtype
        dtype = reader.dtype

        # Allocate output array
        img = np.zeros((height, width, channels), dtype=dtype)

        # Calculate number of tiles
        n_tiles_y = int(np.ceil(height / tile_size))
        n_tiles_x = int(np.ceil(width / tile_size))
        total_tiles = n_tiles_y * n_tiles_x

        logger.info(
            f"Reading {total_tiles} tiles ({n_tiles_y}x{n_tiles_x}) of size {tile_size}x{tile_size}..."
        )

        # Read tiles
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                y_start = i * tile_size
                y_end = min((i + 1) * tile_size, height)
                x_start = j * tile_size
                x_end = min((j + 1) * tile_size, width)

                tile_num = i * n_tiles_x + j + 1
                if tile_num % 10 == 0 or tile_num == 1 or tile_num == total_tiles:
                    logger.info(
                        f"Reading tile {tile_num}/{total_tiles} (Y:{y_start}-{y_end}, X:{x_start}-{x_end})"
                    )

                # Read tile using bfio's native tile-based reading
                # bfio uses [Y, X, Z, C, T] indexing
                tile = reader[y_start:y_end, x_start:x_end, 0:1, 0:channels, 0:1]

                # Squeeze out Z and T dimensions to get (Y, X, C)
                tile = np.squeeze(tile)

                # Handle case where channels dimension might be squeezed out
                if tile.ndim == 2:
                    tile = tile[:, :, np.newaxis]

                # Place tile in output array
                img[y_start:y_end, x_start:x_end, :] = tile

        logger.info("Finished reading all tiles via bfio")

    return img


def load_vsi_with_openslide(input_path, tile_size=4096):
    """
    Load VSI file using OpenSlide with region-based reading.
    OpenSlide is designed for whole slide images and handles large files natively.

    Args:
        input_path: Path to input VSI file
        tile_size: Size of tiles to read (default 4096x4096)

    Returns:
        numpy.ndarray: Image data in YXC format (Height x Width x Channels)
    """
    logger.info(f"Loading VSI with OpenSlide (tile size: {tile_size})...")

    slide = openslide.OpenSlide(str(input_path))

    # Get dimensions at level 0 (highest resolution)
    width, height = slide.dimensions
    logger.info(f"Image dimensions: {height}x{width} (level 0)")

    # Allocate output array (RGB)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate number of tiles
    n_tiles_y = int(np.ceil(height / tile_size))
    n_tiles_x = int(np.ceil(width / tile_size))
    total_tiles = n_tiles_y * n_tiles_x

    logger.info(
        f"Reading {total_tiles} tiles ({n_tiles_y}x{n_tiles_x}) of size {tile_size}x{tile_size}..."
    )

    # Read tiles
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            y_start = i * tile_size
            x_start = j * tile_size

            # Calculate actual tile size (may be smaller at edges)
            actual_height = min(tile_size, height - y_start)
            actual_width = min(tile_size, width - x_start)

            tile_num = i * n_tiles_x + j + 1
            if tile_num % 10 == 0 or tile_num == 1 or tile_num == total_tiles:
                logger.info(
                    f"Reading tile {tile_num}/{total_tiles} (Y:{y_start}-{y_start + actual_height}, X:{x_start}-{x_start + actual_width})"
                )

            # Read region using OpenSlide (returns RGBA PIL Image)
            # Note: OpenSlide uses (x, y) coordinate order
            tile_pil = slide.read_region(
                (x_start, y_start), 0, (actual_width, actual_height)
            )

            # Convert to RGB numpy array (drop alpha channel)
            tile = np.array(tile_pil.convert("RGB"))

            # Place tile in output array
            img[
                y_start : y_start + actual_height, x_start : x_start + actual_width, :
            ] = tile

    slide.close()
    logger.info("Finished reading all tiles via OpenSlide")

    return img


def load_vsi_with_bioformats_direct(input_path, tile_size=2048):
    """
    Load VSI file using Bio-Formats Java API directly via jpype.
    This uses the native openBytes(plane, x, y, w, h) method for true tile-based reading.

    Args:
        input_path: Path to input VSI file
        tile_size: Size of tiles to read (default 2048x2048)

    Returns:
        numpy.ndarray: Image data in YXC format (Height x Width x Channels)
    """
    logger.info(f"Loading VSI with Bio-Formats direct (tile size: {tile_size})...")

    # Start JVM if not already running (AICSImage should have started it)
    if not jpype.isJVMStarted():
        # Try to find bioformats_jar
        try:
            import bioformats_jar

            jpype.startJVM(classpath=[bioformats_jar.get_path()])
        except Exception as e:
            raise RuntimeError(f"Could not start JVM with bioformats: {e}")

    # Import Java classes
    from loci.formats import ImageReader
    from loci.formats import MetadataTools

    # Create reader
    reader = ImageReader()
    meta = MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(meta)
    reader.setId(str(input_path))

    # Find the largest series (scene)
    n_series = reader.getSeriesCount()
    best_series = 0
    max_pixels = 0

    for s in range(n_series):
        reader.setSeries(s)
        pixels = reader.getSizeX() * reader.getSizeY()
        if pixels > max_pixels:
            max_pixels = pixels
            best_series = s

    reader.setSeries(best_series)
    logger.info(f"Selected series {best_series} (largest of {n_series})")

    # Get dimensions
    width = reader.getSizeX()
    height = reader.getSizeY()
    n_channels = reader.getSizeC()
    pixel_type = reader.getPixelType()
    rgb = reader.isRGB()
    samples_per_pixel = reader.getRGBChannelCount()

    logger.info(
        f"Image dimensions: {height}x{width}, channels: {n_channels}, "
        f"RGB: {rgb}, samples/pixel: {samples_per_pixel}"
    )

    # Determine numpy dtype from Bio-Formats pixel type
    # FormatTools pixel types: INT8=0, UINT8=1, INT16=2, UINT16=3, etc.
    dtype_map = {
        0: np.int8,
        1: np.uint8,
        2: np.int16,
        3: np.uint16,
        4: np.int32,
        5: np.uint32,
        6: np.float32,
        7: np.float64,
    }
    dtype = dtype_map.get(pixel_type, np.uint8)
    bytes_per_pixel = np.dtype(dtype).itemsize

    # For RGB images, we want (H, W, 3)
    if rgb and samples_per_pixel == 3:
        out_channels = 3
    else:
        out_channels = n_channels

    # Allocate output array
    img = np.zeros((height, width, out_channels), dtype=dtype)

    # Calculate number of tiles
    n_tiles_y = int(np.ceil(height / tile_size))
    n_tiles_x = int(np.ceil(width / tile_size))
    total_tiles = n_tiles_y * n_tiles_x

    logger.info(
        f"Reading {total_tiles} tiles ({n_tiles_y}x{n_tiles_x}) of size {tile_size}x{tile_size}..."
    )

    # Read tiles using openBytes with tile coordinates
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            x_start = j * tile_size
            y_start = i * tile_size
            actual_width = min(tile_size, width - x_start)
            actual_height = min(tile_size, height - y_start)

            tile_num = i * n_tiles_x + j + 1
            if tile_num % 10 == 0 or tile_num == 1 or tile_num == total_tiles:
                logger.info(
                    f"Reading tile {tile_num}/{total_tiles} (Y:{y_start}-{y_start + actual_height}, X:{x_start}-{x_start + actual_width})"
                )

            if rgb:
                # For RGB, read plane 0 with tile coordinates
                # openBytes(int no, byte[] buf, int x, int y, int w, int h)
                byte_array = reader.openBytes(
                    0, x_start, y_start, actual_width, actual_height
                )

                # Convert Java byte array to numpy
                tile_bytes = np.array(byte_array, dtype=np.int8).view(dtype)

                # Reshape: Bio-Formats returns interleaved RGB
                tile = tile_bytes.reshape(
                    actual_height, actual_width, samples_per_pixel
                )
                img[
                    y_start : y_start + actual_height,
                    x_start : x_start + actual_width,
                    :,
                ] = tile[:, :, :out_channels]
            else:
                # For non-RGB, read each channel separately
                for c in range(out_channels):
                    plane_index = reader.getIndex(0, c, 0)  # Z=0, C=c, T=0
                    byte_array = reader.openBytes(
                        plane_index, x_start, y_start, actual_width, actual_height
                    )
                    tile_bytes = np.array(byte_array, dtype=np.int8).view(dtype)
                    tile = tile_bytes.reshape(actual_height, actual_width)
                    img[
                        y_start : y_start + actual_height,
                        x_start : x_start + actual_width,
                        c,
                    ] = tile

    reader.close()
    logger.info("Finished reading all tiles via Bio-Formats direct")

    return img


def load_vsi_image(input_path):
    """
    Load the largest scene (tissue) from a VSI file using AICSImage.
    Uses dask arrays for lazy tile-based reading to avoid Bio-Formats 2GB memory limit.

    Args:
        input_path: Path to input VSI file

    Returns:
        numpy.ndarray: Image data in YXC format (Height x Width x Channels)
    """
    try:
        logger.debug(f"Loading VSI file: {input_path.name}")
        img_obj = AICSImage(str(input_path))

        # --- NEW LOGIC: Select the largest scene ---
        best_scene_index = 0
        max_pixels = 0

        # Iterate through all available scenes to find the tissue image
        for scene_id in img_obj.scenes:
            img_obj.set_scene(scene_id)
            # Calculate total pixels (Height * Width)
            current_pixels = img_obj.dims.Y * img_obj.dims.X

            if current_pixels > max_pixels:
                max_pixels = current_pixels
                best_scene_index = scene_id

        # Set the image object to the largest scene found
        img_obj.set_scene(best_scene_index)
        logger.debug(
            f"Selected Scene {best_scene_index} with dimensions: {img_obj.dims}"
        )
        # -------------------------------------------

        # Get image dimensions
        height = img_obj.dims.Y
        width = img_obj.dims.X

        # Determine number of channels from shape
        shape = img_obj.shape
        dims_order = img_obj.dims.order
        logger.debug(f"Image shape: {shape}, dims order: {dims_order}")

        # For VSI files, typically TCZYXS with last dim being samples (RGB)
        if dims_order == "TCZYXS" and shape[-1] == 3:
            n_channels = 3
        elif hasattr(img_obj.dims, "C"):
            n_channels = img_obj.dims.C
        else:
            n_channels = 3  # Default to RGB

        logger.debug(f"Image dimensions: {height}x{width}x{n_channels}")

        # Calculate estimated size
        total_pixels = height * width * n_channels
        estimated_gb = (total_pixels * 2) / (1024**3)  # Assuming uint16

        # Always use dask for large images to avoid Bio-Formats 2GB limit
        # Threshold: 500MB to be safe
        needs_dask = estimated_gb > 0.5

        if needs_dask:
            logger.info(
                f"Image is large ({estimated_gb:.2f} GB estimated), using dask for tile-based reading..."
            )

            # Use dask_data for lazy loading - this reads in chunks without loading the full plane
            dask_data = img_obj.dask_data
            logger.debug(
                f"Dask array shape: {dask_data.shape}, chunks: {dask_data.chunks}"
            )

            # Extract the 2D+channels data based on dimension order
            if dims_order == "TCZYXS" and shape[-1] == 3:
                # Shape is (T, C, Z, Y, X, S) where S is samples (RGB)
                # Take first T, first C, first Z
                data_slice = dask_data[0, 0, 0, :, :, :]  # Result: (Y, X, S)
            elif dims_order == "TCZYX" or dims_order == "CZYX":
                # Standard case - need to transpose to YXC
                if dims_order == "TCZYX":
                    data_slice = dask_data[0, :, 0, :, :]  # (C, Y, X)
                else:
                    data_slice = dask_data[:, 0, :, :]  # (C, Y, X)
                # Will need to transpose after compute
                data_slice = data_slice.transpose(1, 2, 0)  # (Y, X, C)
            else:
                # Try generic approach
                logger.warning(
                    f"Unrecognized dims order: {dims_order}, attempting generic extraction"
                )
                data_slice = dask_data
                # Squeeze out singleton dimensions
                while data_slice.ndim > 3:
                    data_slice = data_slice[0]

            # Compute in chunks to avoid memory issues
            # Use smaller tile size for memory efficiency
            tile_size = 4096  # Process 4096x4096 tiles at a time
            n_tiles_y = int(np.ceil(height / tile_size))
            n_tiles_x = int(np.ceil(width / tile_size))
            total_tiles = n_tiles_y * n_tiles_x

            logger.info(
                f"Reading {total_tiles} tiles ({n_tiles_y}x{n_tiles_x}) of size {tile_size}x{tile_size}..."
            )

            # Get dtype from dask array
            dtype = dask_data.dtype

            # Allocate output array
            img = np.zeros((height, width, n_channels), dtype=dtype)
            logger.debug(f"Allocated output array with dtype {dtype}")

            # Read tiles using dask
            for i in range(n_tiles_y):
                for j in range(n_tiles_x):
                    y_start = i * tile_size
                    y_end = min((i + 1) * tile_size, height)
                    x_start = j * tile_size
                    x_end = min((j + 1) * tile_size, width)

                    tile_num = i * n_tiles_x + j + 1
                    if tile_num % 10 == 0 or tile_num == 1 or tile_num == total_tiles:
                        logger.info(
                            f"Reading tile {tile_num}/{total_tiles} (Y:{y_start}-{y_end}, X:{x_start}-{x_end})"
                        )

                    # Extract tile from dask array and compute
                    tile_dask = data_slice[y_start:y_end, x_start:x_end, :]
                    tile = tile_dask.compute()

                    # Place tile in output array
                    img[y_start:y_end, x_start:x_end, :] = tile

            logger.info("Finished reading all tiles via dask")
        else:
            logger.debug("Image is small enough to read in one pass")
            # Get image data in YXC format (original logic for small images)
            if dims_order == "TCZYXS" and shape[-1] == 3:
                other_dims = shape[:-3]
                # Use dask to avoid loading all at once
                dask_data = img_obj.dask_data
                img = dask_data[tuple([0] * len(other_dims)) + (Ellipsis,)].compute()
            else:
                img = img_obj.get_image_data("YXC", T=0, Z=0)

        logger.debug(f"Loaded image with shape: {img.shape}")
        return img

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"AICSImage failed to load VSI file: {error_msg}")

        # Check if this is the 2GB limit error
        if "Image plane too large" in error_msg:
            # Try fallback methods in order: Bio-Formats direct, bfio, then OpenSlide
            last_error = e

            # Try Bio-Formats direct first (same Java library, but with proper tile reading)
            if JPYPE_AVAILABLE:
                logger.info(
                    "Attempting fallback to Bio-Formats direct for tile-based reading..."
                )
                try:
                    return load_vsi_with_bioformats_direct(input_path)
                except Exception as bf_error:
                    logger.warning(f"Bio-Formats direct fallback failed: {bf_error}")
                    last_error = bf_error

            # Try bfio as second fallback
            if BFIO_AVAILABLE:
                logger.info("Attempting fallback to bfio for tile-based reading...")
                try:
                    return load_vsi_with_bfio(input_path)
                except Exception as bfio_error:
                    logger.warning(f"bfio fallback failed: {bfio_error}")
                    last_error = bfio_error

            # Try OpenSlide as third fallback (won't work for VSI but try anyway)
            if OPENSLIDE_AVAILABLE:
                logger.info(
                    "Attempting fallback to OpenSlide for tile-based reading..."
                )
                try:
                    return load_vsi_with_openslide(input_path)
                except Exception as openslide_error:
                    logger.warning(f"OpenSlide fallback failed: {openslide_error}")
                    last_error = openslide_error

            # All fallbacks failed
            logger.error(f"All fallback methods failed. Last error: {last_error}")
            raise last_error
        else:
            logger.error(f"Failed to load VSI file: {e}")
            raise


def save_as_tiff_with_tifffile(image, output_path):
    """
    Save image as TIFF using tifffile.

    Args:
        image: numpy array (H x W x C)
        output_path: Path to output TIFF file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        # Save as TIFF with compression
        tifffile.imwrite(
            str(output_path),
            image,
            compression="jpeg",
            compressionargs={"level": 90},
            photometric="rgb",
            tile=(256, 256),
            metadata={"axes": "YXC"},
        )
        return True
    except Exception as e:
        logger.error(f"tifffile save failed: {e}")
        return False


def save_as_tiff_with_pyvips(image, output_path):
    """
    Save image as pyramidal TIFF using pyvips.

    Args:
        image: numpy array (H x W x C)
        output_path: Path to output TIFF file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        # Convert numpy array to pyvips image
        height, width, bands = image.shape
        vips_image = pyvips.Image.new_from_memory(
            image.tobytes(), width, height, bands, "uchar"
        )

        # Save as pyramidal TIFF with compression
        vips_image.tiffsave(
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
        logger.error(f"pyvips save failed: {e}")
        return False


def convert_single_vsi(input_path, output_path, force=False):
    """
    Convert a single VSI file to TIFF.

    Args:
        input_path: Path to input VSI file
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

    try:
        # Load VSI file
        image = load_vsi_image(input_path)

        # Try to save with available methods
        success = False

        if PYVIPS_AVAILABLE:
            logger.debug("Attempting to save with pyvips (pyramidal TIFF)")
            success = save_as_tiff_with_pyvips(image, output_path)

        if not success and TIFFFILE_AVAILABLE:
            logger.debug("Attempting to save with tifffile")
            success = save_as_tiff_with_tifffile(image, output_path)

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
                "reason": "save_failed",
                "message": "All save methods failed",
            }

    except Exception as e:
        return {
            "status": "error",
            "reason": "conversion_failed",
            "message": f"Conversion failed: {str(e)}",
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
        description="Convert VSI files to TIFF format for Trident preprocessing"
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
        help="Directory containing VSI files",
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

    # Check required libraries
    if not AICSIMAGE_AVAILABLE:
        logger.error(
            "AICSImage not available. Install with: pip install aicsimageio bioformats_jar"
        )
        sys.exit(1)

    if not TIFFFILE_AVAILABLE and not PYVIPS_AVAILABLE:
        logger.error("No TIFF writing library available. Install tifffile or pyvips.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("VSI to TIFF Converter for Trident Preprocessing")
    logger.info("=" * 70)
    logger.info(f"Metadata: {args.metadata}")
    logger.info(f"Image directory: {args.image_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Force overwrite: {args.force}")
    logger.info(f"Test mode: {args.test}")
    logger.info(f"AICSImage available: {AICSIMAGE_AVAILABLE}")
    logger.info(f"jpype available: {JPYPE_AVAILABLE}")
    logger.info(f"bfio available: {BFIO_AVAILABLE}")
    logger.info(f"OpenSlide available: {OPENSLIDE_AVAILABLE}")
    logger.info(f"tifffile available: {TIFFFILE_AVAILABLE}")
    logger.info(f"pyvips available: {PYVIPS_AVAILABLE}")
    logger.info("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")

    # Load metadata
    meta_df = load_metadata(args.metadata)

    # Filter for VSI files
    image_dir = Path(args.image_dir)
    vsi_records = []

    logger.info("Scanning for VSI files...")
    for idx, row in meta_df.iterrows():
        img_label = row["Image Label"]
        lims_id = row["prep_lims_id"]

        vsi_path = image_dir / f"{img_label}.vsi"
        if vsi_path.exists():
            vsi_records.append(
                {
                    "image_label": img_label,
                    "prep_lims_id": lims_id,
                    "input_path": vsi_path,
                    "output_path": output_dir / f"{lims_id}.tiff",
                }
            )

    logger.info(f"Found {len(vsi_records)} VSI files to convert")

    if len(vsi_records) == 0:
        logger.warning("No VSI files found. Exiting.")
        sys.exit(0)

    # Test mode: only convert first file
    if args.test:
        logger.info("TEST MODE: Converting only the first file")
        vsi_records = vsi_records[:1]

    # Convert files
    logger.info(f"\nConverting {len(vsi_records)} VSI files...")
    results = []

    for record in tqdm(vsi_records, desc="Converting VSI files"):
        logger.info(
            f"\nProcessing: {record['image_label']} -> {record['prep_lims_id']}"
        )

        result = convert_single_vsi(
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
