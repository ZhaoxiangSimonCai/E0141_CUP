import os
import logging
from pathlib import Path

# Suppress SLF4J warnings from Java-based image readers (bioformats)
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.slf4j.simpleLogger.defaultLogLevel=off"

import cv2
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from tqdm import tqdm
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ============ Configuration ============
# Paths
META_DATA_PATH = "/home/scai/scratch/E0141_CUP/data/E0141_combined_meta.xlsx"
IMAGE_BASE_PATH = "/mnt/histopathology/E0141/P01"
OUT_PATH = "/home/scai/scratch/E0141_CUP/data/images/patches_256window_hsv/"

# Patch parameters
PATCH_SIZE = 256
SCALE = 1

# Tissue detection method: "hsv" or "simple"
TISSUE_DETECTION_METHOD = "hsv"  # Options: "hsv" (recommended) or "simple" (original)

# Tissue detection thresholds (HSV-based)
SATURATION_THRESHOLD = 15  # Minimum mean saturation for tissue
VALUE_THRESHOLD = 200  # Maximum mean value (brightness) for tissue
STD_THRESHOLD = 20  # Minimum std dev for texture variation

# Simple method thresholds (original approach)
SIMPLE_MIN_INTENSITY = 100  # Minimum mean intensity
SIMPLE_MAX_INTENSITY = 200  # Maximum mean intensity

# ============ Performance Configuration ============
# Toggle each optimization independently
ENABLE_MULTIPROCESSING = True  # Parallel slide processing with joblib
ENABLE_GPU = True  # GPU acceleration for tissue detection
ENABLE_VECTORIZATION = True  # Vectorized patch extraction

# Multiprocessing settings
NUM_WORKERS = 8  # Number of parallel workers (8-10 recommended for 15GB images)
# Lower if you hit RAM limits (each worker loads 1 image)

# GPU settings (32GB GPU memory)
GPU_BATCH_SIZE = 2000  # Patches per GPU batch for tissue detection
# Lower if GPU OOM errors (each 256x256x3 patch = 196KB)
# 2000 patches ≈ 400MB + overhead

# Vectorization settings (1TB RAM)
MAX_PATCHES_IN_MEMORY = 50000  # Maximum patches to extract at once
# For 15GB image: ~200k patches total
# Process in chunks to stay safe
# 50k patches ≈ 3GB RAM

# Try to import GPU libraries if enabled
if ENABLE_GPU:
    try:
        import cupy as cp

        GPU_AVAILABLE = True
        logger.info("GPU acceleration enabled with CuPy")
    except ImportError:
        GPU_AVAILABLE = False
        logger.warning("CuPy not available, falling back to CPU processing")
        ENABLE_GPU = False
else:
    GPU_AVAILABLE = False
    logger.info("GPU acceleration disabled")

# ============ Functions ============


def is_tissue_patch_simple(
    patch, min_intensity=SIMPLE_MIN_INTENSITY, max_intensity=SIMPLE_MAX_INTENSITY
):
    """
    Original simple method: Determine if a patch contains tissue using grayscale intensity.

    This is the original approach that checks if mean intensity falls within a range.
    Background tends to be very bright (high intensity) and completely black areas
    are typically artifacts, so tissue falls in the middle range.

    Args:
        patch: RGB patch image (H x W x C)
        min_intensity: Minimum mean grayscale intensity (0-255)
        max_intensity: Maximum mean grayscale intensity (0-255)

    Returns:
        bool: True if patch likely contains tissue
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    mean_intensity = gray.mean()
    logger.debug(f"Mean intensity: {mean_intensity}")
    return min_intensity < mean_intensity < max_intensity


def is_tissue_patch_hsv(
    patch,
    saturation_threshold=SATURATION_THRESHOLD,
    value_threshold=VALUE_THRESHOLD,
    std_threshold=STD_THRESHOLD,
):
    """
    Advanced method: Determine if a patch contains tissue using HSV color space analysis.

    Background in histology slides is typically white/very light (low saturation,
    high brightness), while tissue has color from H&E staining.

    Args:
        patch: RGB patch image (H x W x C)
        saturation_threshold: Minimum mean saturation for tissue (0-255)
        value_threshold: Maximum mean value/brightness for tissue (0-255)
        std_threshold: Minimum standard deviation for texture

    Returns:
        bool: True if patch likely contains tissue
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)

    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # Calculate metrics
    mean_saturation = hsv[:, :, 1].mean()
    mean_value = hsv[:, :, 2].mean()
    std_dev = gray.std()

    # Tissue should have:
    # - Higher saturation (colored by stains)
    # - Lower brightness (not white background)
    # - Higher standard deviation (texture variation)
    is_tissue = (
        mean_saturation > saturation_threshold
        and mean_value < value_threshold
        and std_dev > std_threshold
    )

    return is_tissue


def is_tissue_patch(patch):
    """
    Wrapper function that calls the appropriate tissue detection method
    based on TISSUE_DETECTION_METHOD configuration.

    Args:
        patch: RGB patch image (H x W x C)

    Returns:
        bool: True if patch likely contains tissue
    """
    if TISSUE_DETECTION_METHOD == "simple":
        return is_tissue_patch_simple(patch)
    elif TISSUE_DETECTION_METHOD == "hsv":
        return is_tissue_patch_hsv(patch)
    else:
        raise ValueError(
            f"Unknown tissue detection method: {TISSUE_DETECTION_METHOD}. "
            f"Choose 'simple' or 'hsv'."
        )


def is_tissue_patch_batch_simple_gpu(
    patches_batch,
    min_intensity=SIMPLE_MIN_INTENSITY,
    max_intensity=SIMPLE_MAX_INTENSITY,
):
    """
    GPU-accelerated batch version: Determine if patches contain tissue using grayscale intensity.

    Args:
        patches_batch: Batch of RGB patches (N x H x W x C)
        min_intensity: Minimum mean grayscale intensity (0-255)
        max_intensity: Maximum mean grayscale intensity (0-255)

    Returns:
        ndarray: Boolean array indicating which patches contain tissue
    """
    # Transfer to GPU
    patches_gpu = cp.asarray(patches_batch)

    # Convert RGB to grayscale using standard weights
    # Gray = 0.299*R + 0.587*G + 0.114*B
    gray_gpu = (
        0.299 * patches_gpu[:, :, :, 0]
        + 0.587 * patches_gpu[:, :, :, 1]
        + 0.114 * patches_gpu[:, :, :, 2]
    )

    # Calculate mean intensity for each patch
    mean_intensity = cp.mean(gray_gpu, axis=(1, 2))

    # Apply threshold
    is_tissue = (mean_intensity > min_intensity) & (mean_intensity < max_intensity)

    # Transfer back to CPU
    return cp.asnumpy(is_tissue)


def is_tissue_patch_batch_hsv_gpu(
    patches_batch,
    saturation_threshold=SATURATION_THRESHOLD,
    value_threshold=VALUE_THRESHOLD,
    std_threshold=STD_THRESHOLD,
):
    """
    GPU-accelerated batch version: Determine if patches contain tissue using HSV analysis.

    Args:
        patches_batch: Batch of RGB patches (N x H x W x C)
        saturation_threshold: Minimum mean saturation for tissue (0-255)
        value_threshold: Maximum mean value/brightness for tissue (0-255)
        std_threshold: Minimum standard deviation for texture

    Returns:
        ndarray: Boolean array indicating which patches contain tissue
    """
    # Transfer to GPU
    patches_gpu = cp.asarray(patches_batch, dtype=cp.uint8)

    # Convert each patch from RGB to HSV
    # CuPy doesn't have cv2.cvtColor, so we'll do it on CPU in batches
    # For very large batches, this is still faster than single-patch processing
    hsv_patches = np.array(
        [cv2.cvtColor(patch, cv2.COLOR_RGB2HSV) for patch in patches_batch]
    )
    hsv_gpu = cp.asarray(hsv_patches)

    # Calculate metrics
    mean_saturation = cp.mean(hsv_gpu[:, :, :, 1], axis=(1, 2))
    mean_value = cp.mean(hsv_gpu[:, :, :, 2], axis=(1, 2))

    # Convert to grayscale for texture analysis
    gray_gpu = (
        0.299 * patches_gpu[:, :, :, 0]
        + 0.587 * patches_gpu[:, :, :, 1]
        + 0.114 * patches_gpu[:, :, :, 2]
    )
    std_dev = cp.std(gray_gpu, axis=(1, 2))

    # Apply thresholds
    is_tissue = (
        (mean_saturation > saturation_threshold)
        & (mean_value < value_threshold)
        & (std_dev > std_threshold)
    )

    # Transfer back to CPU
    return cp.asnumpy(is_tissue)


def is_tissue_patch_batch_cpu(patches_batch):
    """
    CPU batch version: Process multiple patches using the configured method.

    Args:
        patches_batch: Batch of RGB patches (N x H x W x C)

    Returns:
        ndarray: Boolean array indicating which patches contain tissue
    """
    results = np.array([is_tissue_patch(patch) for patch in patches_batch])
    return results


def is_tissue_patch_batch(patches_batch):
    """
    Batch tissue detection wrapper that uses GPU if available, otherwise CPU.

    Args:
        patches_batch: Batch of RGB patches (N x H x W x C)

    Returns:
        ndarray: Boolean array indicating which patches contain tissue
    """
    if ENABLE_GPU and GPU_AVAILABLE:
        try:
            if TISSUE_DETECTION_METHOD == "simple":
                return is_tissue_patch_batch_simple_gpu(patches_batch)
            elif TISSUE_DETECTION_METHOD == "hsv":
                return is_tissue_patch_batch_hsv_gpu(patches_batch)
        except Exception as e:
            logger.warning(f"GPU processing failed: {e}, falling back to CPU")
            return is_tissue_patch_batch_cpu(patches_batch)
    else:
        return is_tissue_patch_batch_cpu(patches_batch)


def extract_patches_vectorized(img, patch_size):
    """
    Extract all patches from an image using vectorized operations.

    Args:
        img: Image array (H x W x C)
        patch_size: Size of patches to extract

    Returns:
        Patches array (N x patch_size x patch_size x C)
        num_patches_rows: Number of patches in row direction
        num_patches_cols: Number of patches in column direction
    """
    h, w, c = img.shape

    # Compute the number of patches in each direction
    num_patches_rows = h // patch_size
    num_patches_cols = w // patch_size

    # Crop image to fit exact number of patches
    img_cropped = img[: num_patches_rows * patch_size, : num_patches_cols * patch_size]

    # Reshape to extract patches
    # From (H, W, C) to (num_rows, patch_size, num_cols, patch_size, C)
    patches = img_cropped.reshape(
        num_patches_rows, patch_size, num_patches_cols, patch_size, c
    )

    # Transpose to (num_rows, num_cols, patch_size, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)

    # Reshape to (N, patch_size, patch_size, C)
    patches = patches.reshape(-1, patch_size, patch_size, c)

    return patches, num_patches_rows, num_patches_cols


def extract_and_filter_patches(img, patch_size):
    """
    Extract patches and filter for tissue content.

    Handles both vectorized and non-vectorized approaches based on configuration.
    For very large images, processes in chunks to avoid memory issues.

    Args:
        img: Image array (H x W x C)
        patch_size: Size of patches to extract

    Returns:
        tissue_patches: Array of tissue patches (N x patch_size x patch_size x C)
        total_possible: Total number of patches extracted
        tissue_count: Number of tissue patches kept
    """
    h, w, c = img.shape
    num_patches_rows = h // patch_size
    num_patches_cols = w // patch_size
    total_possible = num_patches_rows * num_patches_cols

    if ENABLE_VECTORIZATION and total_possible > 0:
        # Vectorized approach
        all_patches, _, _ = extract_patches_vectorized(img, patch_size)

        tissue_patches = []

        # Process in chunks to avoid memory issues
        num_patches = len(all_patches)
        for start_idx in range(0, num_patches, MAX_PATCHES_IN_MEMORY):
            end_idx = min(start_idx + MAX_PATCHES_IN_MEMORY, num_patches)
            chunk = all_patches[start_idx:end_idx]

            # Further split into GPU batch sizes for processing
            chunk_tissue_patches = []
            for batch_start in range(0, len(chunk), GPU_BATCH_SIZE):
                batch_end = min(batch_start + GPU_BATCH_SIZE, len(chunk))
                batch = chunk[batch_start:batch_end]

                # Get tissue mask
                tissue_mask = is_tissue_patch_batch(batch)

                # Keep only tissue patches
                chunk_tissue_patches.append(batch[tissue_mask])

            if chunk_tissue_patches:
                tissue_patches.extend(chunk_tissue_patches)

        # Concatenate all tissue patches
        if tissue_patches:
            tissue_patches = np.concatenate(tissue_patches, axis=0)
        else:
            tissue_patches = np.array([]).reshape(0, patch_size, patch_size, c)

    else:
        # Non-vectorized approach (original method)
        tissue_patches = []
        for i in range(num_patches_rows):
            for j in range(num_patches_cols):
                row_start = i * patch_size
                row_end = row_start + patch_size
                col_start = j * patch_size
                col_end = col_start + patch_size

                # Extract the patch
                patch = img[row_start:row_end, col_start:col_end]

                # Check if patch contains tissue
                if is_tissue_patch(patch):
                    tissue_patches.append(patch)

        tissue_patches = (
            np.array(tissue_patches)
            if tissue_patches
            else np.array([]).reshape(0, patch_size, patch_size, c)
        )

    tissue_count = len(tissue_patches)
    return tissue_patches, total_possible, tissue_count


def process_single_slide(row):
    """
    Process a single slide: load image, extract patches, filter for tissue, and save.

    Args:
        row: Pandas Series containing slide metadata (Image Label, prep_lims_id)

    Returns:
        dict: Processing statistics for this slide
    """
    img_label = row["Image Label"]
    lims_id = row["prep_lims_id"]

    # Try both .czi and .vsi file extensions
    file_path_czi = Path(IMAGE_BASE_PATH) / f"{img_label}.czi"
    file_path_vsi = Path(IMAGE_BASE_PATH) / f"{img_label}.vsi"

    # Determine which file exists
    file_path = None
    file_format = None
    if file_path_czi.exists():
        file_path = file_path_czi
        file_format = "czi"
    elif file_path_vsi.exists():
        file_path = file_path_vsi
        file_format = "vsi"

    output_file = Path(OUT_PATH) / f"{lims_id}.npy"

    # Skip if file doesn't exist
    if file_path is None:
        logger.debug(f"Skipping {img_label}: File not found (.czi or .vsi)")
        return {
            "status": "skipped",
            "reason": "not_found",
            "total_patches": 0,
            "tissue_patches": 0,
        }

    # Skip if already processed
    if output_file.exists():
        logger.debug(f"Skipping {img_label}: Already processed")
        return {
            "status": "skipped",
            "reason": "already_processed",
            "total_patches": 0,
            "tissue_patches": 0,
        }

    # Load image with error handling
    try:
        logger.debug(f"Loading {img_label} as {file_format} file")
        img_obj = AICSImage(str(file_path))
        # Get image data in YXC format (Y=height, X=width, C=channels)
        # Select first scene/timepoint/z-stack if multiple exist
        # AICSImage handles both CZI and VSI formats and normalizes dimensions
        # Get all scenes (which are your RGB channels)
        if img_obj.dims.order == "TCZYXS" and img_obj.shape[-1] == 3:
            # Last dimension is S=3, those are your RGB channels
            other_dims = img_obj.data.shape[:-3]
            img = img_obj.data[tuple([0] * len(other_dims)) + (Ellipsis,)]  # H x W x C
            logger.debug(f"RGB image shape: {img.shape}")
        else:
            # Use normal method
            img = img_obj.get_image_data("YXC", T=0, Z=0, S=0)
            logger.debug(f"Image shape: {img.shape}")
    except Exception as e:
        logger.error(f"Failed to read {img_label}: {e}")
        return {
            "status": "failed",
            "reason": "load_error",
            "total_patches": 0,
            "tissue_patches": 0,
            "error": str(e),
        }

    try:
        # img is now in YXC format (height x width x channels)
        h, w, c = img.shape

        logger.info(f"Processing {img_label} ({lims_id}) [{file_format}]: {w}x{h}x{c}")

        # Apply scaling if needed
        if SCALE != 1:
            img = cv2.resize(img, (int(w * SCALE), int(h * SCALE)))
            h, w = img.shape[:2]

        # Extract patches and filter for tissue using optimized method
        patches, total_possible, tissue_count = extract_and_filter_patches(
            img, PATCH_SIZE
        )

        # Save patches
        np.save(str(output_file), patches)

        tissue_ratio = tissue_count / total_possible if total_possible > 0 else 0
        logger.info(
            f"  Saved {tissue_count}/{total_possible} patches "
            f"({tissue_ratio:.1%} tissue content)"
        )

        return {
            "status": "success",
            "total_patches": total_possible,
            "tissue_patches": tissue_count,
            "img_label": img_label,
            "lims_id": lims_id,
        }

    except Exception as e:
        logger.error(f"Error processing {img_label}: {e}")
        return {
            "status": "failed",
            "reason": "processing_error",
            "total_patches": 0,
            "tissue_patches": 0,
            "error": str(e),
        }


def main():
    """Main execution function."""
    # Create output directory
    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)

    # Load metadata
    logger.info(f"Loading metadata from {META_DATA_PATH}")
    meta_df = pd.read_excel(META_DATA_PATH)

    logger.info(f"Found {len(meta_df)} slides to process")

    # Log configuration
    logger.info(f"Configuration: patch_size={PATCH_SIZE}, scale={SCALE}")
    logger.info(f"Output path: {OUT_PATH}")
    logger.info(f"Tissue detection method: {TISSUE_DETECTION_METHOD}")
    if TISSUE_DETECTION_METHOD == "hsv":
        logger.info(
            f"  HSV thresholds: saturation>{SATURATION_THRESHOLD}, "
            f"value<{VALUE_THRESHOLD}, std>{STD_THRESHOLD}"
        )
    elif TISSUE_DETECTION_METHOD == "simple":
        logger.info(
            f"  Simple thresholds: {SIMPLE_MIN_INTENSITY} < intensity < {SIMPLE_MAX_INTENSITY}"
        )

    # Log performance configuration
    logger.info(f"\nPerformance Configuration:")
    logger.info(
        f"  Multiprocessing: {ENABLE_MULTIPROCESSING} (workers={NUM_WORKERS if ENABLE_MULTIPROCESSING else 1})"
    )
    logger.info(f"  GPU acceleration: {ENABLE_GPU and GPU_AVAILABLE}")
    logger.info(f"  Vectorization: {ENABLE_VECTORIZATION}")
    if ENABLE_GPU and GPU_AVAILABLE:
        logger.info(f"  GPU batch size: {GPU_BATCH_SIZE}")
    if ENABLE_VECTORIZATION:
        logger.info(f"  Max patches in memory: {MAX_PATCHES_IN_MEMORY}")

    # Process slides
    if ENABLE_MULTIPROCESSING and len(meta_df) > 1:
        logger.info(
            f"\nProcessing {len(meta_df)} slides in parallel with {NUM_WORKERS} workers..."
        )

        # Use joblib for parallel processing
        results = Parallel(n_jobs=NUM_WORKERS, backend="loky")(
            delayed(process_single_slide)(row)
            for _, row in tqdm(
                meta_df.iterrows(), total=len(meta_df), desc="Processing slides"
            )
        )
    else:
        logger.info(f"\nProcessing {len(meta_df)} slides sequentially...")
        results = []
        for _, row in tqdm(
            meta_df.iterrows(), total=len(meta_df), desc="Processing slides"
        ):
            result = process_single_slide(row)
            results.append(result)

    # Aggregate statistics
    stats = {
        "slides_processed": sum(1 for r in results if r["status"] == "success"),
        "slides_skipped": sum(1 for r in results if r["status"] == "skipped"),
        "slides_failed": sum(1 for r in results if r["status"] == "failed"),
        "total_patches": sum(r["total_patches"] for r in results),
        "tissue_patches": sum(r["tissue_patches"] for r in results),
    }

    # ============ Summary ============
    logger.info("\n" + "=" * 50)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Slides processed successfully: {stats['slides_processed']}")
    logger.info(f"Slides skipped (not found/already done): {stats['slides_skipped']}")
    logger.info(f"Slides failed: {stats['slides_failed']}")
    logger.info(f"Total patches extracted: {stats['total_patches']}")
    logger.info(f"Tissue patches kept: {stats['tissue_patches']}")
    if stats["total_patches"] > 0:
        overall_ratio = stats["tissue_patches"] / stats["total_patches"]
        logger.info(f"Overall tissue ratio: {overall_ratio:.1%}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
