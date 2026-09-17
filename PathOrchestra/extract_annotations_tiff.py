import numpy as np
import tifffile
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import tempfile
import shutil


def load_ome_tif_level(tif_path: str, level: int = 0) -> np.ndarray:
    """
    Load a specific pyramid level from OME-TIFF file.
    
    INPUTS:
    - tif_path (str): path to .ome.tif file
    - level (int): pyramid level to load (0=highest res, 1=2x downsampled, etc.)
    
    OUTPUTS:
    - img (np.ndarray): image array (H x W x C) or (H x W)
    """
    with tifffile.TiffFile(tif_path) as tif:
        if level < len(tif.series):
            series = tif.series[level]
            img = series.asarray()
        else:
            print(f"Warning: level {level} not found. Loading level 0 instead.")
            series = tif.series[0]
            img = series.asarray()
    
    return img


def color_to_label(classifier_img: np.ndarray) -> np.ndarray:
    """
    Convert tissue classifier color image to label mask.
    Blue pixels (0, 0, 255) in RGB format → label 2 (tissue)
    Red (255, 0, 0) and Green (0, 255, 0) → label 255 (IGNORE)
    
    INPUTS:
    - classifier_img (np.ndarray): RGB image from tissue classifier (H x W x 3)
    
    OUTPUTS:
    - label_mask (np.ndarray): Label mask (H x W) with values {2, 255}
    """
    # Handle RGBA → RGB
    if classifier_img.ndim == 3 and classifier_img.shape[2] == 4:
        classifier_img = classifier_img[:, :, :3]
    
    label_mask = np.full(classifier_img.shape[:2], 255, dtype=np.uint8)
    
    # RGB format: channel 0 = R, channel 1 = G, channel 2 = B
    R = classifier_img[:, :, 0].astype(float)
    G = classifier_img[:, :, 1].astype(float)
    B = classifier_img[:, :, 2].astype(float)
    
    # Blue detection: B >> R and B >> G
    blue_mask = (B > R + 20) & (B > G + 20)
    label_mask[blue_mask] = 2
    
    return label_mask


def save_npz_safe(output_path: str, **arrays):
    """
    Safely save .npz file by writing to temp first, then atomic rename.
    Prevents corruption if script crashes mid-write.
    
    INPUTS:
    - output_path (str): final destination path
    - **arrays: keyword arguments with numpy arrays to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file in same directory (ensures same filesystem for atomic rename)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, 
        suffix='.npz', 
        delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Write to temp
        np.savez(tmp_path, **arrays)
        # Atomic rename
        shutil.move(tmp_path, str(output_path))
    except Exception as e:
        # Clean up temp if something failed
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise e


def extract_patches_from_aligned_omes(
    he_tif_path: str,
    classifier_tif_path: str,
    output_dir: str,
    slide_name: str,
    level: int = 0,
    patch_size: int = 224,
    stride: int = 224,
    min_valid_fraction: float = 0.10,
):
    """
    Extract patches from aligned H&E and tissue classifier OME-TIFF files.
    
    INPUTS:
    - he_tif_path (str): path to H&E .ome.tif file
    - classifier_tif_path (str): path to tissue classifier .ome.tif file
    - output_dir (str): directory to save .npz patches
    - slide_name (str): name for output files (e.g., 'slide_001')
    - level (int): pyramid level to extract from (0=highest res, default 0)
    - patch_size (int): size of square patches (default 224)
    - stride (int): step between patches (default 224, no overlap)
    - min_valid_fraction (float): minimum fraction of blue pixels (default 0.10 = 10%)
    
    OUTPUTS:
    - saves .npz files with 'image' and 'label' keys
    - prints summary statistics
    """
    
    print(f"Loading H&E image from {he_tif_path} (level {level})...")
    he_img = load_ome_tif_level(he_tif_path, level)
    print(f"  Shape: {he_img.shape}, dtype: {he_img.dtype}")
    
    print(f"Loading classifier output from {classifier_tif_path} (level {level})...")
    classifier_img = load_ome_tif_level(classifier_tif_path, level)
    print(f"  Shape: {classifier_img.shape}, dtype: {classifier_img.dtype}")
    
    # Validate shapes match
    if he_img.shape[:2] != classifier_img.shape[:2]:
        raise ValueError(
            f"Image shapes don't match: H&E {he_img.shape[:2]} vs "
            f"Classifier {classifier_img.shape[:2]}"
        )
    
    # Convert classifier colors to labels
    print("Converting classifier colors to labels (blue → 2, else → 255)...")
    label_mask = color_to_label(classifier_img)
    
    height, width = he_img.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract patches
    patch_count = 0
    skipped_count = 0
    
    y_positions = list(range(0, height - patch_size + 1, stride))
    x_positions = list(range(0, width - patch_size + 1, stride))
    
    total_patches = len(y_positions) * len(x_positions)
    
    print(f"\nExtracting {patch_size}×{patch_size} patches (stride={stride}px)...")
    print(f"Expected patches: ~{total_patches}")
    print(f"Filtering: only patches with ≥{min_valid_fraction*100:.1f}% blue pixels")
    
    with tqdm(total=total_patches, desc="Extracting patches") as pbar:
        for y in y_positions:
            for x in x_positions:
                # Extract patch regions
                img_patch = he_img[y:y+patch_size, x:x+patch_size]
                mask_patch = label_mask[y:y+patch_size, x:x+patch_size]
                
                # Validate mask has sufficient blue pixels (label 2)
                blue_pixels = (mask_patch == 2)
                blue_fraction = blue_pixels.mean()
                
                if blue_fraction < min_valid_fraction:
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                # Save to .npz (with safe temp-then-rename)
                filename = os.path.join(
                    output_dir, 
                    f'{slide_name}_tile_y{y}_x{x}.npz'
                )
                save_npz_safe(
                    filename,
                    image=img_patch.astype(np.float32),
                    label=mask_patch.astype(np.uint8)
                )
                
                patch_count += 1
                pbar.update(1)
    
    print(f"\n✓ Extraction complete!")
    print(f"  Saved patches: {patch_count}")
    print(f"  Skipped patches (insufficient blue): {skipped_count}")
    print(f"  Output directory: {output_dir}")


def batch_process_folders(
    he_dir: str,
    classifier_dir: str,
    output_dir: str,
    level: int = 0,
    patch_size: int = 224,
    stride: int = 224,
    min_valid_fraction: float = 0.10,
):
    """
    Batch process H&E and classifier pairs from two separate folders.
    """
    
    he_path = Path(he_dir)
    classifier_path = Path(classifier_dir)
    output_path = Path(output_dir)
    
    # Find all H&E files
    he_files = sorted(he_path.glob("*.ome.tif")) + sorted(he_path.glob("*.tif"))
    # Remove duplicates if both .tif and .ome.tif exist
    he_files = list({f.stem: f for f in he_files}.values())
    
    if not he_files:
        print(f"No .ome.tif or .tif files found in {he_dir}")
        return
    
    print(f"Found {len(he_files)} H&E files to process\n")
    
    for i, he_file in enumerate(he_files, 1):
        slide_name = he_file.stem
        
        # Find corresponding classifier file with same name
        classifier_file = classifier_path / he_file.name
        
        if not classifier_file.exists():
            print(
                f"[{i}/{len(he_files)}] ✗ Skipping {slide_name} - "
                f"classifier file not found in {classifier_dir}"
            )
            continue
        
        # Create output subdirectory
        slide_output_dir = output_path / slide_name
        slide_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i}/{len(he_files)}] Processing {slide_name}...")
        
        try:
            extract_patches_from_aligned_omes(
                he_tif_path=str(he_file),
                classifier_tif_path=str(classifier_file),
                output_dir=str(slide_output_dir),
                slide_name=slide_name,
                level=level,
                patch_size=patch_size,
                stride=stride,
                min_valid_fraction=min_valid_fraction,
            )
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 60)
    print("Batch processing complete!")
    print(f"Output saved to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patches from aligned H&E and tissue classifier OME-TIFF files"
    )
    parser.add_argument(
        "--he_dir", type=str, required=True,
        help="Directory containing H&E .ome.tif files"
    )
    parser.add_argument(
        "--classifier_dir", type=str, required=True,
        help="Directory containing tissue classifier .ome.tif files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for .npz patches"
    )
    parser.add_argument(
        "--level", type=int, default=0,
        help="Pyramid level to extract from (0=full res, 1=2x down, etc.). Default: 0"
    )
    parser.add_argument(
        "--patch_size", type=int, default=224,
        help="Size of square patches in pixels. Default: 224"
    )
    parser.add_argument(
        "--stride", type=int, default=224,
        help="Stride between patches in pixels. Default: 224 (no overlap)"
    )
    parser.add_argument(
        "--min_valid_fraction", type=float, default=0.10,
        help="Minimum fraction of blue pixels to keep patch (0.0-1.0). Default: 0.10 (10%%)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.he_dir).exists():
        raise FileNotFoundError(f"H&E directory not found: {args.he_dir}")
    
    if not Path(args.classifier_dir).exists():
        raise FileNotFoundError(f"Classifier directory not found: {args.classifier_dir}")
    
    if args.patch_size <= 0:
        raise ValueError("patch_size must be positive")
    
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    
    if not (0.0 <= args.min_valid_fraction <= 1.0):
        raise ValueError("min_valid_fraction must be between 0.0 and 1.0")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Extract Patches from Aligned OME-TIFF Files")
    print("=" * 60)
    print(f"H&E directory:       {args.he_dir}")
    print(f"Classifier directory: {args.classifier_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Pyramid level:       {args.level}")
    print(f"Patch size:          {args.patch_size}×{args.patch_size} px")
    print(f"Stride:              {args.stride} px")
    print(f"Min blue fraction:   {args.min_valid_fraction*100:.1f}%")
    print("=" * 60)
    
    batch_process_folders(
        he_dir=args.he_dir,
        classifier_dir=args.classifier_dir,
        output_dir=args.output_dir,
        level=args.level,
        patch_size=args.patch_size,
        stride=args.stride,
        min_valid_fraction=args.min_valid_fraction,
    )
