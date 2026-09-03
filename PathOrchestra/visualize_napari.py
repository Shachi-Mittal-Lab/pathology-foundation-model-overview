import argparse
import numpy as np
from pathlib import Path
import napari
from funlib.persistence import open_ds


def visualize_napari(pred_path, original_path=None, scale="s1"):
    """
    Visualize predictions and original image in napari with per-class coloring.
    
    Args:
        pred_path: Path to predictions zarr
        original_path: Path to original image zarr (optional)
        scale: Zarr scale key (e.g., "s0", "s1", "s2", "s3")
    """
    # Convert to Path objects
    pred_path = Path(pred_path)
    if original_path:
        original_path = Path(original_path)
    
    # Load predictions
    print(f"Loading predictions from {pred_path}...")
    pred_array = open_ds(pred_path)
    preds = pred_array[:]
    
    print(f"Predictions shape: {preds.shape}")
    
    # Check unique classes and counts
    unique_classes = np.unique(preds)
    print(f"Unique classes: {unique_classes}")
    
    for cls in unique_classes:
        count = np.sum(preds == cls)
        pct = 100.0 * count / preds.size
        print(f"  Class {cls}: {count} pixels ({pct:.2f}%)")
    
    # Initialize viewer
    viewer = napari.Viewer()
    
    # Add predictions as labels layer
    # napari automatically assigns different colors to each label value
    viewer.add_labels(preds, name="Predictions", opacity=0.7)
    
    if original_path:
        print(f"\nLoading original image from {original_path}...")
        orig_array = open_ds(original_path / "raw" / scale)
        original = orig_array[:]
        
        print(f"Original shape: {original.shape}")
        print(f"Original dtype: {original.dtype}")
        print(f"Original min: {original.min()}, max: {original.max()}")
        print(f"Original sample values (first pixel): {original[0, 0, :]}")
        
        # DON'T normalize yet - check what values we have
        # if original.max() > 1:
        #     original = original.astype(np.float32) / 255.0
        
        # Add as image layer
        if original.ndim == 3 and original.shape[2] == 3:
            # RGB image - display as-is
            viewer.add_image(original, name="Original", rgb=True)
        else:
            viewer.add_image(original, name="Original", colormap='gray')
    
    print("\nLaunching napari...")
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predictions in napari")
    
    parser.add_argument("--pred_path",      type=str,   required=True,   help="Path to predictions.zarr")
    parser.add_argument("--original_path",  type=str,   default=None,    help="Path to original image.zarr (optional)")
    parser.add_argument("--scale",          type=str,   default="s1",    help="Zarr scale: s0, s1, s2, s3")
    
    args = parser.parse_args()
    
    visualize_napari(
        pred_path=args.pred_path,
        original_path=args.original_path,
        scale=args.scale
    )
