import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from funlib.persistence import open_ds
from utils_annotations import load_annotations


def visualize_inference_region(
    zarr_path: str,
    scale: str = "s1",
    annotations_path: str = None,
    figsize: tuple = (16, 12),
    downsample: int = 4
):
    """
    Visualize the H&E image with annotation bounds overlaid.
    
    Args:
        zarr_path: Path to input.zarr
        scale: Scale key (s0, s1, s2, s3)
        annotations_path: Path to annotations.geojson (optional)
        figsize: Figure size
        downsample: Downsample factor for faster rendering
    """
    zarr_path = Path(zarr_path)
    
    # Load image
    img_array = open_ds(zarr_path / "raw" / scale)
    print(f"Loading image: {scale}")
    print(f"  Shape: {img_array.shape}")
    print(f"  Voxel size: {img_array.voxel_size}")
    
    # Load full image (downsampled for speed)
    img_data = img_array[::downsample, ::downsample, :]
    print(f"  Downsampled shape: {img_data.shape}")
    
    # Normalize to 0-1
    if img_data.max() > 1:
        img_data = img_data.astype(np.float32) / 255.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_data)
    
    # Load and overlay annotations if provided
    if annotations_path:
        anno_path = Path(annotations_path)
        if anno_path.exists():
            anno_polygon = load_annotations(
                anno_path,
                format="geojson",
                return_type="union"
            )
            
            minx, miny, maxx, maxy = anno_polygon.bounds
            
            # Scale annotation bounds to match downsampled image
            minx_ds = minx // downsample
            miny_ds = miny // downsample
            maxx_ds = maxx // downsample
            maxy_ds = maxy // downsample
            
            # Draw bounding box
            rect = patches.Rectangle(
                (minx_ds, miny_ds),
                maxx_ds - minx_ds,
                maxy_ds - miny_ds,
                linewidth=3,
                edgecolor='red',
                facecolor='none',
                label='Annotation bounds'
            )
            ax.add_patch(rect)
            
            # Plot annotation boundary
            from shapely.geometry import box
            boundary_x = []
            boundary_y = []
            boundary = anno_polygon.boundary
            if hasattr(boundary, 'coords'):
                coords = list(boundary.coords)
            else:  # MultiLineString
                coords = []
                for geom in boundary.geoms:
                    coords.extend(list(geom.coords))
            
            for x, y in coords:
                boundary_x.append(x // downsample)
                boundary_y.append(y // downsample)
            
            if boundary_x:
                ax.plot(boundary_x, boundary_y, 'r-', linewidth=1, alpha=0.7, label='Annotation polygon')
            
            print(f"\nAnnotation bounds (pixels at {scale}):")
            print(f"  X: {minx_ds:.0f} to {maxx_ds:.0f}")
            print(f"  Y: {miny_ds:.0f} to {maxy_ds:.0f}")
            print(f"  Size: {(maxx_ds - minx_ds):.0f} x {(maxy_ds - miny_ds):.0f}")
            
            ax.legend(loc='upper right', fontsize=12)
        else:
            print(f"Annotation file not found: {anno_path}")
    
    ax.set_title(f'H&E Image with Inference Region ({scale})', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels, downsampled x{})'.format(downsample))
    ax.set_ylabel('Y (pixels, downsampled x{})'.format(downsample))
    
    plt.tight_layout()
    plt.show()
    
    return img_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize inference region")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to input.zarr")
    parser.add_argument("--scale", type=str, default="s1", help="Scale: s0, s1, s2, s3")
    parser.add_argument("--annotations", type=str, default=None, help="Path to annotations.geojson")
    
    args = parser.parse_args()
    
    visualize_inference_region(
        args.zarr_path,
        scale=args.scale,
        annotations_path=args.annotations
    )
