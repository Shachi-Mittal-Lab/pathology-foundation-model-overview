import os
import argparse
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torchinfo import summary
import torchvision.transforms as T
import zarr

# Funke Lab 
import daisy
from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi

# model
from networks.dpt.models import PathOrchestraDPT

from utils_annotations import load_annotations

import logging
logging.getLogger("daisy").setLevel(logging.WARNING)

# PathOrchestra expects ImageNet normalization
inp_transforms_rgb = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

# Map scales to their downsample factor relative to s0
SCALE_FACTORS = {"s0": 1, "s1": 2, "s2": 4, "s3": 8}


def roi_to_box(roi: Roi):
    """Convert daisy Roi to shapely box (minx, miny, maxx, maxy)."""
    begin = roi.get_begin()
    end = roi.get_end()
    return (begin[1], begin[0], end[1], end[0])  # (minx, miny, maxx, maxy)


def model_prediction_rgb(
    mask: Array,
    img_array: Array,
    patch_size_final: tuple,
    model: torch.nn.Module,
    device: torch.device,
    task: str,
    anno_polygon=None,
    pred_save_path: str = None,
    overlap: int = 56,
    overlap_threshold: float = 0.1  # 10% overlap minimum
):
    """
    Blockwise inference using daisy.
    
    Args:
        mask: Output mask array
        img_array: Input image array
        patch_size_final: Patch size tuple (224, 224)
        model: Model for inference
        device: Device (cpu/cuda)
        task: Daisy task name
        anno_polygon: Optional annotation polygon to restrict inference
        pred_save_path: Path to save output
        overlap: Pixel overlap between blocks
        overlap_threshold: Minimum overlap percentage (0-1) with annotation to predict
    """

    def process_block(block: daisy.Block):
        # Check overlap with annotation if provided
        if anno_polygon is not None:
            from shapely.geometry import box
            
            # Convert block roi to pixel coordinates (divide by voxel_size)
            block_begin = block.read_roi.get_begin()
            block_end = block.read_roi.get_end()
            
            block_minx = block_begin[1] / img_array.voxel_size[1]
            block_miny = block_begin[0] / img_array.voxel_size[0]
            block_maxx = block_end[1] / img_array.voxel_size[1]
            block_maxy = block_end[0] / img_array.voxel_size[0]
            
            block_box = box(block_minx, block_miny, block_maxx, block_maxy)
            
            # Calculate intersection and overlap percentage
            intersection = anno_polygon.intersection(block_box)
            overlap_pct = intersection.area / block_box.area if block_box.area > 0 else 0
            
            # Skip block if overlap is below threshold
            if overlap_pct < overlap_threshold:
                mask[block.write_roi] = 0  # Mark as background
                return
        
        inslices = img_array._Array__slices(block.read_roi)
        patch = img_array[inslices]  # shape: (H, W, C)
        img = Image.fromarray(patch)

        input = inp_transforms_rgb(img).unsqueeze(0).to(device)

        # model prediction
        with torch.no_grad():
            preds = model(input)
            preds = torch.argmax(preds, dim=1)
            preds = preds.squeeze(0)
            preds = preds.cpu().numpy()

        mask[block.write_roi] = preds

    # model expects 224x224 pixels patches
    block_roi = Roi((0, 0), patch_size_final) * img_array.voxel_size  # read 224x224
    write_size = patch_size_final[0] - overlap
    write_roi = Roi((0, 0), (write_size, write_size)) * img_array.voxel_size  # write with overlap

    # Determine total ROI based on annotations
    if anno_polygon is not None:
        minx, miny, maxx, maxy = anno_polygon.bounds
        # Convert pixel bounds to world coordinates
        total_roi = Roi(
            (miny * img_array.voxel_size[0], minx * img_array.voxel_size[1]),
            ((maxy - miny) * img_array.voxel_size[0], (maxx - minx) * img_array.voxel_size[1])
        )
        print(f"Restricting inference to annotation bounds")
        print(f"  Annotation ROI (pixels): X={minx:.0f}-{maxx:.0f}, Y={miny:.0f}-{maxy:.0f}")
        print(f"  World ROI: {total_roi}")
        print(f"  Overlap threshold: {overlap_threshold*100:.1f}%")
    else:
        total_roi = img_array.roi

    pred_task = daisy.Task(
        task,
        total_roi=total_roi,
        read_roi=block_roi,
        write_roi=write_roi,
        read_write_conflict=False,
        num_workers=2,
        process_function=process_block,
    )
    daisy.run_blockwise(tasks=[pred_task], multiprocessing=False)

    print(f"Mask saved to: {pred_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathOrchestra blockwise inference")

    parser.add_argument("--hf_token",           type=str,   required=True,          help="HuggingFace token for PathOrchestra")
    parser.add_argument("--zarr_path",          type=str,   required=True,          help="Path to input.zarr file")
    parser.add_argument("--save_path",          type=str,   required=True,          help="Path to save output mask.zarr")
    parser.add_argument("--scale",              type=str,   default="s1",           help="Zarr scale key: s0, s1, s2, s3")
    parser.add_argument("--annotations",        type=str,   default=None,           help="Path to annotations file (.geojson)")
    parser.add_argument("--anno_format",        type=str,   default="geojson",      help="Annotation format: geojson, ...")
    parser.add_argument("--overlap_threshold",  type=float, default=0.1,            help="Min overlap % with annotation (0-1, default 0.1 = 10%)")
    parser.add_argument("--checkpoint",         type=str,   default=None,           help="Path to model checkpoint (optional)")
    parser.add_argument("--num_classes",        type=int,   default=5,              help="Number of segmentation classes")
    parser.add_argument("--patch_size",         type=int,   default=224,            help="Patch size for inference")
    parser.add_argument("--overlap",            type=int,   default=0,             help="Pixel overlap between blocks")
    parser.add_argument("--task",               type=str,   default="dcis_seg",     help="Daisy task name")
    parser.add_argument("--device",             type=str,   default="auto",         help="Device: 'cpu', 'cuda', or 'auto'")

    args = parser.parse_args()

    # --- Convert paths to pathlib.Path ---
    zarr_path = Path(args.zarr_path.strip())
    save_path = Path(args.save_path.strip())

    print(f"zarr_path: {zarr_path}")
    print(f"save_path: {save_path}")
    print(f"Path exists: {zarr_path.exists()}")

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Load annotations if provided ---
    anno_polygon = None
    if args.annotations:
        anno_path = Path(args.annotations.strip())
        if anno_path.exists():
            # Load at s0
            anno_polygon = load_annotations(
                anno_path,
                format=args.anno_format,
                return_type="union"
            )
            
            # Rescale to match inference scale
            scale_factor = SCALE_FACTORS.get(args.scale, 1)
            if scale_factor > 1:
                minx, miny, maxx, maxy = anno_polygon.bounds
                minx /= scale_factor
                miny /= scale_factor
                maxx /= scale_factor
                maxy /= scale_factor
                
                from shapely.geometry import box
                anno_polygon = box(minx, miny, maxx, maxy)
                print(f"Scaled annotations by factor {scale_factor} to match {args.scale}")
            
            print(f"Loaded annotations from: {anno_path}")
            print(f"  Annotation bounds (pixels): {anno_polygon.bounds}")
        else:
            print(f"Warning: Annotation file not found: {anno_path}")

    # --- Load model ---
    model = PathOrchestraDPT(
        hf_token=args.hf_token,
        num_classes=args.num_classes,
        features=256,
        freeze_encoder=True,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint found — running with randomly initialized decoder.")

    model.eval().to(device)
    summary(model, input_size=(1, 3, 224, 224), device=device)

    # --- Load image array ---
    img_array = open_ds(zarr_path / "raw" / args.scale)
    print(f"Loaded array: raw/{args.scale}")
    print(f"  shape:      {img_array.shape}")
    print(f"  voxel_size: {img_array.voxel_size}")
    print(f"  roi:        {img_array.roi}")

    # --- Prepare output mask ---
    mask = prepare_ds(
        save_path,
        shape=img_array.shape[:2],
        offset=img_array.offset,
        voxel_size=img_array.voxel_size,
        axis_names=["y", "x"],
        units=img_array.units,
        dtype=np.uint8,
        mode="w",
    )

    # --- Run blockwise inference ---
    patch_size = (args.patch_size, args.patch_size)
    model_prediction_rgb(
        mask=mask,
        img_array=img_array,
        patch_size_final=patch_size,
        model=model,
        device=device,
        task=args.task,
        anno_polygon=anno_polygon,
        pred_save_path=save_path,
        overlap=args.overlap,
        overlap_threshold=args.overlap_threshold,
    )
