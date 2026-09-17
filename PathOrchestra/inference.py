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
# from networks.simple_decoder.models import PathOrchestraSimpleDecoder

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


def roi_to_box(roi: Roi):
    """Convert daisy Roi to (minx, miny, maxx, maxy) bounding box."""
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
    overlap: int = 56               # overlap in pixels
):
    # Calculate write size from overlap in pixels
    write_size = patch_size_final[0] - overlap

    def process_block(block: daisy.Block):
        
        inslices = img_array._Array__slices(block.read_roi)
        patch = img_array[inslices]  # shape: (H, W, C)
        # print(f"Input image shape for pt preds: {patch.shape}")
        img = Image.fromarray(patch)

        input = inp_transforms_rgb(img).unsqueeze(0).to(device)
        # print(f"Final input dimensions: {input.shape}")

        # model prediction
        with torch.no_grad():
            preds = model(input)
            # print(f"Raw logits min/max: {preds.min().item():.4f} / {preds.max().item():.4f}")
            # print(f"Raw logits per class: {preds.mean(dim=(0,2,3))}")

            # # record mean logit per class for this block
            # mean_logits = preds.mean(dim=(0, 2, 3)).cpu().numpy().tolist()
            # block_key = str(block.read_roi)
            # logit_log[block_key] = mean_logits

            preds = torch.argmax(preds, dim=1)
            # print(f"Unique predicted classes: {preds.unique()}")
            preds = preds.squeeze(0)
            preds = preds.cpu().numpy()

        mask[block.write_roi] = preds

    # model expects 224x224 pixels patches
    block_roi = Roi((0, 0), patch_size_final) * img_array.voxel_size # read 224x224
    write_roi = Roi((0, 0), (write_size, write_size)) * img_array.voxel_size  # write with overlap

    # Determine total ROI based on annotations
    if anno_polygon is not None:
        minx, miny, maxx, maxy = anno_polygon.bounds
        total_roi = Roi((miny, minx), (maxy - miny, maxx - minx))
        print(f"Restricting inference to annotation bounds: {total_roi}")
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

    # # save logits to json after inference
    # log_path = Path(pred_save_path).parent / "logit_log.json"
    # with open(log_path, "w") as f:
    #     json.dump(logit_log, f, indent=2)
    # print(f"Logits saved to: {log_path}")

    print(f"Mask saved to: {pred_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathOrchestra blockwise inference")

    parser.add_argument("--hf_token",       type=str,   required=True,          help="HuggingFace token for PathOrchestra")
    parser.add_argument("--zarr_path",      type=str,   required=True,          help="Path to input.zarr file")
    parser.add_argument("--save_path",      type=str,   required=True,          help="Path to save output mask.zarr")
    parser.add_argument("--scale",          type=str,   default="s1",           help="Zarr scale key: s0=40x, s1=20x, s2=10x, s3=5x")
    parser.add_argument("--annotations",    type=str,   default=None,           help="Path to annotations file (.geojson) to exclude background (optional)")
    parser.add_argument("--anno_format",    type=str,   default="geojson",      help="Annotation format: geojson, shapefile, etc.")
    parser.add_argument("--checkpoint",     type=str,   default=None,           help="Path to model checkpoint (optional)")
    parser.add_argument("--num_classes",    type=int,   default=5,              help="Number of segmentation classes")
    parser.add_argument("--patch_size",     type=int,   default=224,            help="Patch size for inference")
    parser.add_argument("--overlap",        type=int,   default=56,             help="Overlap in pixels between blocks")
    parser.add_argument("--task",           type=str,   default="dcis_seg",     help="Daisy task name")
    parser.add_argument("--device",         type=str,   default="auto",         help="Device: 'cpu', 'cuda', or 'auto'")

    args = parser.parse_args()

    # --- Convert paths to pathlib.Path (fixes & and [] special characters) ---
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
            anno_polygon = load_annotations(
                anno_path,
                format=args.anno_format,
                return_type="union"
            )
            print(f"Loaded annotations from: {anno_path}")
            print(f"  Annotation bounds: {anno_polygon.bounds}")
        else:
            print(f"Warning: Annotation file not found: {anno_path}")

    # --- Load model ---
    # model = PathOrchestraDPT(
    #     hf_token=args.hf_token,
    #     num_classes=args.num_classes,
    #     freeze_encoder=True,
    # )

    model = PathOrchestraDPT(
            hf_token=args.hf_token,
            num_classes=args.num_classes,
            hidden_dim=256,
            freeze_encoder=True,
        )

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        # verify head weights loaded correctly
        head_w = model.decoder.head.weight
        print(f"Loaded checkpoint from {args.checkpoint}")
        print(f"  head weight mean: {head_w.mean().item():.6f}")
        print(f"  head weight std:  {head_w.std().item():.6f}")
    else:
        print("No checkpoint found — running with randomly initialized decoder.")

    model.eval().to(device)
    summary(model, input_size=(1, 3, 224, 224), device=device)

    # # --- Decoder output sanity check ---
    # dummy = torch.randn(1, 3, 224, 224).to(device)
    # with torch.no_grad():
    #     logits = model(dummy)
    #     print(f"logits — mean: {logits.mean():.4f}, std: {logits.std():.4f}")
    #     print(f"per-class mean: {logits.mean(dim=(0,2,3))}")
    #     probs = torch.softmax(logits, dim=1)
    #     print(f"per-class prob: {probs.mean(dim=(0,2,3))}")

    # dummy = torch.randn(1, 3, 224, 224).to(device)
    # with torch.no_grad():
    #     skips, stem_skips = model.encoder(dummy)
    #     for i, s in enumerate(skips):
    #         print(f"f{[6,12,18,24][i]} — mean: {s.mean():.4f}, std: {s.std():.4f}")
    #     s1, s2 = stem_skips
    #     print(f"s1 — mean: {s1.mean():.4f}, std: {s1.std():.4f}")
    #     print(f"s2 — mean: {s2.mean():.4f}, std: {s2.std():.4f}")

    # --- Load image array (pathlib fixes & and special chars in path) ---
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
    )
