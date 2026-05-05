import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as T
import zarr

# Funke Lab Tools
import daisy
from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi

# Your model
from networks.seg import DCISmodel

# PathOrchestra expects ImageNet normalization
inp_transforms_rgb = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def model_prediction_rgb(
    mask: Array,
    s2_array: Array,
    patch_size_final: tuple,
    model: torch.nn.Module,
    device: torch.device,
    task: str,
    pred_save_path: str = None,
):
    def process_block(block: daisy.Block):
        inslices = s2_array._Array__slices(block.read_roi)
        patch = s2_array[inslices]  # shape: (H, W, C)
        # print(f"Input image shape for pt preds: {patch.shape}")
        img = Image.fromarray(patch)

        input = inp_transforms_rgb(img).unsqueeze(0).to(device)
        # print(f"Final input dimensions: {input.shape}")

        # model prediction
        with torch.no_grad():
            preds = model(input)
            preds = torch.argmax(preds, dim=1)
            preds = preds.squeeze(0)
            preds = preds.cpu().numpy()

        mask[block.write_roi] = preds

    # model expects 224x224 pixels patches
    block_roi = Roi((0, 0), (224, 224)) * s2_array.voxel_size # convert from pixel to world units
    write_roi = Roi((0, 0), (224, 224)) * s2_array.voxel_size

    pred_task = daisy.Task(
        task,
        total_roi=s2_array.roi,
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

    parser.add_argument("--hf_token",       type=str,   required=True,          help="HuggingFace token for PathOrchestra")
    parser.add_argument("--zarr_path",      type=str,   required=True,          help="Path to input.zarr file")
    parser.add_argument("--save_path",      type=str,   required=True,          help="Path to save output mask.zarr")
    parser.add_argument("--scale",          type=str,   default="s1",           help="Zarr scale key: s0=40x, s1=20x, s2=10x, s3=5x")
    parser.add_argument("--checkpoint",     type=str,   default=None,           help="Path to model checkpoint (optional)")
    parser.add_argument("--num_classes",    type=int,   default=3,              help="Number of segmentation classes")
    parser.add_argument("--patch_size",     type=int,   default=224,            help="Patch size for inference")
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

    # --- Load model ---
    model = DCISmodel(
        hf_token=args.hf_token,
        num_classes=args.num_classes,
        freeze_encoder=True,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint found — running with randomly initialized decoder.")

    model.eval().to(device)

    # --- Load image array (pathlib fixes & and special chars in path) ---
    s2_array = open_ds(zarr_path / "raw" / args.scale)
    print(f"Loaded array: raw/{args.scale}")
    print(f"  shape:      {s2_array.shape}")
    print(f"  voxel_size: {s2_array.voxel_size}")
    print(f"  roi:        {s2_array.roi}")

    # --- Prepare output mask ---
    mask = prepare_ds(
        save_path,
        shape=s2_array.shape[:2],
        offset=s2_array.offset,
        voxel_size=s2_array.voxel_size,
        axis_names=["y", "x"],
        units=s2_array.units,
        dtype=np.uint8,
        mode="w",
    )

    # --- Run blockwise inference ---
    patch_size = (args.patch_size, args.patch_size)
    model_prediction_rgb(
        mask=mask,
        s2_array=s2_array,
        patch_size_final=patch_size,
        model=model,
        device=device,
        task=args.task,
        pred_save_path=save_path,
    )