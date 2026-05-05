import os
import argparse
import numpy as np
from PIL import Image
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
        inslices = (inslices[1], inslices[2], inslices[0])
        img = Image.fromarray(s2_array[inslices])
        print(f"Input image shape for pt preds: {s2_array[inslices].shape}")

        input = inp_transforms_rgb(img).unsqueeze(0).to(device)
        print(f"Final input dimensions: {input.shape}")

        with torch.no_grad():
            preds = model(input)
            preds = torch.argmax(preds, dim=1)
            preds = preds.squeeze(0)
            preds = preds.cpu().numpy()

        mask[block.write_roi] = preds

    pred_task = daisy.Task(
        task,
        total_roi=s2_array.roi,
        read_roi=Roi((0, 0), patch_size_final),
        write_roi=Roi((0, 0), patch_size_final),
        read_write_conflict=False,
        num_workers=2,
        process_function=process_block,
    )
    daisy.run_blockwise(tasks=[pred_task], multiprocessing=False)

    if pred_save_path:
        zarr.save_array(pred_save_path, mask.data)
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

    # --- Load image array ---
    s2_array = open_ds(args.zarr_path, f"raw/{args.scale}")
    print(f"Loaded array: raw/{args.scale}")
    print(f"  shape:      {s2_array.shape}")
    print(f"  voxel_size: {s2_array.voxel_size}")
    print(f"  roi:        {s2_array.roi}")

    # --- Prepare output mask ---
    mask = prepare_ds(
        args.save_path,
        shape=s2_array.shape[:2],
        voxel_size=s2_array.voxel_size,
        dtype=np.uint8,
        roi=s2_array.roi,
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
        pred_save_path=args.save_path,
    )