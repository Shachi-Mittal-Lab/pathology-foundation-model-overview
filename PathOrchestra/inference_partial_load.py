import os
import argparse
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as T

# Funke Lab Tools
import daisy
from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi

# Models
from networks.seg import DCISmodel

import logging
logging.getLogger("daisy").setLevel(logging.WARNING)

# PathOrchestra expects ImageNet normalization
inp_transforms_rgb = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> str:
    """
    Loads a checkpoint into the model with automatic compatibility handling.

    Tries strict load first (new architecture checkpoint).
    Falls back to strict=False (old architecture checkpoint) and reports
    exactly which keys are missing or unexpected so you know what's
    randomly initialized vs. restored.

    Returns a string describing what was loaded, for logging.
    """
    raw = torch.load(checkpoint_path, map_location=device)

    # support both bare state dicts and wrapped checkpoints
    # e.g. {"model_state_dict": ..., "epoch": ..., "loss": ...}
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        meta = {k: v for k, v in raw.items() if k != "model_state_dict"}
    else:
        state_dict = raw
        meta = {}

    # --- attempt strict load (new architecture checkpoint) ---
    try:
        model.load_state_dict(state_dict, strict=True)
        msg = "✅ Strict load succeeded — checkpoint matches current architecture exactly."
        if meta:
            msg += f"\n   Checkpoint metadata: {meta}"
        return msg

    except RuntimeError:
        pass  # fall through to partial load

    # --- partial load (old architecture checkpoint) ---
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    lines = ["⚠️  Partial load (strict=False) — old architecture checkpoint detected."]

    if missing:
        lines.append("\n   Missing keys (new components — randomly initialized):")
        for k in missing:
            lines.append(f"     • {k}")

    if unexpected:
        lines.append("\n   Unexpected keys (old components — ignored):")
        for k in unexpected:
            lines.append(f"     • {k}")

    # summarise what DID load cleanly
    loaded_keys = [k for k in state_dict if k not in unexpected and k in dict(model.named_parameters())]
    lines.append(f"\n   Successfully restored: {len(loaded_keys)} / {len(state_dict)} keys from checkpoint.")

    if meta:
        lines.append(f"   Checkpoint metadata: {meta}")

    # warn if decoder components critical to segmentation are missing
    critical = [k for k in missing if any(tag in k for tag in ["head", "bottleneck", "dec2", "dec1", "stem", "f6_proj"])]
    if critical:
        lines.append("\n   ⚠️  Critical decoder keys are randomly initialized — predictions will be unreliable:")
        for k in critical:
            lines.append(f"     • {k}")

    return "\n".join(lines)


def model_prediction_rgb(
    mask: Array,
    s2_array: Array,
    patch_size_final: tuple,
    model: torch.nn.Module,
    device: torch.device,
    task: str,
    pred_save_path: str = None,
):
    logit_log = {}  # store logits per block location

    def process_block(block: daisy.Block):

        inslices = s2_array._Array__slices(block.read_roi)
        patch = s2_array[inslices]  # shape: (H, W, C)
        img = Image.fromarray(patch)

        input_tensor = inp_transforms_rgb(img).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)

            # record mean logit per class for this block
            mean_logits = preds.mean(dim=(0, 2, 3)).cpu().numpy().tolist()
            block_key = str(block.read_roi)
            logit_log[block_key] = mean_logits

            preds = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()

        mask[block.write_roi] = preds

    # model expects 224×224 pixel patches
    block_roi = Roi((0, 0), (224, 224)) * s2_array.voxel_size
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

    # save logits to json after inference
    log_path = Path(pred_save_path).parent / "logit_log.json"
    with open(log_path, "w") as f:
        json.dump(logit_log, f, indent=2)
    print(f"Logits saved to: {log_path}")
    print(f"Mask saved to: {pred_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathOrchestra blockwise inference")

    parser.add_argument("--hf_token",       type=str,   required=True,      help="HuggingFace token for PathOrchestra")
    parser.add_argument("--zarr_path",      type=str,   required=True,      help="Path to input.zarr file")
    parser.add_argument("--save_path",      type=str,   required=True,      help="Path to save output mask.zarr")
    parser.add_argument("--scale",          type=str,   default="s1",       help="Zarr scale key: s0=40x, s1=20x, s2=10x, s3=5x")
    parser.add_argument("--checkpoint",     type=str,   default=None,       help="Path to model checkpoint (optional)")
    parser.add_argument("--num_classes",    type=int,   default=3,          help="Number of segmentation classes")
    parser.add_argument("--patch_size",     type=int,   default=224,        help="Patch size for inference")
    parser.add_argument("--task",           type=str,   default="dcis_seg", help="Daisy task name")
    parser.add_argument("--device",         type=str,   default="auto",     help="Device: 'cpu', 'cuda', or 'auto'")

    args = parser.parse_args()

    # --- Convert paths (fixes & and [] special characters) ---
    zarr_path = Path(args.zarr_path.strip())
    save_path = Path(args.save_path.strip())

    print(f"zarr_path:    {zarr_path}")
    print(f"save_path:    {save_path}")
    print(f"Path exists:  {zarr_path.exists()}")

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

    # --- Load checkpoint with compatibility handling ---
    if args.checkpoint and os.path.exists(args.checkpoint):
        report = load_checkpoint(model, args.checkpoint, device)
        print(report)

        # verify head weights as a sanity check
        head_w = model.decoder.head.weight
        print(f"\n  head weight mean: {head_w.mean().item():.6f}")
        print(f"  head weight std:  {head_w.std().item():.6f}")
    else:
        print("⚠️  No checkpoint provided — running with randomly initialized decoder.")
        print("    Predictions will be meaningless. Pass --checkpoint to load trained weights.")

    model.eval().to(device)

    # --- Load image array ---
    s2_array = open_ds(zarr_path / "raw" / args.scale)
    print(f"\nLoaded array: raw/{args.scale}")
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
