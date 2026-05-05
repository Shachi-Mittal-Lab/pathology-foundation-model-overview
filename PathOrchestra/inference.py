import os
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
    T.Resize(224),                                        # ensure correct patch size
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),               # ImageNet stats
                std=(0.229, 0.224, 0.225)),
])


def model_prediction_rgb(
    mask: Array,
    s2_array: Array,          # 10x array (s0=40x, s1=20x, s2=10x...)
    patch_size_final: tuple,
    model: torch.nn.Module,
    device: torch.device,
    task: str,
    pred_save_path: str = None,
):
    def process_block(block: daisy.Block):
        # in data slcie
        inslices = s2_array._Array__slices(block.read_roi)
        inslices = (inslices[1], inslices[2], inslices[0])  # (H, W, C) for PIL
        img = Image.fromarray(s2_array[inslices])
        print(f"Input image shape for pt preds: {s2_array[inslices].shape}")

        # apply normalization + tensor conversion
        input = inp_transforms_rgb(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        print(f"Final input dimensions: {input.shape}")

        # model prediction
        with torch.no_grad():
            preds = model(input)                # (1, num_classes, H, W)
            preds = torch.argmax(preds, dim=1)  # (1, H, W)
            preds = preds.squeeze(0)            # (H, W)
            preds = preds.cpu().numpy()

        mask[block.write_roi] = preds

    pred_task = daisy.Task(
        task,
        total_roi=s2_array.roi,
        read_roi=Roi((0, 0), patch_size_final), # (offsett, shape)
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
    # --- Config ---
    HF_TOKEN       = "your_hf_token"
    CHECKPOINT     = "checkpoints/breast_seg.pth"
    ZARR_PATH      = "path/to/your/image.zarr"
    PRED_SAVE_PATH = "path/to/output_mask.zarr"
    PATCH_SIZE     = (224, 224)
    NUM_CLASSES    = 3
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    model = DCISmodel(
        hf_token=HF_TOKEN,
        num_classes=NUM_CLASSES,
        freeze_encoder=True,
    )

    # Only load checkpoint if one exists
    if CHECKPOINT and os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        print(f"Loaded checkpoint from {CHECKPOINT}")
    else:
        print("No checkpoint found — running with randomly initialized decoder.")

    model.eval().to(DEVICE)

    # --- Load image array ---
    s2_array = open_ds(ZARR_PATH, "s2")   # adjust scale key as needed

    # --- Prepare output mask array ---
    mask = prepare_ds(
        PRED_SAVE_PATH,
        shape=s2_array.shape[:2],         # (H, W) — no channel dim for class map
        voxel_size=s2_array.voxel_size,
        dtype=np.uint8,                   # class indices 0/1/2 fit in uint8
        roi=s2_array.roi,
    )

    # --- Run blockwise inference ---
    model_prediction_rgb(
        mask=mask,
        s2_array=s2_array,
        patch_size_final=PATCH_SIZE,
        model=model,
        device=DEVICE,
        task="breast_seg_inference",
        pred_save_path=PRED_SAVE_PATH,
    )