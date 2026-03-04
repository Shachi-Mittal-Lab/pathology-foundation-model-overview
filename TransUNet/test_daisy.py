import argparse
import logging
import os
import random
import sys
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# Funke Lab Tools
import daisy
from funlib.persistence import Array
from funlib.geometry import Coordinate, Roi

# utils
inp_transforms_rgb = T.Compose(
    [
        T.ToTensor(),
    ]
)

def model_prediction_rgb(
    mask: Array,
    s2_array: Array, # this is the 10x array (s0 = 40x, s1 = 20x, s2 = 10x etc.)
    patch_size_final: Array,
    model: torch.nn.Module,
    device: torch.device,
    task: str,
    pred_save_path: str = None,
):

    def process_block(block: daisy.Block):
        # in data slice
        inslices = s2_array._Array__slices(block.read_roi)
        # it was [:, 0:512, 0:512]
        # we want [0:512, 0:512, :] --> grabbing in voxel units
        inslices = (inslices[1], inslices[2], inslices[0]) # PIL expects (H, W, C)
        img = Image.fromarray(s2_array[inslices])
        print(f"Input image shape for pt preds: {s2_array[inslices].shape}")

        # apply normalization + tensor conversion
        input = inp_transforms_rgb(img).unsqueeze(0).to(device) # look into inp_transforms_rgb
        print(f"Final input dimenstions {input.shape}")
        
        # model prediction
        with torch.no_grad():
            preds = model(input)                 # [1, C, H, W]
            preds = torch.argmax(preds, dim=1)   # choose class per pixel → [1, H, W]
            preds = preds.squeeze(0)             # remove batch dim → [H, W]
            preds = preds.cpu().numpy()          # move to CPU numpy
        mask[block.write_roi] = preds

    pred_task = daisy.Task(
        task,
        total_roi=s2_array.roi,
        read_roi=Roi((0, 0), patch_size_final),  # (offset, shape)
        write_roi=Roi((0, 0), patch_size_final),
        read_write_conflict=False,
        num_workers=2,
        process_function=process_block,
    )
    daisy.run_blockwise(tasks=[pred_task], multiprocessing=False)

    # Save the mask after inference if path is provided
    if pred_save_path:
        import zarr
        # Create a new zarr file with the mask data
        zarr.save_array(pred_save_path, mask.data)
        print(f"Mask saved to: {pred_save_path}")

    return