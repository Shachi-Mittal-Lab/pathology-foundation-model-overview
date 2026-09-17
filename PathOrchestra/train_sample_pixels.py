import random
import yaml
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks.dpt.models import PathOrchestraDPT
from data.dataset import HEdataset


def scan_patch_class_pixels(data_root, ignore_index=255):
    """Scan all .npz files and count pixels per class in each patch."""
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise ValueError(f"Data root directory not found: {data_root}")
    
    npz_files = sorted(list(data_root.rglob("*.npz")))
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {data_root} or subdirectories")
    
    patch_class_counts = {}
    
    print(f"Scanning {len(npz_files)} patches for class pixel counts...")
    for npz_file in tqdm(npz_files, desc="Scanning patches"):
        try:
            data = np.load(npz_file)
            if "label" not in data:
                continue
            
            label_array = data["label"]
            unique, counts = np.unique(label_array, return_counts=True)
            
            class_counts = {}
            for cls, count in zip(unique, counts):
                if cls != ignore_index:
                    class_counts[int(cls)] = int(count)
            
            if class_counts:
                patch_class_counts[str(npz_file)] = class_counts
        except Exception as e:
            print(f"Warning: Error reading {npz_file.name}: {e}")
            continue
    
    if len(patch_class_counts) == 0:
        raise ValueError(f"No valid patches found in {data_root}")
    
    print(f"Successfully loaded {len(patch_class_counts)} patches\n")
    return patch_class_counts


def select_balanced_patches(patch_class_counts, target_pixels_per_class=500000):
    """Greedily select patches to reach target pixel count per class."""
    all_classes = set()
    for class_dict in patch_class_counts.values():
        all_classes.update(class_dict.keys())
    
    all_classes = sorted(list(all_classes))
    print(f"\nFound {len(all_classes)} unique classes: {all_classes}")
    
    selected_patches = []
    class_pixel_totals = defaultdict(int)
    
    patch_paths = list(patch_class_counts.keys())
    random.seed(42)
    random.shuffle(patch_paths)
    
    for patch_path in tqdm(patch_paths, desc="Selecting balanced patches"):
        class_counts = patch_class_counts[patch_path]
        
        needs_pixels = False
        for cls in all_classes:
            if class_pixel_totals[cls] < target_pixels_per_class:
                needs_pixels = True
                break
        
        if not needs_pixels:
            break
        
        selected_patches.append(patch_path)
        for cls, count in class_counts.items():
            class_pixel_totals[cls] += count
    
    print(f"\nSelected {len(selected_patches)} / {len(patch_paths)} patches")
    print("\nClass pixel totals:")
    for cls in all_classes:
        total = class_pixel_totals[cls]
        pct = (total / target_pixels_per_class * 100) if target_pixels_per_class > 0 else 0
        print(f"  Class {cls}: {total:,} pixels ({pct:.1f}% of target {target_pixels_per_class:,})")
    
    return selected_patches, dict(class_pixel_totals)


def compute_patch_weights(file_list, ignore_index=255):
    """
    Compute sampling weights for patches based on class frequency.
    Patches containing rare classes get higher weight.
    
    INPUTS:
    - file_list (list): List of .npz file paths
    - ignore_index (int): Label value to ignore (default 255)
    
    OUTPUTS:
    - weights (np.ndarray): Shape (len(file_list),), normalized sampling weights
    """
    class_patch_count = defaultdict(int)
    
    print("\nComputing patch weights based on class frequency...")
    for npz_file in tqdm(file_list, desc="Computing weights"):
        try:
            data = np.load(npz_file)
            if "label" not in data:
                continue
            
            label_array = data["label"]
            unique_classes = set(np.unique(label_array)) - {ignore_index}
            
            for cls in unique_classes:
                class_patch_count[cls] += 1
        except Exception as e:
            continue
    
    if not class_patch_count:
        print("Warning: No classes found, using uniform weights")
        return np.ones(len(file_list))
    
    # Compute weight per patch based on rarest class it contains
    weights = []
    for npz_file in file_list:
        try:
            data = np.load(npz_file)
            if "label" not in data:
                weights.append(1.0)
                continue
            
            label_array = data["label"]
            unique_classes = set(np.unique(label_array)) - {ignore_index}
            
            if not unique_classes:
                weights.append(0.1)  # Mostly background
                continue
            
            # Weight = inverse of frequency (rare classes → higher weight)
            patch_weight = 1.0 / max([class_patch_count[cls] for cls in unique_classes])
            weights.append(patch_weight)
        except Exception as e:
            weights.append(1.0)
            continue
    
    # Normalize so sum = 1 (required by WeightedRandomSampler)
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"✓ Patch weights computed\n")
    return weights


def print_model_summary(model, device):
    """Print model architecture and parameter counts."""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nTotal Parameters:       {total_params:>15,}")
    print(f"Trainable Parameters:   {trainable_params:>15,}")
    print(f"Frozen Parameters:      {frozen_params:>15,}")
    print(f"Trainable %:            {(trainable_params/total_params*100):>14.2f}%")
    
    model_size_mb = total_params * 4 / (1024 ** 2)
    print(f"Model Size (float32):   {model_size_mb:>14.2f} MB")
    
    print("\n" + "="*70 + "\n")


def plot_loss_curve(train_losses, val_losses, output_path):
    """Plot train and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Loss curve saved → {output_path}")
    plt.close()


def get_trainable_parameters(model):
    """Get trainable parameters (all non-encoder parameters)."""
    trainable_params = []
    for name, param in model.named_parameters():
        if 'encoder' not in name and param.requires_grad:
            trainable_params.append(param)
    return trainable_params


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Checkpoint directory ---
    ckpt_path = Path(cfg.get("checkpoint_path") or "checkpoints/best_model.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_path.parent}")

    # --- Scan patches and select balanced subset (greedy) ---
    patch_class_counts = scan_patch_class_pixels(cfg["data_root"])
    target_pixels = cfg.get("target_pixels_per_class", 500000)
    selected_patches, class_totals = select_balanced_patches(patch_class_counts, target_pixels)
    
    if len(selected_patches) == 0:
        raise ValueError("No patches selected for training!")
    
    # --- Data split (file-level, deterministic) ---
    rng = random.Random(42)
    rng.shuffle(selected_patches)
    n_val       = int(len(selected_patches) * 0.15)
    val_patches = selected_patches[:n_val]
    train_patches = selected_patches[n_val:]
    print(f"\nTrain: {len(train_patches)} patches | Val: {len(val_patches)} patches")

    train_ds = HEdataset(cfg["data_root"], split="train", file_list=train_patches)
    val_ds   = HEdataset(cfg["data_root"], split="val",   file_list=val_patches)

    # --- Compute patch weights for training data (class balancing) ---
    train_weights = compute_patch_weights(train_patches)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_patches),
        replacement=True
    )

    # Use sampler for training (no shuffle), regular DataLoader for validation
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=train_sampler,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=0, pin_memory=True)

    # --- Model ---
    model = PathOrchestraDPT(
        hf_token=cfg["hf_token"],
        num_classes=cfg["num_classes"],
        features=cfg.get("features", 256),
        freeze_encoder=True,
    ).to(device)

    print_model_summary(model, device)

    # --- Get trainable parameters ---
    trainable_params = get_trainable_parameters(model)
    lr = float(cfg["lr"])
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # --- Loss function ---
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    # --- Gradient accumulation settings ---
    accumulation_steps = cfg.get("accumulation_steps", 2)
    effective_batch_size = cfg["batch_size"] * accumulation_steps
    print(f"Gradient accumulation: {accumulation_steps} steps")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Using WeightedRandomSampler for class balancing\n")

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    plot_freq = cfg.get("plot_freq", 5)
    checkpoint_freq = cfg.get("checkpoint_freq", 10)
    plot_path = ckpt_path.parent / "loss_curve.png"

    for epoch in range(cfg["epochs"]):

        # --- Training ---
        model.train()
        train_loss = 0.0
        accum_loss = 0.0
        batch_count = 0

        with tqdm(total=len(train_dl), desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", leave=False) as pbar:
            for batch_idx, (imgs, masks) in enumerate(train_dl):
                imgs, masks = imgs.to(device), masks.to(device)
                
                # Forward pass
                logits = model(imgs)
                loss = ce_loss(logits, masks)
                
                # Backward pass (accumulate gradients)
                loss.backward()
                accum_loss += loss.item()
                batch_count += 1
                
                # Optimizer step after accumulation
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dl):
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    train_loss += accum_loss / batch_count
                    accum_loss = 0.0
                    batch_count = 0
                
                pbar.update(1)
        
        train_loss /= (len(train_dl) // accumulation_steps + 1)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(val_dl), desc=f"Epoch {epoch+1}/{cfg['epochs']} [Val]", leave=False) as pbar:
                for imgs, masks in val_dl:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs)
                    loss = ce_loss(logits, masks)
                    val_loss += loss.item()
                    pbar.update(1)
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{cfg['epochs']} "
              f"— train loss: {train_loss:.4f} "
              f"— val loss: {val_loss:.4f} "
              f"— lr: {current_lr:.2e}")

        # --- Save best checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Best checkpoint saved → {ckpt_path.name} (val loss: {val_loss:.4f})")

        # --- Save periodic checkpoint every N epochs ---
        if (epoch + 1) % checkpoint_freq == 0:
            periodic_ckpt = ckpt_path.parent / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), periodic_ckpt)
            print(f"  ✓ Periodic checkpoint saved → checkpoint_epoch_{epoch+1}.pth")

        # --- Update loss plot ---
        if (epoch + 1) % plot_freq == 0 or epoch == cfg["epochs"] - 1:
            plot_loss_curve(train_losses, val_losses, plot_path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"All checkpoints saved in: {ckpt_path.parent.resolve()}")
    print(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
