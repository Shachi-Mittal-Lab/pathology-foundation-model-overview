import random
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from networks.seg import DCISmodel
from data.dataset import HEdataset
from utils import DiceLoss


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Checkpoint directory ---
    ckpt_path = Path(cfg.get("checkpoint_path") or "checkpoints/best_model.pth") # will create a checkpoints folder if none is provided
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_path.parent}")

    # --- Data split (file-level, deterministic) ---
    all_files = sorted(Path(cfg["data_root"]).glob("*.npz"))
    if len(all_files) == 0:
        raise ValueError(f"No.npz files found in {cfg['data_root']}")

    rng = random.Random(42)
    rng.shuffle(all_files)
    n_val       = int(len(all_files) * 0.15)
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"Train: {len(train_files)} tiles | Val: {len(val_files)} tiles")

    train_ds = HEdataset(cfg["data_root"], split="train", file_list=train_files)
    val_ds   = HEdataset(cfg["data_root"], split="val",   file_list=val_files)

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=4, pin_memory=True)

    # --- Model ---
    model = DCISmodel(
        hf_token=cfg["hf_token"],
        num_classes=cfg["num_classes"],
        freeze_encoder=cfg["freeze_encoder"],
    ).to(device)

    params      = model.decoder.parameters() if cfg["freeze_encoder"] else model.parameters()
    optimizer   = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) # TODO

    ce_loss     = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss   = DiceLoss(n_classes=cfg["num_classes"], ignore_index=255)
    ce_weight   = cfg.get("ce_weight",   0.5) # TODO: define in the configurations file (default.yaml)
    dice_weight = cfg.get("dice_weight", 0.5) # TODO: define in the configurations file (default.yaml)

    best_val_loss = float("inf")

    for epoch in range(cfg["epochs"]):

        # --- Training ---
        model.train()
        train_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits    = model(imgs)
            loss_ce   = ce_loss(logits, masks)
            loss_dice = dice_loss(logits, masks, softmax=True)
            loss      = ce_weight * loss_ce + dice_weight * loss_dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                logits    = model(imgs)
                loss_ce   = ce_loss(logits, masks)
                loss_dice = dice_loss(logits, masks, softmax=True)
                val_loss += (ce_weight * loss_ce + dice_weight * loss_dice).item()
        val_loss /= len(val_dl)

        scheduler.step() # TODO: please make sure this works

        print(f"Epoch {epoch+1}/{cfg['epochs']} "
              f"— train loss: {train_loss:.4f} "
              f"— val loss: {val_loss:.4f}")

        # --- Save best checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Best checkpoint saved → {ckpt_path.name} (val loss: {val_loss:.4f})")

        # TODO: save a checkpoint model every nth epoch --> can define n in the configurations file

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"All checkpoints saved in: {ckpt_path.parent.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)