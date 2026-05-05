import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.segmentation_model import BreastSegModel
from data.dataset import BreastPatchDataset
import yaml, argparse

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BreastPatchDataset(cfg["data_root"], split="train")
    n_val   = int(len(dataset) * 0.15)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=4, pin_memory=True)

    model = BreastSegModel(
        hf_token=cfg["hf_token"],
        num_classes=cfg["num_classes"],
        freeze_encoder=cfg["freeze_encoder"],
    ).to(device)

    # Only optimize decoder params if encoder is frozen
    params = model.decoder.parameters() if cfg["freeze_encoder"] else model.parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        model.train()
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)                    # (B, C, H, W)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation loop (add Dice/IoU via utils/metrics.py)
        print(f"Epoch {epoch+1}/{cfg['epochs']} — loss: {loss.item():.4f}")

    torch.save(model.state_dict(), cfg["checkpoint_path"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)