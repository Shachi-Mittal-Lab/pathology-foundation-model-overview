"""
Compare encoder features and decoder logits for background vs tissue patches
for both random and trained models.

Run from the PathOrchestra project root:
    python compare_encoder_decoder.py
"""

import json
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from funlib.persistence import open_ds
from funlib.geometry import Roi

from networks.seg import DCISmodel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
HF_TOKEN      = "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"
ZARR_PATH     = Path(r"E:\[PROJ]_DCIS\PathOrchestra\TEMP_test_inference\logit_comparison\2015003_H&E.zarr")
CHECKPOINT    = Path(r"E:\[PROJ]_DCIS\Decoder_training\05132026_less_background.pth\best_model.pth")
SCALE         = "s2"
NUM_CLASSES   = 4
NUM_SAMPLES   = 2000        # number of patches to sample per model
BG_THRESHOLD  = 200       # brightness threshold for background vs tissue
OUTPUT_DIR    = Path(r"E:\[PROJ]_DCIS\PathOrchestra\TEMP_test_inference")
# ──────────────────────────────────────────────────────────────────────────────

inp_transforms = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def sample_patches(s2_array, n_samples: int):
    """Sample n_samples patches evenly across the slide."""
    H, W, _ = s2_array.shape
    patch_size = 224

    ys = np.linspace(0, H - patch_size, int(np.sqrt(n_samples)), dtype=int)
    xs = np.linspace(0, W - patch_size, int(np.sqrt(n_samples)), dtype=int)

    patches = []
    coords  = []
    for y in ys:
        for x in xs:
            patch = s2_array[y:y+patch_size, x:x+patch_size, :]
            if patch.shape == (patch_size, patch_size, 3):
                patches.append(patch)
                coords.append((int(y), int(x)))

    return patches, coords


def analyze_patch(patch, model, device):
    """Run a single patch through encoder and decoder, return features + logits."""
    img = Image.fromarray(patch)
    x   = inp_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # encoder features
        skips = model.encoder(x)
        f6, f12, f18, f24 = skips

        # decoder logits
        logits = model.decoder(skips)  # (1, num_classes, 224, 224)

    return {
        "brightness":      float(patch.mean()),
        "is_background":   bool(patch.mean() > BG_THRESHOLD),

        # encoder features — mean/std of each skip level
        "f6_mean":         float(f6.mean().item()),
        "f6_std":          float(f6.std().item()),
        "f12_mean":        float(f12.mean().item()),
        "f12_std":         float(f12.std().item()),
        "f18_mean":        float(f18.mean().item()),
        "f18_std":         float(f18.std().item()),
        "f24_mean":        float(f24.mean().item()),
        "f24_std":         float(f24.std().item()),

        # decoder logits — mean per class across all pixels
        "logits":          logits.mean(dim=(0, 2, 3)).cpu().numpy().tolist(),
        "predicted_class": int(logits.argmax(dim=1).flatten().mode().values.item()),
    }


def run_model(model, patches, coords, device, label):
    """Run all patches through the model and return results."""
    results = {}
    for i, (patch, (y, x)) in enumerate(zip(patches, coords)):
        key = f"y{y}_x{x}"
        results[key] = analyze_patch(patch, model, device)
        if (i + 1) % 10 == 0:
            print(f"  [{label}] processed {i+1}/{len(patches)} patches")
    return results


def summarize(results, label):
    """Print summary statistics split by background vs tissue."""
    background = [v for v in results.values() if v["is_background"]]
    tissue     = [v for v in results.values() if not v["is_background"]]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total patches:     {len(results)}")
    print(f"  Background blocks: {len(background)}")
    print(f"  Tissue blocks:     {len(tissue)}")

    for group_name, group in [("BACKGROUND", background), ("TISSUE", tissue)]:
        if len(group) == 0:
            print(f"\n  {group_name}: no patches")
            continue

        logits   = np.array([v["logits"]  for v in group])
        f24_mean = np.array([v["f24_mean"] for v in group])
        f24_std  = np.array([v["f24_std"]  for v in group])
        winners  = np.argmax(logits, axis=1)

        print(f"\n  {group_name} (n={len(group)}):")
        print(f"    Brightness:       mean={np.mean([v['brightness'] for v in group]):.1f}")
        print(f"    Encoder f24 mean: mean={f24_mean.mean():.4f}  std={f24_mean.std():.4f}")
        print(f"    Encoder f24 std:  mean={f24_std.mean():.4f}  std={f24_std.std():.4f}")
        print(f"    Decoder logits:   mean={logits.mean(axis=0)}")
        print(f"    Decoder logit std:{logits.std(axis=0)}")
        print(f"    Class winners:")
        for c in range(logits.shape[1]):
            print(f"      class {c}: {(winners==c).sum():4d} / {len(group)}")

    # encoder feature difference between background and tissue
    if background and tissue:
        bg_f24  = np.array([v["f24_mean"] for v in background])
        tis_f24 = np.array([v["f24_mean"] for v in tissue])
        bg_log  = np.array([v["logits"]   for v in background])
        tis_log = np.array([v["logits"]   for v in tissue])

        print(f"\n  BACKGROUND vs TISSUE DIFFERENCE:")
        print(f"    Encoder f24 mean diff: {tis_f24.mean() - bg_f24.mean():.4f}")
        print(f"    Decoder logit diff:    {tis_log.mean(axis=0) - bg_log.mean(axis=0)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load slide
    s2_array = open_ds(ZARR_PATH / "raw" / SCALE)
    print(f"Slide shape: {s2_array.shape}")

    # sample patches once — same patches for both models
    print(f"\nSampling {NUM_SAMPLES} patches...")
    patches, coords = sample_patches(s2_array, NUM_SAMPLES)
    print(f"Sampled {len(patches)} patches")
    print(f"Brightness range: {np.mean([p.mean() for p in patches]):.1f} mean")

    bg_count  = sum(1 for p in patches if p.mean() > BG_THRESHOLD)
    tis_count = len(patches) - bg_count
    print(f"Background patches: {bg_count} | Tissue patches: {tis_count}")

    # ── RANDOM WEIGHTS MODEL ──────────────────────────────────────────────────
    print("\nLoading model with RANDOM weights...")
    model_random = DCISmodel(
        hf_token=HF_TOKEN,
        num_classes=NUM_CLASSES,
        freeze_encoder=True,
    ).eval().to(device)

    print("Running random weights model...")
    results_random = run_model(model_random, patches, coords, device, "random")

    # save
    out_random = OUTPUT_DIR / "logit_comparison_random.json"
    with open(out_random, "w") as f:
        json.dump(results_random, f, indent=2)
    print(f"Saved: {out_random}")

    # ── TRAINED MODEL ─────────────────────────────────────────────────────────
    print(f"\nLoading model with TRAINED weights from {CHECKPOINT}...")
    model_trained = DCISmodel(
        hf_token=HF_TOKEN,
        num_classes=NUM_CLASSES,
        freeze_encoder=True,
    )
    model_trained.load_state_dict(
        torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    )
    model_trained.eval().to(device)

    print("Running trained model...")
    results_trained = run_model(model_trained, patches, coords, device, "trained")

    # save
    out_trained = OUTPUT_DIR / "logit_comparison_trained.json"
    with open(out_trained, "w") as f:
        json.dump(results_trained, f, indent=2)
    print(f"Saved: {out_trained}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    summarize(results_random,  "RANDOM WEIGHTS")
    summarize(results_trained, "TRAINED WEIGHTS")

    # ── ENCODER COMPARISON (same for both since encoder is frozen) ────────────
    print(f"\n{'='*60}")
    print("  ENCODER FEATURES (should be identical for both models)")
    print(f"{'='*60}")
    all_f24_means = np.array([v["f24_mean"] for v in results_random.values()])
    bg_mask  = np.array([v["is_background"] for v in results_random.values()])
    tis_mask = ~bg_mask

    if bg_mask.any() and tis_mask.any():
        print(f"  f24 mean — background: {all_f24_means[bg_mask].mean():.4f}")
        print(f"  f24 mean — tissue:     {all_f24_means[tis_mask].mean():.4f}")
        print(f"  f24 mean — difference: {all_f24_means[tis_mask].mean() - all_f24_means[bg_mask].mean():.4f}")
        print(f"\n  → If difference is large: encoder distinguishes background vs tissue ✅")
        print(f"  → If difference is small: encoder cannot distinguish them ❌")


if __name__ == "__main__":
    main()
