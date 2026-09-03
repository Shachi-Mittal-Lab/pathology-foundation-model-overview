import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from funlib.persistence import open_ds
from networks.seg import DCISmodel

device = torch.device("cuda")
model = DCISmodel(hf_token="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", num_classes=4).eval().to(device)

inp_transforms = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

s2_array = open_ds(Path(r"E:\[PROJ]_DCIS\PathOrchestra\TEMP_test_inference\2015003_H&E.zarr") / "raw" / "s0")

# grab patches
bg_patch     = s2_array[0:224,       0:224,       :]  # background
tissue_patch = s2_array[27695:(27695+224), 42927:(42927+224), :]  # tissue

print(f"Background brightness: {bg_patch.mean():.1f}")
print(f"Tissue brightness:     {tissue_patch.mean():.1f}")

features = {}
for name, patch in [("background", bg_patch), ("tissue", tissue_patch)]:
    img = Image.fromarray(patch)
    x = inp_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        skips = model.encoder(x)
        f24 = skips[-1].flatten()  # flatten to 1D vector
    features[name] = f24

# cosine similarity between background and tissue features
sim = F.cosine_similarity(
    features["background"].unsqueeze(0),
    features["tissue"].unsqueeze(0)
).item()

print(f"\nCosine similarity (background vs tissue): {sim:.4f}")
print("  1.0 = identical, 0.0 = orthogonal, -1.0 = opposite")
print("  < 0.9 = meaningfully different ✅")
print("  > 0.95 = very similar ❌")