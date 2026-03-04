### AI GENERATED

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# Path to your converted 3-class model
model_path = "C:/Users/Waluigi/Desktop/github_repos/TransUNet/data/project_TransUNet/model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149_3class.pth"

# Load the checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

print("=" * 50)
print("MODEL INSPECTION")
print("=" * 50)

# 1. Check the keys in the state dict
print(f"\n1. Total number of layers: {len(checkpoint.keys())}")
print("\nFirst 10 layer names:")
for i, key in enumerate(list(checkpoint.keys())[:10]):
    print(f"   {i+1}. {key}")

# 2. Check the segmentation head specifically
print("\n" + "=" * 50)
print("2. SEGMENTATION HEAD INSPECTION")
print("=" * 50)

# Find all segmentation head layers
seg_head_layers = [k for k in checkpoint.keys() if 'segmentation_head' in k]
print(f"\nSegmentation head layers: {seg_head_layers}")

# Check the weights of the final conv layer
if 'segmentation_head.0.weight' in checkpoint:
    weight_shape = checkpoint['segmentation_head.0.weight'].shape
    print(f"\nFinal conv layer weight shape: {weight_shape}")
    print(f"   → This means: {weight_shape[0]} output classes, {weight_shape[1]} input channels")
    
    # Check if it's 3 classes
    if weight_shape[0] == 3:
        print("   ✅ Correct! Model has 3 output classes")
    else:
        print(f"   ❌ Wrong! Model has {weight_shape[0]} classes, expected 3")

if 'segmentation_head.0.bias' in checkpoint:
    bias_shape = checkpoint['segmentation_head.0.bias'].shape
    print(f"\nFinal conv layer bias shape: {bias_shape}")
    if bias_shape[0] == 3:
        print("   ✅ Correct! Bias has 3 elements")

# 3. Compare with original 9-class model
print("\n" + "=" * 50)
print("3. COMPARISON WITH ORIGINAL 9-CLASS MODEL")
print("=" * 50)

original_model_path = "C:/Users/Waluigi/Desktop/github_repos/TransUNet/data/project_TransUNet/model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth"
original_checkpoint = torch.load(original_model_path, map_location='cpu')

# Check original segmentation head
orig_weight_shape = original_checkpoint['segmentation_head.0.weight'].shape
print(f"\nOriginal 9-class model final layer shape: {orig_weight_shape}")
print(f"Converted 3-class model final layer shape: {weight_shape}")

# 4. Verify that other layers have the same weights
print("\n" + "=" * 50)
print("4. VERIFYING NON-HEAD LAYERS")
print("=" * 50)

# Check a few key layers to ensure they were transferred correctly
layers_to_check = [
    'transformer.embeddings.patch_embeddings.weight',
    'transformer.encoder.layer.0.attn.query.weight',
    'decoder.blocks.0.conv1.0.weight',
]

all_match = True
for layer in layers_to_check:
    if layer in checkpoint and layer in original_checkpoint:
        original_weight = original_checkpoint[layer]
        converted_weight = checkpoint[layer]
        if torch.allclose(original_weight, converted_weight):
            print(f"   ✅ {layer}: weights match")
        else:
            print(f"   ❌ {layer}: weights DON'T match")
            all_match = False
    else:
        print(f"   ⚠️ {layer}: not found in one of the models")

if all_match:
    print("\n   ✅ All checked non-head layers match the original model!")

# 5. Check the first few values of the new segmentation head
print("\n" + "=" * 50)
print("5. NEW SEGMENTATION HEAD VALUES")
print("=" * 50)

# Get the new head weights
new_head_weight = checkpoint['segmentation_head.0.weight']
new_head_bias = checkpoint['segmentation_head.0.bias']

print(f"\nFirst 5x5 corner of class 0 weights:")
print(new_head_weight[0, :5, 0, 0])  # First class, first 5 channels

print(f"\nFirst 5 bias values: {new_head_bias[:5]}")

# Compare with original head's first 3 classes
orig_head_weight = original_checkpoint['segmentation_head.0.weight']
orig_head_bias = original_checkpoint['segmentation_head.0.bias']

print(f"\nOriginal head's class 0 weights (first 5 values): {orig_head_weight[0, :5, 0, 0]}")
print(f"Converted head's class 0 weights (first 5 values): {new_head_weight[0, :5, 0, 0]}")

if torch.allclose(orig_head_weight[:3], new_head_weight):
    print("\n   ✅ New head's weights match first 3 classes of original head!")
else:
    print("\n   ⚠️ New head's weights are different from original (may be intentional if you reinitialized)")

print("\n" + "=" * 50)
print("INSPECTION COMPLETE")
print("=" * 50)