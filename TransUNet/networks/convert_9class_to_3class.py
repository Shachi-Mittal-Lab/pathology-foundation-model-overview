import torch
import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def convert_checkpoint(args):
    # Load configuration for 9 classes first
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = 9
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), 
                                   int(args.img_size/args.vit_patches_size))
    
    # Create 9-class model and load pre-trained weights
    print(f"Loading 9-class model from {args.checkpoint_path}")
    model_9class = ViT_seg(config_vit, img_size=args.img_size, num_classes=9)
    
    # Load checkpoint (handle both DataParallel and non-DataParallel)
    state_dict = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Remove 'module.' prefix if it exists (from DataParallel training)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model_9class.load_state_dict(state_dict)
    
    # create model for 3 classes
    config_vit.n_classes = 3
    model_3class = ViT_seg(config_vit, img_size=args.img_size, num_classes=3)
    
    # Get state dicts
    model_3class_dict = model_3class.state_dict()
    
    # Copy all weights except the segmentation head
    pretrained_dict = {k: v for k, v in model_9class.state_dict().items() 
                      if k in model_3class_dict and 
                      not k.startswith('segmentation_head.')}
    
    # Update the 3-class model with pretrained weights
    model_3class_dict.update(pretrained_dict)
    model_3class.load_state_dict(model_3class_dict)
    
    # Optional: Initialize the new segmentation head with something smarter
    # You can copy weights from the first 3 classes of the old head
    old_head_weight = model_9class.state_dict()['segmentation_head.0.weight']
    old_head_bias = model_9class.state_dict()['segmentation_head.0.bias']
    
    # Copy first 3 classes' weights to new head
    with torch.no_grad():
        model_3class.segmentation_head[0].weight.copy_(old_head_weight[:3])
        model_3class.segmentation_head[0].bias.copy_(old_head_bias[:3])
    
    print(f"Weights transferred: {len(pretrained_dict)} layers")
    print(f"New segmentation head initialized with first 3 classes from old model")
    
    # Save the modified checkpoint
    output_path = args.output_path or args.checkpoint_path.replace('.pth', '_3class.pth')
    torch.save(model_3class.state_dict(), output_path)
    print(f"Saved 3-class checkpoint to {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to your trained 9-class checkpoint')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the modified checkpoint')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                       help='ViT model name (use same as training)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--n_skip', type=int, default=3,
                       help='Number of skip connections')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                       help='ViT patches size')
    
    args = parser.parse_args()
    
    # Use the same values you used for training
    # You can also hardcode them if you know them:
    # args.vit_name = 'R50-ViT-B_16'  # or whatever you used
    # args.img_size = 224
    # args.n_skip = 3
    # args.vit_patches_size = 16
    
    output_path = convert_checkpoint(args)
    print(f"\nNow you can use this checkpoint in your train.py:")
    print(f"1. In train.py, replace the pretrained loading with:")
    print(f'   checkpoint = torch.load("{output_path}")')
    print(f"   net.load_state_dict(checkpoint)")
    print(f"2. Set args.num_classes = 3 in train.py")