import torch
import torch.nn as nn


class SimpleDecoder(nn.Module):
    """
    Simple linear decoder for ViT features.
    Minimal learnable parameters, good for quick testing.
    """
    def __init__(self, in_channels=1024, num_classes=5, hidden_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True)
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1024, 14, 14) ViT features
        
        Returns:
            logits: (B, num_classes, 224, 224)
        """
        x = self.proj(x)           # (B, 256, 14, 14)
        x = self.head(x)           # (B, num_classes, 14, 14)
        x = self.upsample(x)       # (B, num_classes, 224, 224)
        return x