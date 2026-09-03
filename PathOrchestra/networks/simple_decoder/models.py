import torch
import torch.nn as nn
from ..backbones.PathOrchestra import PathOrchestraEncoder
from .blocks import SimpleDecoder


class PathOrchestraSimpleDecoder(nn.Module):
    """
    PathOrchestra ViT encoder + simple linear decoder.
    Quick, lightweight alternative to full DPT.
    
    Args:
        hf_token: HuggingFace token for PathOrchestra
        num_classes: Number of segmentation classes
        hidden_dim: Hidden dimension (default 256)
        freeze_encoder: Whether to freeze ViT weights
    """
    
    def __init__(self, hf_token, num_classes=5, hidden_dim=256, freeze_encoder=True):
        super().__init__()
        
        # Encoder
        self.encoder = PathOrchestraEncoder(hf_token, freeze_encoder=freeze_encoder)
        
        # Decoder
        self.decoder = SimpleDecoder(
            in_channels=1024,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) input image
        
        Returns:
            logits: (B, num_classes, 224, 224)
        """
        # Extract ViT features
        vit_skips = self.encoder(x)  # List of 4 tensors (B, 1024, 14, 14)
        
        # Use only the final (deepest) skip
        features = vit_skips[-1]  # (B, 1024, 14, 14)
        
        # Decode
        logits = self.decoder(features)  # (B, num_classes, 224, 224)
        
        return logits