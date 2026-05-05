import torch.nn as nn
from .encoder import PathOrchestraEncoder
from .decoder import UNetDecoder


class DCISmodel(nn.Module):
    def __init__(self, hf_token: str, num_classes: int = 3, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = PathOrchestraEncoder(hf_token, freeze_encoder)
        self.decoder = UNetDecoder(encoder_dim=1024, num_classes=num_classes)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)