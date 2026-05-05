import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login

login(token="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")

encoder = timm.create_model(
    "hf-hub:AI4Pathology/PathOrchestra",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True,
)
encoder.eval()


def extract_features(model, x, layers=[6, 12, 18, 24]):
    """Extract intermediate ViT block outputs for multi-scale UNet skip connections."""
    features = []
    B, C, H, W = x.shape
    
    # Patchify + embed
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    
    for i, block in enumerate(model.blocks):
        x = block(x)
        if (i + 1) in layers: # if this is block 6, 12, 18, or 24...
            # Remove CLS token, reshape to 2D spatial grid for decoder
            tokens = x[:, 1:, :]  # [B, 196, 1024]
            h = w = int(tokens.shape[1] ** 0.5)  # 14x14
            spatial = tokens.permute(0, 2, 1).reshape(B, -1, h, w)
            features.append(spatial)
    
    return features  # list of [B, 1024, 14, 14]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_dim=1024, num_classes=8):
        super().__init__()
        # Each stage upsamples 2x and reduces channels
        self.up4 = nn.ConvTranspose2d(encoder_dim, 512, 2, stride=2)   # 14→28
        self.conv4 = ConvBlock(512 + 1024, 512)  # +skip from layer 18
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)            # 28→56
        self.conv3 = ConvBlock(256 + 1024, 256)  # +skip from layer 12
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)            # 56→112
        self.conv2 = ConvBlock(128 + 1024, 128)  # +skip from layer 6
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)             # 112→224
        self.conv1 = ConvBlock(64, 64)
        
        self.head = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, features):
        # features = [f6, f12, f18, f24] each [B, 1024, 14, 14]
        f6, f12, f18, f24 = features
        
        x = self.up4(f24)                          # [B, 512, 28, 28]
        x = self.conv4(torch.cat([x, F.interpolate(f18, x.shape[2:])], dim=1))
        
        x = self.up3(x)                            # [B, 256, 56, 56]
        x = self.conv3(torch.cat([x, F.interpolate(f12, x.shape[2:])], dim=1))
        
        x = self.up2(x)                            # [B, 128, 112, 112]
        x = self.conv2(torch.cat([x, F.interpolate(f6, x.shape[2:])], dim=1))
        
        x = self.up1(x)                            # [B, 64, 224, 224]
        x = self.conv1(x)
        
        return self.head(x)                        # [B, num_classes, 224, 224]
    

class PathOrchestraUNet(nn.Module):
    def __init__(self, encoder, num_classes=8, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder(encoder_dim=1024, num_classes=num_classes)
        
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        features = extract_features(self.encoder, x, layers=[6, 12, 18, 24])
        return self.decoder(features) # returns segmentation map