import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double conv block: the canonical UNet pattern.
    First conv absorbs concatenated channels, second refines at the new width.
    Replaces the single-conv ConvBnRelu from the original decoder.py.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Upsample 2x via ConvTranspose2d, concatenate skip connection,
    then refine with a double ConvBlock.

    F.interpolate is used as a safety resize on the skip before concat —
    guards against spatial misalignment at non-standard input sizes
    (important for daisy block inference later).
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    4-stage decoder: 14 → 28 → 56 → 112 → 224

    Skips arrive as raw 1024-dim features from PathOrchestra — no projection.
    Preserving full 1024-dim skips retains more pretrained feature richness
    compared to projecting them down first.

    Channel schedule per stage:
        Bottleneck : (B, 1024, 14,  14)
        dec4 out   : (B,  512, 28,  28)   skip from layer 18: (B, 1024, 14, 14)
        dec3 out   : (B,  256, 56,  56)   skip from layer 12: (B, 1024, 28, 28)
        dec2 out   : (B,  128, 112, 112)  skip from layer  6: (B, 1024, 56, 56)
        dec1 out   : (B,   64, 224, 224)  no skip
        head out   : (B, num_classes, 224, 224)

    num_classes: 3 for (stroma, epithelium, other); change to 5 when ready.
    """

    def __init__(self, encoder_dim: int = 1024, num_classes: int = 3):
        super().__init__()

        # Bottleneck: initial projection of the deepest skip (f24)
        self.bottleneck = ConvBlock(encoder_dim, 512)

        # in_ch=512, skip_ch=1024 (raw), out_ch=512 → concat = 256 + 1024 = 1280 in ConvBlock
        self.dec4 = DecoderBlock(in_ch=512,  skip_ch=1024, out_ch=512)
        self.dec3 = DecoderBlock(in_ch=512,  skip_ch=1024, out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256,  skip_ch=1024, out_ch=128)

        # Final upsample 112→224, no skip connection at this stage
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ConvBlock(64, 64),
        )

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, skips: list):
        """
        Args:
            skips: [f6, f12, f18, f24], each (B, 1024, 14, 14)
        Returns:
            logits: (B, num_classes, 224, 224)
        """
        f6, f12, f18, f24 = skips

        x = self.bottleneck(f24)        # (B, 512,  14,  14)
        x = self.dec4(x, skip=f18)      # (B, 512,  28,  28)
        x = self.dec3(x, skip=f12)      # (B, 256,  56,  56)
        x = self.dec2(x, skip=f6)       # (B, 128, 112, 112)
        x = self.dec1(x)                # (B,  64, 224, 224)

        return self.head(x)             # (B, num_classes, 224, 224)