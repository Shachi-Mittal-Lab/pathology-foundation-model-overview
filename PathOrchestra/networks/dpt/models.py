import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.PathOrchestra import PathOrchestraEncoder

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_scratch
    # forward_vit
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class PathOrchestraDPT(nn.Module):
    def __init__(self, hf_token, num_classes=3, features=256, freeze_encoder=True):
        super().__init__()
        
        # encoder
        self.encoder = PathOrchestraEncoder(hf_token, freeze_encoder=freeze_encoder)
        
        # create reassemble layers
        self.scratch = _make_scratch([1024, 1024, 1024, 1024], features) # projects 1024 -> 256 channels (features) for each skip

        # fusion projection
        self.fusion_proj = nn.Conv2d(
            features * 4,
            features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        
        # fusion blocks for progressive upsampling
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn=True) # 14x14 -> 28x28
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn=True) # 28x28 -> 56x56
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn=True) # 56x56 -> 112x112
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn=True) # 112x112 -> 224x224
        
        # output segmentation head
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(features, num_classes, kernel_size=1), # project to n classes
            # Interpolate(scale_factor=2, mode="bilinear", align_corners=True), # no need for extra upsampling
        )
    
    def forward(self, x):
        vit_skips = self.encoder(x)             # extract the ViT feature maps
                                                # vit_skips = [f6, f12, f18, f24]
                                                # Each: (B, 1024, 14, 14)
        
        # reassemble from 1024 -> 256 channels
        layer_1_rn = self.scratch.layer1_rn(vit_skips[0])
        layer_2_rn = self.scratch.layer2_rn(vit_skips[1])
        layer_3_rn = self.scratch.layer3_rn(vit_skips[2])
        layer_4_rn = self.scratch.layer4_rn(vit_skips[3])

        # Concatenate all 4 features at same resolution
        x = torch.cat([layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn], dim=1)  # (B, 1024, 14, 14)
        
        # Project concatenated features back to single feature
        x = self.fusion_proj(x)  # (B, 256, 14, 14)

        # progressive upsampling (14→28→56→112→224)
        # no skip connections since all ViT skips are at same resolution
        path_4 = self.scratch.refinenet4(x)    # (B, 256, 28, 28)
        path_3 = self.scratch.refinenet3(path_4)  # (B, 256, 56, 56)
        path_2 = self.scratch.refinenet2(path_3)  # (B, 256, 112, 112)
        path_1 = self.scratch.refinenet1(path_2)  # (B, 256, 224, 224)
        
        out = self.scratch.output_conv(path_1)
        return out
    

# class DPT(BaseModel):
#     def __init__(
#         self,
#         head,
#         features=256,
#         backbone="vitb_rn50_384",
#         readout="project",
#         channels_last=False,
#         use_bn=False,
#         enable_attention_hooks=False,
#     ):

#         super(DPT, self).__init__()

#         self.channels_last = channels_last

#         hooks = {
#             "vitb_rn50_384": [0, 1, 8, 11],
#             "vitb16_384": [2, 5, 8, 11],
#             "vitl16_384": [5, 11, 17, 23],
#         }

#         # # Instantiate backbone and reassemble blocks
#         # self.pretrained, self.scratch = _make_encoder(
#         #     backbone,
#         #     features,
#         #     False,  # Set to true of you want to train from scratch, uses ImageNet weights
#         #     groups=1,
#         #     expand=False,
#         #     exportable=False,
#         #     hooks=hooks[backbone],
#         #     use_readout=readout,
#         #     enable_attention_hooks=enable_attention_hooks,
#         # )

#         self.encoder = PathOrchestraEncoder(hf_token, freeze_encoder)
#         self.scratch = _make_scratch([1024, 1024, 1024, 1024], features)  # All 4 are 1024-dim

#         self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
#         self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
#         self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
#         self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

#         self.scratch.output_conv = head

#     def forward(self, x):
#         if self.channels_last == True:
#             x.contiguous(memory_format=torch.channels_last)

#         layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

#         layer_1_rn = self.scratch.layer1_rn(layer_1)
#         layer_2_rn = self.scratch.layer2_rn(layer_2)
#         layer_3_rn = self.scratch.layer3_rn(layer_3)
#         layer_4_rn = self.scratch.layer4_rn(layer_4)

#         path_4 = self.scratch.refinenet4(layer_4_rn)
#         path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
#         path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
#         path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

#         out = self.scratch.output_conv(path_1)

#         return out


# class DPTDepthModel(DPT):
#     def __init__(
#         self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
#     ):
#         features = kwargs["features"] if "features" in kwargs else 256

#         self.scale = scale
#         self.shift = shift
#         self.invert = invert

#         head = nn.Sequential(
#             nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
#             Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
#             nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(True) if non_negative else nn.Identity(),
#             nn.Identity(),
#         )

#         super().__init__(head, **kwargs)

#         if path is not None:
#             self.load(path)

#     def forward(self, x):
#         inv_depth = super().forward(x).squeeze(dim=1)

#         if self.invert:
#             depth = self.scale * inv_depth + self.shift
#             depth[depth < 1e-8] = 1e-8
#             depth = 1.0 / depth
#             return depth
#         else:
#             return inv_depth


# class DPTSegmentationModel(DPT):
#     def __init__(self, num_classes, path=None, **kwargs):

#         features = kwargs["features"] if "features" in kwargs else 256

#         kwargs["use_bn"] = True

#         head = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(True),
#             nn.Dropout(0.1, False),
#             nn.Conv2d(features, num_classes, kernel_size=1),
#             Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
#         )

#         super().__init__(head, **kwargs)

#         self.auxlayer = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(True),
#             nn.Dropout(0.1, False),
#             nn.Conv2d(features, num_classes, kernel_size=1),
#         )

#         if path is not None:
#             self.load(path)