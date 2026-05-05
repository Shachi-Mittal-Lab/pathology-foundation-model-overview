import timm
import torch
import torch.nn as nn
from huggingface_hub import login


class PathOrchestraEncoder(nn.Module):
    """
    Wraps PathOrchestra ViT-L/16 and exposes intermediate patch token grids
    for use as UNet skip connections.

    PathOrchestra has 24 transformer blocks. We tap layers at indices
    [6, 12, 18, 24] (1-indexed) to get 4 scales of features.
    All have dim=1024, spatially reshaped to (B, 1024, H, W)
    where H, W are inferred dynamically from token count.
    """

    SKIP_LAYERS = [6, 12, 18, 24]  # 1-indexed: quarter, half, three-quarter, final

    def __init__(self, hf_token: str, freeze_encoder: bool = True):
        super().__init__()
        login(token=hf_token)
        self.vit = timm.create_model(
            "hf-hub:AI4Pathology/PathOrchestra",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        if freeze_encoder:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.embed_dim = 1024   # ViT-L hidden dim

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) — typically (B, 3, 224, 224)
        Returns:
            skips: list of 4 tensors, each (B, 1024, h, w)
                   where h, w are inferred dynamically from token count
        """
        B = x.shape[0]

        # prepare tokens (patch + positional embeddings)
        x = self.vit.patch_embed(x)   # (B, N, 1024)
        x = self.vit._pos_embed(x)    # (B, N+1, 1024) — adds CLS token + positional embed
        x = self.vit.patch_drop(x)    # patch dropout (no-op at eval, but correct to include)
        x = self.vit.norm_pre(x)      # pre-transformer LayerNorm (identity in some configs)

        # feature extraction for skip connections
        skips = []
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if (i + 1) in self.SKIP_LAYERS:
                patch_tokens = x[:, 1:, :]                        # drop CLS: (B, N, 1024)
                h = w = int(patch_tokens.shape[1] ** 0.5)         # infer grid size dynamically
                spatial = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
                skips.append(spatial)                              # (B, 1024, h, w)

        return skips   