import timm
import torch
import torch.nn as nn
from huggingface_hub import login


class PathOrchestraEncoder(nn.Module):
    """
    Wraps PathOrchestra ViT-L/16 and exposes intermediate patch token grids
    for use in dense prediction tasks.

    PathOrchestra has 24 transformer blocks. We tap layers at indices
    [6, 12, 18, 24] (1-indexed) to get 4 semantic feature maps.
    All ViT skips have dim=1024, spatially reshaped to (B, 1024, H, W)
    where H, W are inferred dynamically from token count.

    ViT skips (all same spatial resolution for standard 224×224 input):
        f6:  (B, 1024, 14, 14)  — early semantic
        f12: (B, 1024, 14, 14)  — mid semantic
        f18: (B, 1024, 14, 14)  — deep semantic
        f24: (B, 1024, 14, 14)  — final features
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

        self.embed_dim = 1024  # ViT-L hidden dim

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) — typically (B, 3, 224, 224)

        Returns:
            vit_skips: [f6, f12, f18, f24], each (B, 1024, h, w)
                       where h = w = H // patch_size (14 for 224×224)
        """
        B = x.shape[0]

        # --- ViT forward (semantic skips) ---
        t = self.vit.patch_embed(x)     # (B, N, 1024)
        t = self.vit._pos_embed(t)      # (B, N+1, 1024) — CLS + positional embed
        t = self.vit.patch_drop(t)      # patch dropout (no-op at eval)
        t = self.vit.norm_pre(t)        # pre-transformer LayerNorm

        vit_skips = []
        for i, block in enumerate(self.vit.blocks):
            t = block(t)
            if (i + 1) in self.SKIP_LAYERS:
                patch_tokens = t[:, 1:, :]                          # drop CLS: (B, N, 1024)
                h = w = int(patch_tokens.shape[1] ** 0.5)           # infer grid size dynamically
                spatial = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
                vit_skips.append(spatial)                            # (B, 1024, h, w)

        return vit_skips