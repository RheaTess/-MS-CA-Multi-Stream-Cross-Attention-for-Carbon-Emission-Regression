"""The full MS-CA network (single-task regression).

    Stream A (Environment)   : [B, 3, 16, 16]  -- NO2, SO2, CO
    Stream B (Socio-Infra)   : [B, 4, 16, 16]  -- Nightlight, Urban Fraction,
                                                  Power Plant, Fossil Capacity

Population is excluded from Stream B, hence ``in_channels_b=4`` (was 5).
"""

from __future__ import annotations

from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn

from msca.models.encoders import BaseEncoder, ViTPatchEncoder
from msca.models.fusion import CrossAttentionFusion
from msca.models.heads import RegressionHead

__all__ = ["MSCANet"]


class MSCANet(nn.Module):
    """Multi-Stream Cross-Attention network for single-task regression.

    Composition (everything injected, nothing hard-wired):
        encoder_a : BaseEncoder for Stream A (Environment, 3ch)
        encoder_b : BaseEncoder for Stream B (Socio-Infra, 4ch)
        fusion    : CrossAttentionFusion
        reg_head  : RegressionHead (single scalar output)

    To swap the backbone, change ``encoder_a_cls`` / ``encoder_b_cls`` only.
    """

    def __init__(
        self,
        img_size: Sequence[int] = (16, 16),
        patch_size: int = 4,
        embed_dim: int = 128,
        encoder_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        in_channels_a: int = 3,
        in_channels_b: int = 4,
        encoder_a_cls: Type[BaseEncoder] = ViTPatchEncoder,
        encoder_b_cls: Type[BaseEncoder] = ViTPatchEncoder,
        **encoder_kwargs,
    ) -> None:
        """
        Args:
            img_size: (H, W) of the raw input crop. Accepts a list so it can be
                round-tripped through YAML/JSON config without conversion.
            patch_size: side length of each square patch (4 -> L = 16).
            embed_dim, encoder_depth, dropout: shared hyperparameters forwarded
                to both stream encoders.
            num_heads: attention heads, used by BOTH the ViT self-attention AND
                the cross-attention fusion layer.
            in_channels_a: Stream A channels (NO2, SO2, CO) -> 3.
            in_channels_b: Stream B channels (Nightlight, Urban Fraction, Power
                Plant, Fossil Capacity) -> 4. Population excluded.
            encoder_a_cls / encoder_b_cls: ``BaseEncoder`` subclasses -- the
                single point of control for backbone swaps.
            **encoder_kwargs: extra backbone-specific hyperparameters forwarded
                verbatim to both encoder constructors.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size: Tuple[int, int] = tuple(img_size)  # type: ignore[assignment]

        common_encoder_kwargs = dict(
            img_size=self.img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            dropout=dropout,
            **encoder_kwargs,
        )

        # Independent encoders -- weights are NOT shared, because the two
        # modalities have very different statistics.
        self.encoder_a = encoder_a_cls(in_channels=in_channels_a, **common_encoder_kwargs)
        self.encoder_b = encoder_b_cls(in_channels=in_channels_b, **common_encoder_kwargs)

        self.fusion = CrossAttentionFusion(
            embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.reg_head = RegressionHead(embed_dim, dropout=dropout)

    def forward(
        self,
        stream_a: torch.Tensor,
        stream_b: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            stream_a: [B, 3, 16, 16] environment proxy.
            stream_b: [B, 4, 16, 16] socio-infrastructure proxy.
            return_attention: also return the raw XAI attention matrix.

        Returns:
            reg_out: [B, 1] continuous carbon-emission prediction.
            attn_weights: [B, num_heads, Lq, Lk] -- only if ``return_attention``.
        """
        tokens_a = self.encoder_a(stream_a)  # [B, La, D]
        tokens_b = self.encoder_b(stream_b)  # [B, Lb, D]

        fused_tokens, attn_weights = self.fusion(tokens_b, tokens_a)  # [B, Lb, D]

        # Global average pool over the sequence -> one vector per sample.
        pooled = fused_tokens.mean(dim=1)  # [B, D]
        reg_out = self.reg_head(pooled)    # [B, 1]

        if return_attention:
            return reg_out, attn_weights
        return reg_out
