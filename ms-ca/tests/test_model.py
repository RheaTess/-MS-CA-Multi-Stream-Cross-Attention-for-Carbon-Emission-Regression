"""Architecture contract tests."""

from __future__ import annotations

import pytest
import torch

from msca.models import MSCANet, ViTPatchEncoder, reshape_attention_to_spatial_grid


def test_forward_returns_single_scalar_per_sample():
    model = MSCANet(embed_dim=32, encoder_depth=1, num_heads=4)
    out = model(torch.randn(5, 3, 16, 16), torch.randn(5, 4, 16, 16))
    # Pure regression: one tensor out, not a (reg, cls) tuple.
    assert isinstance(out, torch.Tensor)
    assert out.shape == (5, 1)


def test_channel_layout_is_three_and_four():
    model = MSCANet(embed_dim=32, encoder_depth=1)
    assert model.encoder_a.patch_embed.in_channels == 3
    assert model.encoder_b.patch_embed.in_channels == 4


def test_wrong_stream_b_channel_count_raises():
    model = MSCANet(embed_dim=32, encoder_depth=1)
    with pytest.raises(RuntimeError):
        # 5 channels = the old population-included layout.
        model(torch.randn(2, 3, 16, 16), torch.randn(2, 5, 16, 16))


def test_model_has_no_classification_head():
    model = MSCANet(embed_dim=32, encoder_depth=1)
    names = dict(model.named_modules())
    assert not any("cls" in name.lower() for name in names)
    assert hasattr(model, "reg_head")


def test_encoder_grid_matches_sequence_length():
    encoder = ViTPatchEncoder(in_channels=3, embed_dim=32, depth=1)
    tokens = encoder(torch.randn(2, 3, 16, 16))
    gh, gw = encoder.grid_size
    assert tokens.shape == (2, 16, 32)
    assert gh * gw == encoder.num_patches == tokens.shape[1]


def test_attention_reshapes_to_spatial_grid():
    model = MSCANet(embed_dim=32, encoder_depth=1, num_heads=4)
    _, attn = model(
        torch.randn(2, 3, 16, 16), torch.randn(2, 4, 16, 16), return_attention=True
    )
    assert attn.shape == (2, 4, 16, 16)  # [B, heads, Lq, Lk]

    spatial = reshape_attention_to_spatial_grid(
        attn, model.encoder_b.grid_size, model.encoder_a.grid_size
    )
    assert spatial.shape == (2, 4, 4, 4, 4)  # [B, gh_q, gw_q, gh_k, gw_k]


def test_reshape_rejects_mismatched_grid():
    attn = torch.rand(2, 4, 16, 16)
    with pytest.raises(ValueError, match="query_grid"):
        reshape_attention_to_spatial_grid(attn, (3, 3), (4, 4))


def test_model_config_roundtrips_through_a_plain_dict():
    # This is exactly what checkpoint save/load does.
    config = {
        "img_size": [16, 16],  # list, as it comes back from YAML/JSON
        "patch_size": 4,
        "embed_dim": 32,
        "encoder_depth": 1,
        "num_heads": 4,
        "dropout": 0.1,
        "in_channels_a": 3,
        "in_channels_b": 4,
    }
    model = MSCANet(**config)
    rebuilt = MSCANet(**config)
    rebuilt.load_state_dict(model.state_dict())
    assert rebuilt.img_size == (16, 16)
