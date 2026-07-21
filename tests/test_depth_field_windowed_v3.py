import unittest
from dataclasses import replace

import torch

from optollama.model.depth_field_diffusion import (
    DepthFieldModelConfig,
    DepthFieldWindowedOptoLlamaDiffusion,
    DepthFieldWindowedOptoLlamaV2Diffusion,
    DepthFieldWindowedOptoLlamaV3Diffusion,
    build_depth_field_model,
)


class DepthFieldWindowedV3Tests(unittest.TestCase):
    """Verify the role-separated V3 conditioning path."""

    def setUp(self) -> None:
        """Create a small windowed model configuration for CPU tests."""
        self.config = DepthFieldModelConfig(
            spectrum_shape=(3, 16),
            num_materials=5,
            depth_bins=6,
            model_type="windowed_depth_v3",
            d_model=8,
            n_blocks=1,
            n_heads=2,
            timesteps=10,
            dropout=0.0,
            spectrum_patch_size=4,
            spectrum_patch_stride=2,
            spectrum_encoder_blocks=1,
            spectrum_encoder_heads=2,
        )

    def test_v3_appends_global_spectrum_token_and_keeps_time_condition_separate(self) -> None:
        """Route the pooled spectrum through cross-attention, not AdaLN."""
        model = DepthFieldWindowedOptoLlamaV3Diffusion(self.config).eval()
        spectra = torch.randn(2, 3, 16)
        timesteps = torch.tensor([1, 7])

        with torch.no_grad():
            window_tokens = DepthFieldWindowedOptoLlamaDiffusion._spectrum_tokens(model, spectra)
            spectrum_tokens = model._spectrum_tokens(spectra)
            time_token = model.time_embedding(timesteps)
            condition = model._block_condition(spectrum_tokens, time_token)

        self.assertEqual(spectrum_tokens.shape[1], window_tokens.shape[1] + 1)
        torch.testing.assert_close(spectrum_tokens[:, :-1], window_tokens)
        torch.testing.assert_close(
            spectrum_tokens[:, -1],
            model.global_spectrum_condition(window_tokens.mean(dim=1)),
        )
        torch.testing.assert_close(condition, time_token.squeeze(1))

    def test_v3_forward_is_finite_and_uses_expected_shape(self) -> None:
        """Build V3 through the public factory and run a finite forward pass."""
        model = build_depth_field_model(self.config).eval()
        self.assertIsInstance(model, DepthFieldWindowedOptoLlamaV3Diffusion)

        with torch.no_grad():
            logits = model(
                torch.randn(2, 3, 16),
                torch.randint(0, self.config.num_materials + 1, (2, self.config.depth_bins)),
                torch.tensor([0, self.config.timesteps - 1]),
            )

        self.assertEqual(tuple(logits.shape), (2, self.config.depth_bins, self.config.num_materials))
        self.assertTrue(torch.isfinite(logits).all())

    def test_v2_and_v3_state_dict_shapes_are_compatible(self) -> None:
        """Keep V2 and V3 parameter names and shapes compatible."""
        v2 = DepthFieldWindowedOptoLlamaV2Diffusion(
            replace(self.config, model_type="optollama_depth_windowed_v2")
        )
        v3 = DepthFieldWindowedOptoLlamaV3Diffusion(self.config)
        v2_state = v2.state_dict()
        v3_state = v3.state_dict()

        self.assertEqual(v2_state.keys(), v3_state.keys())
        self.assertEqual(
            {name: tuple(value.shape) for name, value in v2_state.items()},
            {name: tuple(value.shape) for name, value in v3_state.items()},
        )
        v3.load_state_dict(v2_state, strict=True)


if __name__ == "__main__":
    unittest.main()
