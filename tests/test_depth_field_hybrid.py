import unittest
from dataclasses import replace

import torch

from optollama.model.depth_field_diffusion import (
    DepthFieldHybridDiffusion,
    DepthFieldModelConfig,
    DepthFieldWindowedOptoLlamaV3Diffusion,
    DepthwiseSeparableConv1d,
    build_depth_field_model,
)


class DepthFieldHybridTests(unittest.TestCase):
    """Verify interleaved transformer and depth-convolution behavior."""

    def setUp(self) -> None:
        """Create a compact hybrid configuration for CPU tests."""
        self.config = DepthFieldModelConfig(
            spectrum_shape=(3, 16),
            num_materials=5,
            depth_bins=12,
            model_type="hybrid",
            d_model=8,
            n_blocks=4,
            kernel_size=7,
            n_heads=2,
            timesteps=10,
            dropout=0.0,
            conv_type="separable",
            hybrid_dilations=(1, 2, 4, 8),
            hybrid_residual_init=1.0e-3,
            spectrum_patch_size=4,
            spectrum_patch_stride=2,
            spectrum_encoder_blocks=1,
            spectrum_encoder_heads=2,
        )

    def test_factory_builds_interleaved_separable_blocks(self) -> None:
        """Build the requested dilation sequence with gated separable blocks."""
        model = build_depth_field_model(self.config)

        self.assertIsInstance(model, DepthFieldHybridDiffusion)
        self.assertEqual(model.config.model_type, "optollama_depth_hybrid")
        self.assertEqual(tuple(block.dilation for block in model.conv_blocks), (1, 2, 4, 8))
        self.assertTrue(all(isinstance(block.conv1, DepthwiseSeparableConv1d) for block in model.conv_blocks))
        self.assertTrue(
            all(abs(float(block.residual_scale.detach()) - 1.0e-3) < 1.0e-8 for block in model.conv_blocks)
        )
        self.assertEqual(model.convolution_receptive_field_bins, 181)

    def test_forward_is_finite_and_uses_expected_shape(self) -> None:
        """Produce finite material logits for every depth position."""
        model = build_depth_field_model(self.config).eval()
        with torch.no_grad():
            logits = model(
                torch.randn(2, 3, 16),
                torch.randint(0, self.config.num_materials + 1, (2, self.config.depth_bins)),
                torch.tensor([0, self.config.timesteps - 1]),
            )

        self.assertEqual(tuple(logits.shape), (2, self.config.depth_bins, self.config.num_materials))
        self.assertTrue(torch.isfinite(logits).all())

    def test_backward_reaches_transformer_convolution_and_residual_gate(self) -> None:
        """Backpropagate through both paths despite the small initial convolution gate."""
        model = build_depth_field_model(self.config)
        logits = model(
            torch.randn(2, 3, 16),
            torch.randint(0, self.config.num_materials + 1, (2, self.config.depth_bins)),
            torch.tensor([1, 6]),
        )
        logits.square().mean().backward()

        gradients = (
            model.blocks[0].self_attn.in_proj_weight.grad,
            model.conv_blocks[0].conv1.pointwise.weight.grad,
            model.conv_blocks[0].residual_scale.grad,
        )
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None))
        self.assertTrue(all(torch.count_nonzero(gradient) > 0 for gradient in gradients if gradient is not None))

    def test_zero_conv_scales_recover_v3_forward(self) -> None:
        """Make the hybrid exactly match V3 when convolution residuals are disabled."""
        v3_config = replace(self.config, model_type="optollama_depth_windowed_v3")
        v3 = DepthFieldWindowedOptoLlamaV3Diffusion(v3_config).eval()
        hybrid = DepthFieldHybridDiffusion(self.config).eval()
        incompatible = hybrid.load_state_dict(v3.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        with torch.no_grad():
            for block in hybrid.conv_blocks:
                block.residual_scale.zero_()
            spectra = torch.randn(2, 3, 16)
            fields = torch.randint(0, self.config.num_materials + 1, (2, self.config.depth_bins))
            timesteps = torch.tensor([2, 8])
            expected = v3(spectra, fields, timesteps)
            actual = hybrid(spectra, fields, timesteps)

        torch.testing.assert_close(actual, expected)

    def test_config_round_trip_preserves_hybrid_settings(self) -> None:
        """Retain architecture settings in checkpoint model metadata."""
        restored = DepthFieldModelConfig.from_dict(self.config.to_dict())
        self.assertEqual(restored, self.config)

    def test_hybrid_requires_one_dilation_per_transformer_block(self) -> None:
        """Reject ambiguous transformer-to-convolution stage mappings."""
        with self.assertRaisesRegex(ValueError, "one dilation per transformer block"):
            replace(self.config, hybrid_dilations=(1, 2))


if __name__ == "__main__":
    unittest.main()
