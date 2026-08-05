from __future__ import annotations

import argparse
import unittest

import torch

import optollama.model
from scripts.train_depth_field_diffusion import (
    resolve_init_checkpoint,
    spectral_aux_config,
    spectral_auxiliary_loss,
)


def small_surrogate() -> optollama.model.DepthFieldSpectrumSurrogate:
    config = optollama.model.DepthFieldSpectrumSurrogateConfig(
        num_materials=4,
        void_id=3,
        depth_bins=16,
        spectrum_width=9,
        dz_nm=5.0,
        d_model=16,
        conv_dilations=(1, 2),
        kernel_size=3,
        depth_pool=4,
        decoder_blocks=1,
        decoder_heads=4,
    )
    return optollama.model.DepthFieldSpectrumSurrogate(config)


class _NoiseCore:
    @staticmethod
    def noise_probability(timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / 10.0


class DepthFieldSpectralAuxTests(unittest.TestCase):
    def test_surrogate_compacts_void_gaps_and_conserves_rat(self) -> None:
        model = small_surrogate().eval()
        field_with_gaps = torch.tensor([[0, 3, 1, 1, 3, 2] + [3] * 10])
        compact_field = torch.tensor([[0, 1, 1, 2] + [3] * 12])

        with torch.no_grad():
            gap_spectrum = model(field_with_gaps)
            compact_spectrum = model(compact_field)

        torch.testing.assert_close(gap_spectrum, compact_spectrum)
        torch.testing.assert_close(gap_spectrum.sum(dim=1), torch.ones_like(gap_spectrum[:, 0]))

    def test_straight_through_surrogate_path_reaches_inverse_logits(self) -> None:
        model = small_surrogate().eval().requires_grad_(False)
        logits = torch.randn(2, 16, 4, requires_grad=True)
        probabilities = optollama.model.straight_through_material_probabilities(logits)

        model(probabilities)[:, 0].mean().backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_spectral_auxiliary_loss_selects_eligible_samples_and_ramps(self) -> None:
        surrogate = small_surrogate().eval().requires_grad_(False)
        logits = torch.randn(3, 16, 4, requires_grad=True)
        target = torch.rand(3, 3, 9)
        target = target / target.sum(dim=1, keepdim=True)
        output = {
            "logits": logits,
            "timesteps": torch.tensor([1, 8, 2]),
            "condition_dropped": torch.tensor([False, False, True]),
        }
        config = {
            "enabled": True,
            "every_n_steps": 1,
            "max_samples_per_rank": 8,
            "max_noise_probability": 0.5,
            "skip_dropped_conditions": True,
            "start_after_samples": 0,
            "weight_ramp_samples": 100,
            "weight": 0.2,
            "channels": (0, 2),
            "straight_through_temperature": 1.0,
            "derivative_weight": 0.25,
            "huber_delta": 0.02,
        }

        result = spectral_auxiliary_loss(
            output=output,
            target_spectra=target,
            core_model=_NoiseCore(),
            surrogate=surrogate,
            config=config,
            global_samples_seen=50,
            batch_index=0,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["samples"], 1)
        self.assertAlmostEqual(result["weight"], 0.1)
        result["loss"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_spectral_aux_config_is_backward_compatible_when_absent(self) -> None:
        args = argparse.Namespace(
            spectral_aux_enabled=None,
            spectral_aux_checkpoint=None,
            spectral_aux_weight=None,
            spectral_aux_every_n_steps=None,
            spectral_aux_max_samples_per_rank=None,
        )

        config = spectral_aux_config({"DEPTH_FIELD": {"TRAIN": {}}}, args)

        self.assertFalse(config["enabled"])
        self.assertIsNone(config["checkpoint"])

    def test_resolve_init_checkpoint_supports_new_run_initialization(self) -> None:
        args = argparse.Namespace(init_from=None)
        cfg = {"CHECKPOINT": {"INIT_FROM": "source.pt"}}

        path, source = resolve_init_checkpoint(cfg, args)

        self.assertEqual(str(path), "source.pt")
        self.assertEqual(source, "CHECKPOINT.INIT_FROM")


if __name__ == "__main__":
    unittest.main()
