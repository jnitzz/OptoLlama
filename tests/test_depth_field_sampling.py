import unittest
from unittest import mock

import torch

from optollama.model.depth_field_diffusion import DepthFieldDiffusion, DepthFieldModelConfig


class DepthFieldSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DepthFieldDiffusion(
            DepthFieldModelConfig(
                spectrum_shape=(3, 8),
                num_materials=4,
                depth_bins=4,
                d_model=8,
                n_blocks=1,
                timesteps=4,
                kernel_size=3,
            )
        )

    def test_nonfinite_logits_are_repaired_before_multinomial(self) -> None:
        logits = torch.tensor(
            [
                [
                    [float("nan"), 1.0, -1.0, 0.0],
                    [float("inf"), 0.0, float("-inf"), 1.0],
                    [float("nan"), float("nan"), float("nan"), float("nan")],
                    [1.0, 2.0, 3.0, 4.0],
                ]
            ]
        )

        sampled, confidence, repaired = self.model._sample_logits(
            logits,
            temperature=1.0,
            top_k=0,
            deterministic=False,
            generator=torch.Generator().manual_seed(3),
        )

        self.assertEqual(tuple(sampled.shape), (1, 4))
        self.assertTrue(torch.all((sampled >= 0) & (sampled < 4)))
        self.assertTrue(torch.isfinite(confidence).all())
        self.assertTrue(torch.all((confidence >= 0.0) & (confidence <= 1.0)))
        self.assertEqual(int(repaired), 3)

    def test_sampling_survives_and_reports_nonfinite_logits(self) -> None:
        spectra = torch.zeros((1, 3, 8))

        def nonfinite_forward(*_args, **_kwargs):
            return torch.full((1, 4, 4), float("nan"))

        with mock.patch.object(self.model, "forward", side_effect=nonfinite_forward):
            with self.assertWarnsRegex(RuntimeWarning, "repaired 8 depth positions"):
                sampled = self.model.sample(
                    spectra,
                    steps=2,
                    temperature=1.0,
                    deterministic=False,
                    remask_strategy="random",
                    generator=torch.Generator().manual_seed(7),
                )

        self.assertEqual(tuple(sampled.shape), (1, 4))
        self.assertTrue(torch.all((sampled >= 0) & (sampled < 4)))
        self.assertEqual(self.model._last_sampling_nonfinite_positions, 8)
        self.assertEqual(self.model._last_sampling_first_nonfinite_step, 0)


if __name__ == "__main__":
    unittest.main()
