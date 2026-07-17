import unittest
from unittest import mock

import torch

from optollama.model.depth_field_diffusion import (
    DepthFieldCorruptionConfig,
    DepthFieldDiffusion,
    DepthFieldModelConfig,
    depth_field_corruption_mask,
)


class DepthFieldCorruptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hybrid = DepthFieldCorruptionConfig(
            mode="hybrid",
            iid_fraction=0.2,
            span_fraction=0.6,
            layer_fraction=0.2,
            span_min_bins=4,
            span_max_bins=8,
            span_scale_with_noise=True,
        )

    def test_uppercase_config_and_budget(self) -> None:
        policy = DepthFieldCorruptionConfig.from_dict(
            {
                "MODE": "hybrid",
                "IID_FRACTION": 0.2,
                "SPAN_FRACTION": 0.6,
                "LAYER_FRACTION": 0.2,
                "SPAN_MIN_BINS": 4,
                "SPAN_MAX_BINS": 8,
                "SPAN_SCALE_WITH_NOISE": True,
            }
        )
        fields = torch.arange(40).repeat(2, 1)
        mask = depth_field_corruption_mask(
            fields,
            torch.tensor([0.25, 0.50]),
            config=policy,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(mask.sum(dim=1).tolist(), [10, 20])

    def test_hybrid_mask_is_reproducible(self) -> None:
        fields = torch.arange(40).repeat(2, 1)
        first = depth_field_corruption_mask(
            fields,
            torch.tensor([0.4, 0.4]),
            config=self.hybrid,
            generator=torch.Generator().manual_seed(23),
        )
        second = depth_field_corruption_mask(
            fields,
            torch.tensor([0.4, 0.4]),
            config=self.hybrid,
            generator=torch.Generator().manual_seed(23),
        )
        self.assertTrue(torch.equal(first, second))

    def test_layer_only_policy_selects_a_complete_run(self) -> None:
        fields = torch.tensor([[0] * 4 + [1] * 4 + [2] * 4 + [3] * 4])
        policy = DepthFieldCorruptionConfig(
            mode="hybrid",
            iid_fraction=0.0,
            span_fraction=0.0,
            layer_fraction=1.0,
            span_min_bins=2,
            span_max_bins=4,
        )
        mask = depth_field_corruption_mask(
            fields,
            torch.tensor([0.25]),
            config=policy,
            generator=torch.Generator().manual_seed(7),
        )[0]
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        self.assertEqual(selected.numel(), 4)
        self.assertEqual(int(selected[-1] - selected[0] + 1), 4)
        self.assertEqual(torch.unique(fields[0, selected]).numel(), 1)

    def test_span_policy_creates_contiguous_missing_regions(self) -> None:
        fields = torch.zeros((1, 40), dtype=torch.long)
        policy = DepthFieldCorruptionConfig(
            mode="hybrid",
            iid_fraction=0.0,
            span_fraction=1.0,
            layer_fraction=0.0,
            span_min_bins=5,
            span_max_bins=5,
            span_scale_with_noise=False,
        )
        mask = depth_field_corruption_mask(
            fields,
            torch.tensor([0.25]),
            config=policy,
            generator=torch.Generator().manual_seed(3),
        )[0]
        padded = torch.cat([torch.tensor([False]), mask, torch.tensor([False])])
        changes = torch.nonzero(padded[1:] != padded[:-1], as_tuple=False).flatten()
        run_lengths = changes[1::2] - changes[::2]
        self.assertEqual(int(mask.sum()), 10)
        self.assertGreaterEqual(int(run_lengths.max()), 5)

    def test_training_and_random_sampling_use_hybrid_selector(self) -> None:
        model = DepthFieldDiffusion(
            DepthFieldModelConfig(
                spectrum_shape=(3, 8),
                num_materials=4,
                depth_bins=40,
                d_model=8,
                n_blocks=1,
                timesteps=10,
                kernel_size=3,
            )
        )
        clean = torch.arange(40).remainder(4).view(1, -1)
        noised, corrupted = model.corrupt(
            clean,
            torch.tensor([4]),
            random_replace_prob=0.0,
            corruption_config=self.hybrid,
            generator=torch.Generator().manual_seed(5),
        )
        self.assertEqual(int(corrupted.sum()), 10)
        self.assertTrue(torch.all(noised[corrupted] == model.mask_id))

        spectra = torch.zeros((1, 3, 8))
        target = "optollama.model.depth_field_diffusion.depth_field_corruption_mask"
        with mock.patch(target, wraps=depth_field_corruption_mask) as selector:
            sampled = model.sample(
                spectra,
                steps=3,
                deterministic=True,
                remask_strategy="random",
                corruption_config=self.hybrid,
            )
        self.assertEqual(tuple(sampled.shape), (1, 40))
        self.assertGreaterEqual(selector.call_count, 1)


if __name__ == "__main__":
    unittest.main()
