import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from optollama.model.depth_field_diffusion import depth_field_boundary_mask, weighted_depth_field_loss
from optollama.model.optollama import AdaLayerNormGaussian
from scripts.train_depth_field_diffusion import (
    depth_field_training_loss,
    finite_tensor_stats,
    first_nonfinite_model_tensor,
    first_nonfinite_optimizer_tensor,
    normalize_amp_dtype,
    reduced_epoch_metrics,
    resolve_amp_dtype,
    run_epoch,
    synchronized_finite_flags,
)


class DepthFieldTrainingSafetyTests(unittest.TestCase):
    def test_boundary_mask_marks_transition_context(self) -> None:
        fields = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]])

        adjacent = depth_field_boundary_mask(fields, radius_bins=0)
        expanded = depth_field_boundary_mask(fields, radius_bins=1)

        self.assertEqual(adjacent.tolist(), [[False, True, True, False, True, True, False, False]])
        self.assertEqual(expanded.tolist(), [[True, True, True, True, True, True, True, False]])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            depth_field_boundary_mask(fields, radius_bins=-1)

    def test_training_loss_applies_boundary_weight_and_null_condition_dropout(self) -> None:
        class UniformModel(torch.nn.Module):
            timesteps = 1
            num_materials = 2

            def __init__(self) -> None:
                super().__init__()
                self.last_spectra = None

            def corrupt(self, clean_fields, _timesteps, **_kwargs):
                return clean_fields, torch.ones_like(clean_fields, dtype=torch.bool)

            def forward(self, spectra, clean_fields, _timesteps):
                self.last_spectra = spectra
                return torch.zeros((*clean_fields.shape, self.num_materials), device=clean_fields.device)

        spectra = torch.ones((1, 3, 4))
        fields = torch.tensor([[0, 0, 1, 1]])
        model = UniformModel()

        baseline = depth_field_training_loss(model, spectra, fields, void_id=-1)
        weighted = depth_field_training_loss(
            model,
            spectra,
            fields,
            void_id=-1,
            condition_dropout_prob=1.0,
            boundary_loss_enabled=True,
            boundary_loss_radius_bins=0,
            boundary_loss_weight=2.0,
        )

        self.assertAlmostEqual(float(baseline["loss"]), math.log(2.0), places=6)
        self.assertAlmostEqual(float(weighted["loss"]), 1.5 * math.log(2.0), places=6)
        self.assertTrue(torch.all(model.last_spectra == 0.0))
        self.assertEqual(weighted["condition_dropped"].tolist(), [True])
        self.assertEqual(weighted["boundary"].tolist(), [[False, True, True, False]])

    def test_mixed_depth_field_loss_weights_clean_and_corrupted_bins(self) -> None:
        per_bin = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        corrupted = torch.tensor([[True, True, False, False]])

        mixed = weighted_depth_field_loss(
            per_bin,
            corrupted,
            corrupted_loss_weight=1.0,
            uncorrupted_loss_weight=0.1,
        )
        strict = weighted_depth_field_loss(
            per_bin,
            corrupted,
            corrupted_loss_weight=1.0,
            uncorrupted_loss_weight=0.1,
            loss_on_corrupted_only=True,
        )
        clean_only = weighted_depth_field_loss(
            torch.tensor([[4.0]]),
            torch.tensor([[False]]),
            corrupted_loss_weight=1.0,
            uncorrupted_loss_weight=0.1,
        )

        self.assertAlmostEqual(float(mixed), 3.7 / 2.2, places=6)
        self.assertAlmostEqual(float(strict), 1.5, places=6)
        self.assertAlmostEqual(float(clean_only), 4.0, places=6)

    def test_mixed_depth_field_loss_rejects_invalid_weights(self) -> None:
        per_bin = torch.ones((1, 2))
        corrupted = torch.tensor([[True, False]])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            weighted_depth_field_loss(per_bin, corrupted, corrupted_loss_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            weighted_depth_field_loss(
                per_bin,
                corrupted,
                corrupted_loss_weight=0.0,
                uncorrupted_loss_weight=0.0,
            )

    def test_amp_dtype_aliases_and_cpu_disable(self) -> None:
        self.assertEqual(normalize_amp_dtype("fp16"), "float16")
        self.assertEqual(normalize_amp_dtype("bf16"), "bfloat16")
        self.assertEqual(normalize_amp_dtype("auto"), "auto")
        self.assertIsNone(resolve_amp_dtype(enabled=True, device=torch.device("cpu"), requested="auto"))
        with self.assertRaisesRegex(ValueError, "Unknown AMP dtype"):
            normalize_amp_dtype("float8")

    def test_finite_flags_and_statistics_report_nan(self) -> None:
        finite = torch.tensor([1.0, 2.0])
        invalid = torch.tensor([1.0, float("nan"), float("inf")])
        local, global_flags = synchronized_finite_flags(finite, invalid)
        self.assertEqual(local, [True, False])
        self.assertEqual(global_flags, [True, False])
        self.assertEqual(
            finite_tensor_stats(invalid),
            {
                "shape": [3],
                "dtype": "torch.float32",
                "numel": 3,
                "nonfinite": 2,
                "minimum": 1.0,
                "maximum": 1.0,
                "abs_maximum": 1.0,
            },
        )

    def test_metric_reduction_reports_stability_without_poisoning_loss(self) -> None:
        metrics = reduced_epoch_metrics(
            counts=torch.tensor([8, 10, 4, 5, 4, 5], dtype=torch.float64),
            loss_sum=0.5,
            batches=2,
            active_nm_sum=100.0,
            full_count=0,
            samples=2,
            seen_samples=2,
            overlimit_seen=0,
            skipped_overlimit=0,
            dz_nm=5.0,
            device=torch.device("cpu"),
            stability_counts=torch.tensor([1, 2, 3, 4], dtype=torch.float64),
        )
        self.assertEqual(metrics["loss"], 0.25)
        self.assertEqual(metrics["nonfinite_forward_steps"], 1)
        self.assertEqual(metrics["nonfinite_gradient_steps"], 2)
        self.assertEqual(metrics["amp_skipped_steps"], 3)
        self.assertEqual(metrics["optimizer_steps"], 4)

    def test_checkpoint_state_scans_find_nonfinite_tensors(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        self.assertIsNone(first_nonfinite_model_tensor(model))
        self.assertIsNone(first_nonfinite_optimizer_tensor(optimizer))

        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        self.assertEqual(first_nonfinite_model_tensor(model), "weight")

        state = next(iter(optimizer.state.values()))
        state["exp_avg"].view(-1)[0] = float("inf")
        self.assertRegex(first_nonfinite_optimizer_tensor(optimizer) or "", r"parameter_\d+\.exp_avg")

    def test_adaln_computes_finite_normalization_and_preserves_dtype(self) -> None:
        layer = AdaLayerNormGaussian(hidden_size=8, cond_dim=8)
        x = torch.randn(2, 4, 8, dtype=torch.float16)
        cond = torch.randn(2, 8, dtype=torch.float16)
        output = layer(x, cond)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())

    def test_run_epoch_skips_nonfinite_loss_without_poisoning_metrics(self) -> None:
        model = torch.nn.Linear(1, 4, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        spectra = torch.ones((1, 1), dtype=torch.float32)
        stacks = torch.zeros((1, 1), dtype=torch.long)
        fields = torch.zeros((1, 2), dtype=torch.long)
        loader = [(spectra, stacks, torch.tensor([11])), (spectra, stacks, torch.tensor([12]))]
        call_count = 0

        def fake_loss(current_model, current_spectra, *_args, **_kwargs):
            nonlocal call_count
            logits = current_model(current_spectra).view(1, 2, 2)
            loss = logits.mean()
            if call_count == 0:
                loss = loss * torch.tensor(float("nan"))
            call_count += 1
            return {
                "loss": loss,
                "logits": logits,
                "timesteps": torch.zeros(1, dtype=torch.long),
                "noised_fields": fields,
                "corrupted": torch.zeros_like(fields, dtype=torch.bool),
            }

        args = SimpleNamespace(
            max_thickness_nm=10.0,
            keep_overlimit_stacks=False,
            dz_nm=5.0,
            out_dir=str(Path.cwd()),
            grad_clip=1.0,
            max_consecutive_nonfinite_steps=2,
            void_loss_weight=0.05,
            random_replace_prob=0.15,
            corruption_config=None,
            loss_on_corrupted_only=False,
            corrupted_loss_weight=1.0,
            uncorrupted_loss_weight=1.0,
        )
        vocab = SimpleNamespace(void_id=1)
        with (
            mock.patch(
                "scripts.train_depth_field_diffusion.optollama.data.token_stack_total_thickness_nm",
                return_value=torch.zeros(1),
            ),
            mock.patch(
                "scripts.train_depth_field_diffusion.optollama.data.rasterize_stack_to_depth_field",
                return_value=fields,
            ),
            mock.patch("scripts.train_depth_field_diffusion.depth_field_training_loss", side_effect=fake_loss),
            mock.patch(
                "scripts.train_depth_field_diffusion.save_nonfinite_diagnostic",
                return_value=Path("diagnostic.json"),
            ) as diagnostic,
            mock.patch("scripts.train_depth_field_diffusion.ddp_rank", return_value=1),
        ):
            metrics = run_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scaler=scaler,
                amp_dtype=None,
                device=torch.device("cpu"),
                idx_to_token={},
                vocab=vocab,
                args=args,
                eos_idx=0,
                pad_idx=0,
                msk_idx=0,
                epoch=0,
                epochs=1,
                train=True,
            )

        self.assertEqual(metrics["nonfinite_forward_steps"], 1)
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        diagnostic.assert_called_once()


if __name__ == "__main__":
    unittest.main()
