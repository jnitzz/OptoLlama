from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import safetensors.torch
import torch

import optollama.data
from scripts.mine_depth_field_solutions import select_topology_distinct_candidates
from scripts.train_depth_field_diffusion import run_epoch

# ruff: noqa: D101,D102,D103


def bank_tensors() -> dict[str, torch.Tensor]:
    spectra = torch.ones((4, 3, 5), dtype=torch.float32)
    fields = torch.tensor(
        [
            [0, 0, 1, 2, 2, 2],
            [0, 1, 1, 2, 2, 2],
            [1, 1, 0, 0, 2, 2],
            [1, 0, 0, 0, 2, 2],
        ],
        dtype=torch.int16,
    )
    return {
        "anchor_spectra": spectra,
        "fields": fields,
        "pred_spectra": spectra.clone(),
        "anchor_indices": torch.tensor([10, 10, 20, 20]),
        "candidate_indices": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        "level_mae": torch.tensor([0.005, 0.008, 0.012, 0.018]),
        "derivative_mae": torch.tensor([0.001, 0.002, 0.003, 0.004]),
        "score": torch.tensor([0.00525, 0.0085, 0.01275, 0.019]),
        "quality_tier": torch.tensor([2, 2, 1, 1], dtype=torch.uint8),
        "topology_hash": torch.tensor([101, 102, 201, 202]),
        "run_count": torch.tensor([2, 2, 2, 2], dtype=torch.int32),
        "active_bins": torch.tensor([3, 3, 4, 4], dtype=torch.int32),
    }


class DepthFieldSolutionBankTests(unittest.TestCase):
    def test_compaction_and_spectrum_metrics(self) -> None:
        compacted = optollama.data.compact_depth_fields(torch.tensor([[0, 2, 1, 2, 0]]), void_id=2)
        self.assertEqual(compacted.tolist(), [[0, 1, 0, 2, 2]])

        target = torch.zeros((1, 3, 4))
        predicted = target.clone()
        predicted[:, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3])
        level, derivative = optollama.data.spectrum_error_metrics(
            predicted,
            target,
            wavelengths=torch.tensor([400, 500, 600, 700]),
            channels=(0, 2),
        )
        self.assertAlmostEqual(float(level), 0.075, places=6)
        self.assertAlmostEqual(float(derivative), 0.05, places=6)

    def test_candidate_selection_deduplicates_topologies(self) -> None:
        selected, tiers = select_topology_distinct_candidates(
            level_mae=torch.tensor([0.005, 0.006, 0.008, 0.015, 0.03]),
            score=torch.tensor([0.005, 0.006, 0.008, 0.015, 0.03]),
            topology_hash=torch.tensor([11, 11, 12, 13, 14]),
            run_count=torch.tensor([3, 3, 4, 5, 6]),
            active_bins=torch.tensor([10, 10, 10, 10, 10]),
            batch_size=1,
            mc_samples=5,
            gold_max_mae=0.01,
            silver_max_mae=0.02,
            keep_gold=3,
            keep_silver=2,
            max_runs=10,
        )
        self.assertEqual(selected.tolist(), [0, 2, 3])
        self.assertEqual(tiers.tolist(), [2, 2, 1])

    def test_writer_and_replay_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            writer = optollama.data.SolutionBankShardWriter(
                temp_dir,
                rank=0,
                shard_size=2,
                metadata={"material_names": ["A", "B", "<VOID>"], "dz_nm": 5.0},
            )
            writer.append(bank_tensors())
            paths = writer.close()
            self.assertEqual(len(paths), 1)

            replay = optollama.data.DepthFieldSolutionBankReplay(
                temp_dir,
                replay_fraction=0.3,
                gold_fraction=2.0 / 3.0,
                seed=3,
                expected_spectrum_shape=(3, 5),
                expected_depth_bins=6,
                expected_material_names=("A", "B", "<VOID>"),
                expected_dz_nm=5.0,
            )
            spectra = torch.zeros((10, 3, 5))
            fields = torch.full((10, 6), 2, dtype=torch.long)
            mixed_spectra, mixed_fields, count = replay.mix_batch(
                spectra,
                fields,
                epoch=1,
                batch_index=7,
                rank=0,
            )
            self.assertEqual(count, 3)
            self.assertEqual(int((mixed_spectra.sum(dim=(1, 2)) > 0).sum()), 3)
            self.assertEqual(int((mixed_fields != 2).any(dim=1).sum()), 3)
            self.assertEqual(replay.summary()["samples"], 4)

    def test_sharded_dataset_start_index_selects_next_anchor_range(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "samples-0.safetensors"
            safetensors.torch.save_file(
                {
                    "spectra": torch.arange(10, dtype=torch.float32).view(10, 1, 1).repeat(1, 3, 2),
                    "thin_films": torch.arange(10, dtype=torch.long).view(10, 1),
                },
                str(path),
            )
            dataset = optollama.data.ShardedSpectraDataset(
                temp_dir,
                split="train",
                subset_n=4,
                start_index=3,
                rank=0,
                world_size=1,
                shuffle=False,
            )
            indices = [int(item[2]) for item in dataset]
            self.assertEqual(indices, [3, 4, 5, 6])

    def test_training_epoch_reports_realized_replay_fraction(self) -> None:
        class TinyModel(torch.nn.Module):
            timesteps = 1
            num_materials = 3

            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(3))

            def corrupt(self, clean_fields, _timesteps, **_kwargs):
                return clean_fields, torch.ones_like(clean_fields, dtype=torch.bool)

            def forward(self, _spectra, fields, _timesteps):
                return self.bias.view(1, 1, 3).expand(fields.size(0), fields.size(1), -1)

        class Replay:
            def mix_batch(self, spectra, fields, **_kwargs):
                spectra = spectra.clone()
                fields = fields.clone()
                spectra[:2] = 1.0
                fields[:2] = 0
                return spectra, fields, 2

        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        spectra = torch.zeros((4, 3, 5))
        stacks = torch.zeros((4, 1), dtype=torch.long)
        loader = [(spectra, stacks, torch.arange(4))]
        args = SimpleNamespace(
            max_thickness_nm=10.0,
            keep_overlimit_stacks=False,
            dz_nm=5.0,
            out_dir=str(Path.cwd()),
            grad_clip=1.0,
            max_consecutive_nonfinite_steps=2,
            void_loss_weight=0.1,
            random_replace_prob=0.1,
            corruption_config=None,
            loss_on_corrupted_only=False,
            corrupted_loss_weight=1.0,
            uncorrupted_loss_weight=0.1,
        )
        vocab = SimpleNamespace(void_id=2)
        with (
            mock.patch(
                "scripts.train_depth_field_diffusion.optollama.data.token_stack_total_thickness_nm",
                return_value=torch.zeros(4),
            ),
            mock.patch(
                "scripts.train_depth_field_diffusion.optollama.data.rasterize_stack_to_depth_field",
                return_value=torch.full((4, 2), 2, dtype=torch.long),
            ),
        ):
            metrics = run_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scaler=torch.amp.GradScaler("cpu", enabled=False),
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
                solution_bank=Replay(),
            )
        self.assertEqual(metrics["solution_bank_replay_samples"], 2)
        self.assertAlmostEqual(metrics["solution_bank_replay_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
