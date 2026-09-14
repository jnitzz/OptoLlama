from __future__ import annotations

import math

import torch

from scripts.inference_open_vocab_depth_field import (
    apply_fixed_candidate_bank,
    candidate_bank_names,
    local_fields_to_runs,
    summarize_split_records,
)


def test_candidate_banks_and_local_fields_decode_per_row() -> None:
    """Resolve candidate ids and compact local depth fields into material runs."""
    banks = candidate_bank_names(
        torch.tensor([[2, 0, -1], [1, 2, -1]]),
        torch.tensor([[True, True, False], [True, True, False]]),
        ("A", "B", "C"),
    )
    assert banks == [("C", "A"), ("B", "C")]
    fields = torch.tensor([[0, 0, 1, 1, 2, 2]])
    assert local_fields_to_runs(fields, banks[0], void_id=2, dz_nm=5.0) == [
        [
            {"material": "C", "thickness_nm": 10.0},
            {"material": "A", "thickness_nm": 10.0},
        ]
    ]


def test_fixed_candidate_bank_remaps_local_fields_to_catalog_order() -> None:
    """Remove target-dependent candidate ordering before benchmark inference."""
    batch = {
        "clean_fields": torch.tensor([[0, 0, 1, 1, 3]]),
        "candidate_global_ids": torch.tensor([[2, 0, -1]]),
        "candidate_mask": torch.tensor([[True, True, False]]),
        "candidate_nk": torch.zeros((1, 3, 4, 2)),
    }
    curves = torch.rand(3, 4, 2)
    fixed = apply_fixed_candidate_bank(batch, curves, max_candidates=3)
    assert fixed["clean_fields"].tolist() == [[2, 2, 0, 0, 3]]
    assert fixed["candidate_global_ids"].tolist() == [[0, 1, 2]]
    assert fixed["candidate_mask"].tolist() == [[True, True, True]]
    torch.testing.assert_close(fixed["candidate_nk"][0], curves)


def test_split_summary_distinguishes_single_mean_and_best_of_mc() -> None:
    """Report single-draw and oracle MC metrics separately."""
    records = [
        {
            "single_draw_mae": 0.3,
            "mean_candidate_mae": 0.2,
            "best_mae": 0.1,
            "single_draw_channel_mae": {"R": 0.2, "A": 0.3, "T": 0.4},
            "mean_candidate_channel_mae": {"R": 0.1, "A": 0.2, "T": 0.3},
            "best_channel_mae": {"R": 0.05, "A": 0.1, "T": 0.15},
        },
        {
            "single_draw_mae": 0.5,
            "mean_candidate_mae": 0.4,
            "best_mae": 0.2,
            "single_draw_channel_mae": {"R": 0.4, "A": 0.5, "T": 0.6},
            "mean_candidate_channel_mae": {"R": 0.3, "A": 0.4, "T": 0.5},
            "best_channel_mae": {"R": 0.1, "A": 0.2, "T": 0.3},
        },
    ]
    summary = summarize_split_records(records)
    assert summary["num_samples"] == 2
    assert math.isclose(summary["rat_mae"]["single_draw"]["mean"], 0.4)
    assert math.isclose(summary["rat_mae"]["mean_candidate"]["mean"], 0.3)
    assert math.isclose(summary["rat_mae"]["best_of_mc"]["mean"], 0.15)
    assert math.isclose(summary["channel_mae_mean"]["best"]["T"], 0.225, rel_tol=1e-6)
