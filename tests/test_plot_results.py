from __future__ import annotations

import pytest

from scripts.plot_results import _sample_plot_filename, select_sample_result


def test_select_sample_result_merges_mc_candidate_over_parent() -> None:
    results = [
        {
            "dataset_index": 4,
            "target_spectra": [[1.0]],
            "tokens": ["parent"],
            "all_mc": [
                {"tokens": ["candidate-0"], "mae": 0.2},
                {"tokens": ["candidate-1"], "mae": 0.1},
            ],
        }
    ]

    selected = select_sample_result(results, sample_index=0, mc_index=1)

    assert selected == {
        "dataset_index": 4,
        "target_spectra": [[1.0]],
        "tokens": ["candidate-1"],
        "mae": 0.1,
    }
    assert "all_mc" not in selected


def test_select_sample_result_rejects_missing_or_invalid_mc_index() -> None:
    with pytest.raises(ValueError, match="no recorded all_mc"):
        select_sample_result([{"mae": 0.1}], sample_index=0, mc_index=0)

    with pytest.raises(IndexError, match="MC index 2 is out of range"):
        select_sample_result([{"all_mc": [{"mae": 0.1}]}], sample_index=0, mc_index=2)


def test_sample_plot_filename_includes_mc_index() -> None:
    path = "samples-260805-0000.json"

    assert _sample_plot_filename(path, 0) == "sample_0_260805-0000.pdf"
    assert _sample_plot_filename(path, 0, 667) == "sample_0_mc_667_260805-0000.pdf"
