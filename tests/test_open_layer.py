from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from optollama.data.open_layer import (
    MaterialCatalog,
    OpenLayerBatchCollator,
    ThicknessTransform,
    layer_batch_to_runs,
    load_open_layer_target,
)
from optollama.model.open_layer_flow import OpenLayerFlow, OpenLayerFlowConfig

# ruff: noqa: D103


def synthetic_catalog() -> MaterialCatalog:
    wavelengths = np.asarray([300.0, 500.0, 700.0], dtype=np.float64)
    return MaterialCatalog(
        names=("A", "B", "C"),
        wavelengths_nm=(wavelengths, wavelengths, wavelengths),
        n_values=(np.asarray([1.5, 1.6, 1.7]), np.asarray([2.0, 2.1, 2.2]), np.asarray([2.5, 2.4, 2.3])),
        k_values=(np.zeros(3), np.asarray([0.1, 0.2, 0.3]), np.asarray([0.3, 0.2, 0.1])),
    )


def tiny_model() -> OpenLayerFlow:
    return OpenLayerFlow(
        OpenLayerFlowConfig(
            d_model=32,
            n_blocks=2,
            n_heads=4,
            query_encoder_blocks=1,
            max_layers=6,
            wavelength_fourier_bands=2,
        )
    )


def synthetic_condition(batch: int = 2) -> dict[str, torch.Tensor]:
    wavelengths = torch.tensor([[300.0, 500.0, 700.0]]).expand(batch, -1)
    target = torch.rand(batch, 3, 2)
    curves = synthetic_catalog().interpolate(wavelengths[0]).unsqueeze(0).expand(batch, -1, -1, -1)
    return {
        "wavelengths_nm": wavelengths,
        "target_spectrum": target,
        "query_mask": torch.ones(batch, 3, dtype=torch.bool),
        "candidate_nk": curves,
        "candidate_mask": torch.ones(batch, 3, dtype=torch.bool),
    }


def test_thickness_transform_roundtrip() -> None:
    transform = ThicknessTransform(5.0, 10_000.0)
    values = torch.tensor([5.0, 20.0, 300.0, 10_000.0])
    assert torch.allclose(transform.decode(transform.encode(values)), values, rtol=1.0e-5, atol=1.0e-4)


def test_coordinate_target_loader_accepts_long_form_csv(tmp_path: Path) -> None:
    path = tmp_path / "target.csv"
    path.write_text("wavelength_nm,R,T\n700,0.8,0.1\n300,0.2,0.7\n", encoding="ascii")
    spectra, wavelengths = load_open_layer_target(path)
    assert wavelengths.tolist() == [300.0, 700.0]
    assert torch.allclose(spectra[0], torch.tensor([0.2, 0.8]))
    assert torch.allclose(spectra[2], torch.tensor([0.7, 0.1]))


def test_collator_builds_local_material_targets_and_merges_neighbors() -> None:
    catalog = synthetic_catalog()
    tokens = ["<PAD>", "<MSK>", "<EOS>", "A_10", "A_20", "B_30"]
    idx_to_token = {idx: token for idx, token in enumerate(tokens)}
    collator = OpenLayerBatchCollator(
        wavelengths_nm=torch.tensor([300.0, 500.0, 700.0]),
        catalog=catalog,
        idx_to_token=idx_to_token,
        eos_idx=2,
        pad_idx=0,
        msk_idx=1,
        max_layers=4,
        max_candidates=3,
        min_query_points=3,
        max_query_points=3,
        query_sampling="full",
        randomize_candidates=False,
    )
    batch = collator([(torch.rand(3, 3), torch.tensor([3, 4, 5, 2, 0]), 7)])
    assert batch["layer_mask"].tolist() == [[True, True, False, False]]
    assert batch["material_targets"][0, :2].tolist() == [0, 1]
    assert batch["thickness_nm"][0, :2].tolist() == [30.0, 30.0]
    assert batch["candidate_global_ids"].tolist() == [[0, 1, 2]]


def test_collator_keeps_empty_stacks_batch_safe_but_unsupervised() -> None:
    catalog = synthetic_catalog()
    collator = OpenLayerBatchCollator(
        wavelengths_nm=torch.tensor([300.0, 500.0, 700.0]),
        catalog=catalog,
        idx_to_token={0: "<PAD>", 1: "<MSK>", 2: "<EOS>"},
        eos_idx=2,
        pad_idx=0,
        msk_idx=1,
        max_layers=3,
        max_candidates=3,
        min_query_points=3,
        max_query_points=3,
        query_sampling="full",
        randomize_candidates=False,
        random_distractors=False,
    )
    batch = collator([(torch.rand(3, 3), torch.tensor([2, 0, 0]), 121)])
    assert batch["sample_mask"].tolist() == [False]
    assert batch["layer_mask"].tolist() == [[True, False, False]]
    assert batch["material_targets"].tolist() == [[-100, -100, -100]]
    assert batch["candidate_mask"].any()


def test_material_holdout_removes_distractor_exposure_and_supervision() -> None:
    catalog = synthetic_catalog()
    tokens = ["<PAD>", "<MSK>", "<EOS>", "A_10", "B_20", "C_30"]
    collator = OpenLayerBatchCollator(
        wavelengths_nm=torch.tensor([300.0, 500.0, 700.0]),
        catalog=catalog,
        idx_to_token={idx: token for idx, token in enumerate(tokens)},
        eos_idx=2,
        pad_idx=0,
        msk_idx=1,
        max_layers=3,
        max_candidates=3,
        min_query_points=3,
        max_query_points=3,
        query_sampling="full",
        randomize_candidates=False,
        holdout_materials=("C",),
    )
    spectra = torch.rand(3, 3)
    batch = collator(
        [
            (spectra, torch.tensor([3, 4, 2]), 0),
            (spectra, torch.tensor([5, 2, 0]), 1),
        ]
    )
    assert batch["sample_mask"].tolist() == [True, False]
    assert 2 not in batch["candidate_global_ids"][0].tolist()


def test_candidate_permutation_only_permutes_pointer_logits() -> None:
    torch.manual_seed(2)
    model = tiny_model().eval()
    condition = synthetic_condition(batch=1)
    material_ids = torch.full((1, 3), -1, dtype=torch.long)
    thickness = torch.zeros(1, 3)
    layer_mask = torch.ones(1, 3, dtype=torch.bool)
    timestep = torch.tensor([0.7])
    base = model(
        **condition,
        material_ids=material_ids,
        thickness_state=thickness,
        layer_mask=layer_mask,
        timesteps=timestep,
    )["material_logits"]

    permutation = torch.tensor([2, 0, 1])
    permuted_condition = dict(condition)
    permuted_condition["candidate_nk"] = condition["candidate_nk"][:, permutation]
    permuted_condition["candidate_mask"] = condition["candidate_mask"][:, permutation]
    permuted = model(
        **permuted_condition,
        material_ids=material_ids,
        thickness_state=thickness,
        layer_mask=layer_mask,
        timesteps=timestep,
    )["material_logits"]
    assert torch.allclose(permuted, base[:, :, permutation], atol=1.0e-5, rtol=1.0e-5)


def test_training_loss_and_sampling_are_finite() -> None:
    torch.manual_seed(4)
    model = tiny_model()
    condition = synthetic_condition(batch=2)
    batch = {
        **condition,
        "material_targets": torch.tensor([[0, 1, -100], [2, 1, 0]]),
        "thickness_targets": torch.tensor([[-0.5, 0.2, -1.0], [-0.2, 0.1, 0.5]]),
        "layer_mask": torch.tensor([[True, True, False], [True, True, True]]),
    }
    losses = model.training_loss(batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])

    sampled = model.eval().sample(**condition, layer_counts=torch.tensor([2, 3]), steps=4)
    assert sampled["material_ids"].shape == (2, 3)
    assert torch.all(sampled["material_ids"][sampled["layer_mask"]] >= 0)
    assert torch.all(sampled["thickness_nm"][sampled["layer_mask"]] >= 5.0)
    assert torch.all(sampled["thickness_nm"].sum(dim=1) <= 10_000.001)


def test_all_holdout_batch_has_differentiable_zero_loss() -> None:
    model = tiny_model()
    condition = synthetic_condition(batch=1)
    batch = {
        **condition,
        "material_targets": torch.tensor([[0, 1]]),
        "thickness_targets": torch.tensor([[-0.5, 0.2]]),
        "layer_mask": torch.tensor([[True, True]]),
        "sample_mask": torch.tensor([False]),
    }
    result = model.training_loss(batch)
    result["loss"].backward()
    assert result["loss"].item() == 0.0
    assert result["supervised_samples"].item() == 0


def test_layer_batch_to_runs_merges_sampled_adjacent_materials() -> None:
    runs = layer_batch_to_runs(
        material_ids=torch.tensor([[0, 0, 1]]),
        thickness_nm=torch.tensor([[10.0, 20.0, 30.0]]),
        candidate_global_ids=torch.tensor([[2, 0, 1]]),
        catalog=synthetic_catalog(),
        layer_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert runs == [[{"material": "C", "thickness_nm": 30.0}, {"material": "A", "thickness_nm": 30.0}]]
