from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import torch

from optollama.data.open_layer import (
    MaterialCatalog,
    OpenLayerBatchCollator,
    ThicknessTransform,
    layer_batch_to_runs,
    load_open_layer_target,
    sample_query_indices,
)
from optollama.model.open_layer_flow import OpenLayerFlow, OpenLayerFlowConfig, layer_slot_corruption_mask
from scripts.train_open_layer_flow import averaged_metrics, make_loader, run_loss_epoch

# ruff: noqa: D103


def synthetic_catalog() -> MaterialCatalog:
    wavelengths = np.asarray([300.0, 500.0, 700.0], dtype=np.float64)
    return MaterialCatalog(
        names=("A", "B", "C"),
        wavelengths_nm=(wavelengths, wavelengths, wavelengths),
        n_values=(np.asarray([1.5, 1.6, 1.7]), np.asarray([2.0, 2.1, 2.2]), np.asarray([2.5, 2.4, 2.3])),
        k_values=(np.zeros(3), np.asarray([0.1, 0.2, 0.3]), np.asarray([0.3, 0.2, 0.1])),
    )


def tiny_model(**overrides: object) -> OpenLayerFlow:
    values = {
        "d_model": 32,
        "n_blocks": 2,
        "n_heads": 4,
        "query_encoder_blocks": 1,
        "max_layers": 6,
        "wavelength_fourier_bands": 2,
    }
    values.update(overrides)
    return OpenLayerFlow(OpenLayerFlowConfig(**values))  # type: ignore[arg-type]


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


def test_query_shapes_can_be_synchronized_without_sharing_positions() -> None:
    shape_a = torch.Generator().manual_seed(41)
    shape_b = torch.Generator().manual_seed(41)
    content_a = torch.Generator().manual_seed(3)
    content_b = torch.Generator().manual_seed(19)
    position_differences = 0
    for _ in range(32):
        indices_a = sample_query_indices(
            281,
            min_points=64,
            max_points=281,
            mode="mixed",
            generator=content_a,
            shape_generator=shape_a,
        )
        indices_b = sample_query_indices(
            281,
            min_points=64,
            max_points=281,
            mode="mixed",
            generator=content_b,
            shape_generator=shape_b,
        )
        assert len(indices_a) == len(indices_b)
        position_differences += int(not torch.equal(indices_a, indices_b))
    assert position_differences > 0


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


def test_hybrid_layer_corruption_respects_budget_and_builds_spans() -> None:
    config = tiny_model(
        material_process="full_remask",
        material_corruption_mode="hybrid",
        material_iid_fraction=0.0,
        material_span_fraction=1.0,
        material_span_min_layers=2,
        material_span_max_layers=2,
        material_span_scale_with_noise=False,
    ).config
    active = torch.tensor([[True, True, True, True, True, True]])
    selected = layer_slot_corruption_mask(
        active,
        torch.tensor([0.5]),
        config=config,
        generator=torch.Generator().manual_seed(5),
    )
    assert selected.sum().item() == 3
    assert torch.any(selected[:, :-1] & selected[:, 1:])


def test_noise_complement_keeps_high_noise_endpoint_all_mask() -> None:
    model = tiny_model(
        material_process="full_remask",
        material_corruption_mode="hybrid",
        material_iid_fraction=0.3,
        material_span_fraction=0.7,
        material_random_replace_prob=1.0,
        material_random_replace_schedule="noise_complement",
    )
    clean = torch.tensor([[0, 1, 2, 0]])
    active = torch.ones_like(clean, dtype=torch.bool)
    candidates = torch.ones((1, 3), dtype=torch.bool)
    noised, corrupted, replaced = model.corrupt_materials(clean, active, candidates, torch.ones(1))
    assert torch.all(corrupted)
    assert not torch.any(replaced)
    assert torch.all(noised == model.MASK_MATERIAL)


def test_random_replacements_are_valid_wrong_local_candidates() -> None:
    model = tiny_model(
        material_process="full_remask",
        material_random_replace_prob=1.0,
        material_random_replace_schedule="constant",
    )
    clean = torch.tensor([[0, 1, 2, 0]])
    active = torch.ones_like(clean, dtype=torch.bool)
    candidates = torch.ones((1, 3), dtype=torch.bool)
    noised, corrupted, replaced = model.corrupt_materials(
        clean,
        active,
        candidates,
        torch.ones(1),
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.all(corrupted & replaced)
    assert torch.all((noised >= 0) & (noised < 3))
    assert torch.all(noised != clean)

    sparse_clean = torch.tensor([[0, 2]])
    sparse_candidates = torch.tensor([[True, False, True]])
    sparse_noised, _, sparse_replaced = model.corrupt_materials(
        sparse_clean,
        torch.ones_like(sparse_clean, dtype=torch.bool),
        sparse_candidates,
        torch.ones(1),
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.all(sparse_replaced)
    assert sparse_noised.tolist() == [[2, 0]]


def test_full_remask_trains_visible_slots_and_samples_without_masks() -> None:
    torch.manual_seed(8)
    model = tiny_model(
        material_process="full_remask",
        material_corruption_mode="hybrid",
        material_iid_fraction=0.3,
        material_span_fraction=0.7,
        material_uncorrupted_loss_weight=0.1,
    )
    condition = synthetic_condition(batch=1)
    batch = {
        **condition,
        "material_targets": torch.tensor([[0, 1, 2]]),
        "thickness_targets": torch.tensor([[-0.5, 0.2, 0.4]]),
        "layer_mask": torch.ones((1, 3), dtype=torch.bool),
    }
    losses = model.training_loss(batch, timesteps=torch.zeros(1))
    assert losses["corrupted"].sum().item() == 0
    assert losses["material_loss"].item() > 0.0
    losses["loss"].backward()

    sampled = model.eval().sample(**condition, layer_counts=torch.tensor([3]), steps=4)
    assert torch.all(sampled["material_ids"] >= 0)


def test_full_remask_rewrites_already_visible_materials() -> None:
    model = tiny_model(material_process="full_remask")
    condition = synthetic_condition(batch=1)
    proposal_steps: list[int] = []

    def fake_encode(*_args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        empty = torch.empty(0)
        return empty, empty

    def fake_forward(**kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        material_ids = kwargs["material_ids"]
        candidate = len(proposal_steps) % 3
        proposal_steps.append(candidate)
        logits = torch.full((*material_ids.shape, 3), -10.0)
        logits[:, :, candidate] = 10.0
        return {"material_logits": logits, "thickness_velocity": torch.zeros_like(kwargs["thickness_state"])}

    model.encode_condition = fake_encode  # type: ignore[method-assign]
    model.forward = fake_forward  # type: ignore[method-assign]
    sampled = model.sample(
        **condition,
        layer_counts=torch.tensor([3]),
        steps=3,
        deterministic=True,
        generator=torch.Generator().manual_seed(11),
    )
    assert proposal_steps == [0, 1, 2]
    assert sampled["material_ids"].tolist() == [[2, 2, 2]]


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


def test_sharded_open_layer_loader_forces_single_worker_and_finite_length() -> None:
    class FakeShardedDataset(torch.utils.data.IterableDataset):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()

        def __len__(self) -> int:
            return 8

        def __iter__(self):
            yield from range(8)

    cfg = {
        "DATA_PATH_TRAIN": "unused",
        "TRAIN_BATCH_SIZE": 2,
        "NUM_WORKERS": 8,
        "SHARDED_LOADING": True,
    }
    with mock.patch("optollama.data.ShardedSpectraDataset", FakeShardedDataset):
        _, loader = make_loader(
            cfg,
            split="train",
            collator=lambda batch: batch,  # type: ignore[arg-type]
            subset_n=8,
            rank=0,
            world_size=1,
        )
    assert loader.num_workers == 0
    assert len(loader) == 4
    assert list(loader) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_averaged_metrics_are_sample_weighted() -> None:
    totals = torch.tensor([10.0, 3.0, 4.0])
    assert averaged_metrics(totals, ("loss", "material_accuracy")) == {
        "loss": 2.5,
        "material_accuracy": 0.75,
    }


def test_open_layer_epoch_skips_nonfinite_gradient_without_poisoning_weights() -> None:
    class InfiniteBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
            del ctx
            return value * 0.0 + 1.0

        @staticmethod
        def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
            del ctx
            return (torch.full_like(grad_output, float("inf")),)

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    calls = 0

    def fake_training_loss(_model: torch.nn.Module, _batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        loss = InfiniteBackward.apply(model.weight.sum()) if calls == 1 else model.weight.square().sum()
        zero = loss.detach() * 0.0
        return {
            "loss": loss,
            "material_loss": zero,
            "thickness_loss": zero,
            "material_accuracy": zero,
            "full_material_accuracy": zero,
            "mean_timestep": zero,
            "corrupted_fraction": zero,
            "masked_fraction": zero,
            "replaced_fraction": zero,
            "supervised_samples": torch.ones((), dtype=torch.long),
        }

    loader = [{"dummy": torch.zeros(1)}, {"dummy": torch.zeros(1)}]
    with mock.patch("scripts.train_open_layer_flow.compute_training_loss", side_effect=fake_training_loss):
        metrics = run_loss_epoch(
            model=model,
            loader=loader,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=None,
            grad_clip=1.0,
            epoch=0,
            epochs=1,
            max_consecutive_nonfinite_steps=2,
        )
    assert metrics["nonfinite_gradient_steps"] == 1
    assert metrics["optimizer_steps"] == 1
    assert torch.isfinite(model.weight).all()
