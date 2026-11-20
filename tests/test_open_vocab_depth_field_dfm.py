from unittest.mock import patch

import pytest
import torch

from optollama.model.open_vocab_depth_field import OpenVocabularyDepthFieldConfig
from optollama.model.open_vocab_depth_field_dfm import DiscreteFlowConfig, OpenVocabularyDepthFieldDFM
from scripts.train_open_vocab_depth_field import compute_loss, run_epoch


def tiny_flow_model(
    flow_config: DiscreteFlowConfig | None = None, *, depth_patch_size: int = 1, depth_bins: int = 12
) -> OpenVocabularyDepthFieldDFM:
    """Build a CPU-sized Patch4/DFM model."""
    config = OpenVocabularyDepthFieldConfig(
        spectrum_shape=(3, 8),
        depth_bins=depth_bins,
        depth_patch_size=depth_patch_size,
        max_candidates=3,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        kernel_size=3,
        hybrid_dilations=(1, 2),
        spectrum_patch_size=2,
        spectrum_patch_stride=2,
        spectrum_encoder_blocks=1,
        spectrum_encoder_heads=2,
    )
    return OpenVocabularyDepthFieldDFM(config, flow_config or DiscreteFlowConfig())


def model_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Build candidate-local fields with a void suffix."""
    return {
        "wavelengths_nm": torch.linspace(300.0, 700.0, 8).expand(batch_size, -1),
        "target_spectrum": torch.rand(batch_size, 8, 3),
        "candidate_nk": torch.rand(batch_size, 3, 8, 2),
        "candidate_mask": torch.tensor([[True, True, False]]).expand(batch_size, -1).clone(),
        "clean_fields": torch.tensor([[0] * 4 + [1] * 3 + [3] * 5]).expand(batch_size, -1).clone(),
        "incidence_angle_deg": torch.zeros(batch_size),
        "polarization_id": torch.zeros(batch_size, dtype=torch.long),
    }


def test_source_uses_only_mask_void_and_valid_candidates() -> None:
    """Source draws respect candidate masks and configured proportions."""
    model = tiny_flow_model()
    mask = torch.tensor([[True, False, True], [False, True, False]])
    source = model.sample_source(mask, generator=torch.Generator().manual_seed(1))
    assert source.shape == (2, model.depth_bins)
    assert not bool((source[0] == 1).any())
    assert not bool((source[1] == 0).any() or (source[1] == 2).any())
    draws = torch.cat([model.sample_source(mask) for _ in range(100)], dim=1)
    assert 0.75 < float((draws == model.mask_id).float().mean()) < 0.85
    assert 0.05 < float((draws == model.void_id).float().mean()) < 0.15
    assert bool((model.sample_source(mask, deterministic=True) == model.mask_id).all())


def test_block_source_and_corruption_share_patch_aligned_blocks() -> None:
    """Source labels and time switches stay coherent within each sampled block."""
    flow = DiscreteFlowConfig(source_mode="block", block_min_bins=2, block_max_bins=4)
    model = tiny_flow_model(flow)
    candidates = model_batch()["candidate_mask"]
    source, block_ids, block_count = model._sample_block_source(
        candidates, generator=torch.Generator().manual_seed(7)
    )
    assert source.shape == block_ids.shape == (2, model.depth_bins)
    assert int(block_ids.max()) < block_count
    assert not bool((source == 2).any())
    for row in range(2):
        for block in block_ids[row].unique():
            assert source[row, block_ids[row] == block].unique().numel() == 1

    clean = model_batch()["clean_fields"]
    with patch.object(model, "_sample_block_source", return_value=(source, block_ids, block_count)):
        noisy, from_source = model.corrupt(
            clean, torch.full((2,), 0.5), candidates, generator=torch.Generator().manual_seed(8)
        )
    for row in range(2):
        for block in block_ids[row].unique():
            selected = block_ids[row] == block
            assert from_source[row, selected].unique().numel() == 1
    torch.testing.assert_close(noisy, torch.where(from_source, source, clean))


def test_block_partition_aligns_with_model_patches() -> None:
    """Block boundaries respect the model patch grid and the quadratic path."""
    flow = DiscreteFlowConfig(source_mode="block", block_min_bins=4, block_max_bins=8)
    model = tiny_flow_model(flow, depth_patch_size=4)
    candidates = model_batch(batch_size=128)["candidate_mask"]
    source, block_ids, _ = model._sample_block_source(
        candidates, generator=torch.Generator().manual_seed(11)
    )
    for start in range(0, model.depth_bins, model.depth_patch_size):
        assert bool((block_ids[:, start : start + model.depth_patch_size] == block_ids[:, start : start + 1]).all())
    assert 0.73 < float((source == model.mask_id).float().mean()) < 0.87
    clean = model_batch(batch_size=128)["clean_fields"]
    _, from_source = model.corrupt(
        clean, torch.full((128,), 0.5), candidates, generator=torch.Generator().manual_seed(12)
    )
    assert 0.18 < float(from_source.float().mean()) < 0.32


def test_block_sampler_redraws_entire_blocks() -> None:
    """Inference revises complete blocks rather than isolated depth bins."""
    flow = DiscreteFlowConfig(source_mode="block", block_min_bins=4, block_max_bins=4)
    model = tiny_flow_model(flow).eval()
    batch = model_batch(batch_size=1)
    block_ids = torch.arange(model.depth_bins).div(4, rounding_mode="floor").unsqueeze(0)
    source = torch.tensor([[model.mask_id] * 4 + [model.void_id] * 4 + [0] * 4])

    def predict(*args: object, **kwargs: object) -> torch.Tensor:
        logits = torch.full((1, model.depth_bins, model.num_materials), -100.0)
        logits[..., 1] = 100.0
        return logits

    traced: list[torch.Tensor] = []
    with patch.object(model, "_sample_block_source", return_value=(source, block_ids, 3)), patch.object(
        model, "forward", side_effect=predict
    ):
        result = model.sample(
            spectra=batch["target_spectrum"].transpose(1, 2),
            wavelengths_nm=batch["wavelengths_nm"],
            candidate_nk=batch["candidate_nk"],
            candidate_mask=batch["candidate_mask"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            polarization_id=batch["polarization_id"],
            steps=3,
            generator=torch.Generator().manual_seed(4),
            trace_step=lambda _step, _time, _endpoint, state: traced.append(state),
        )
    for state in traced:
        for block in range(3):
            assert state[0, block_ids[0] == block].unique().numel() == 1
    assert bool((result == 1).all())


def multiscale_flow(probs: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)) -> DiscreteFlowConfig:
    """Build the run 07 source and scale mixture."""
    return DiscreteFlowConfig.from_dict({
        "SOURCE_MODE": "block",
        "SOURCE_MASK_PROB": 1.0,
        "SOURCE_VOID_PROB": 0.0,
        "SOURCE_MATERIAL_PROB": 0.0,
        "BLOCK_RANGES": [[8, 16], [16, 64], [64, 128]],
        "BLOCK_SCALE_PROBS": probs,
    })


def test_multiscale_partition_alignment_coverage_and_reproducibility() -> None:
    """Partitions cover the field on the patch grid, including partial edge blocks."""
    model = tiny_flow_model(multiscale_flow((1.0, 0.0, 0.0)), depth_patch_size=4, depth_bins=324)
    candidates = model_batch(batch_size=256)["candidate_mask"]
    source, ids, count = model._sample_block_source(candidates, generator=torch.Generator().manual_seed(11))
    source_again, ids_again, count_again = model._sample_block_source(
        candidates, generator=torch.Generator().manual_seed(11)
    )
    assert source.shape == ids.shape == (256, 324)
    assert count_again == count
    torch.testing.assert_close(source_again, source)
    torch.testing.assert_close(ids_again, ids)
    assert bool((source == model.mask_id).all())
    assert bool((ids[:, 0] == 0).all()) and int(ids.max()) < count
    jumps = ids.diff(dim=1)
    assert bool(((jumps == 0) | (jumps == 1)).all())
    assert not bool(jumps[:, torch.arange(1, model.depth_bins) % 4 != 0].any())
    assert int(ids.max()) == count - 1
    assert bool((ids[:, 4] == 1).any()) and bool((ids[:, 4] == 0).any())
    torch.testing.assert_close(
        model.sample_source(candidates, generator=torch.Generator().manual_seed(11)), source
    )


@pytest.mark.parametrize("scale", [0, 1, 2, None])
def test_multiscale_draws_supported_widths_and_mixes_per_example(scale: int | None) -> None:
    """One batch contains independent widths and scale draws with the intended frequencies."""
    probs = (1 / 3,) * 3 if scale is None else tuple(float(i == scale) for i in range(3))
    flow = multiscale_flow(probs)
    bins = 320
    model = tiny_flow_model(flow, depth_patch_size=4, depth_bins=bins)
    candidates = model_batch(batch_size=1536)["candidate_mask"]
    _, ids, count = model._sample_block_source(candidates, generator=torch.Generator().manual_seed(9))
    # The field is long enough that each row contains at least one complete block.
    sizes = torch.zeros((len(ids), count), dtype=torch.long).scatter_add_(1, ids, torch.ones_like(ids))
    widths = sizes.amax(dim=1)
    if scale is not None:
        low, high = flow.block_ranges[scale]
        assert widths.unique().tolist() == list(range(low, high + 1, 4))
    else:
        # The ranges overlap at 16 and 64 bins, so check the mass at strictly interior widths.
        for (low, high), weight in zip(flow.block_ranges, flow.block_scale_probs, strict=True):
            n_widths = (high - low) // 4 + 1
            frequency = ((widths > low) & (widths < high)).float().mean().item()
            assert frequency == pytest.approx(weight * (n_widths - 2) / n_widths, abs=0.04)
    assert DiscreteFlowConfig.from_dict(flow.to_dict()) == flow


def test_multiscale_config_rejects_invalid_ranges_and_weights() -> None:
    """Fail early for invalid mixtures, oversized blocks, or nonaligned ranges."""
    with pytest.raises(ValueError, match="matching scale probabilities"):
        multiscale_flow((1.0,))
    with pytest.raises(ValueError, match="sum to one"):
        multiscale_flow((0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="ordered pair of positive integers"):
        DiscreteFlowConfig(source_mode="block", block_ranges=((16, 8),), block_scale_probs=(1.0,))
    with pytest.raises(ValueError, match="multiples of the depth patch size"):
        flow = DiscreteFlowConfig(source_mode="block", block_ranges=((8, 18),), block_scale_probs=(1.0,))
        tiny_flow_model(flow, depth_patch_size=4, depth_bins=320)
    with pytest.raises(ValueError, match="fit the field"):
        tiny_flow_model(multiscale_flow(), depth_patch_size=4, depth_bins=64)


def test_multiscale_corruption_uses_one_partition_and_unchanged_path() -> None:
    """Source-versus-clean decisions reuse the drawn partition at probability t squared."""
    model = tiny_flow_model(multiscale_flow(), depth_patch_size=4, depth_bins=320)
    candidates = model_batch(batch_size=32)["candidate_mask"]
    clean = torch.arange(model.depth_bins).remainder(2).expand(len(candidates), -1)
    times = torch.linspace(0.0, 1.0, len(candidates))
    generator = torch.Generator().manual_seed(23)
    source, ids, count = model._sample_block_source(candidates, generator=generator)
    expected = (torch.rand((len(candidates), count), generator=generator) < times[:, None].square()).gather(1, ids)
    with patch.object(model, "_sample_block_source", wraps=model._sample_block_source) as partition:
        noisy, actual = model.corrupt(clean, times, candidates, generator=torch.Generator().manual_seed(23))
    partition.assert_called_once()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(noisy, torch.where(expected, source, clean))
    assert not bool(actual[0].any()) and bool(actual[-1].all())


def test_multiscale_sampling_keeps_candidate_partitions_for_the_trajectory() -> None:
    """All steps reuse the initial per-candidate jump groups and emit valid endpoints."""
    model = tiny_flow_model(multiscale_flow(), depth_patch_size=4, depth_bins=320).eval()
    batch = model_batch(batch_size=6)
    generator = torch.Generator().manual_seed(20)
    source, ids, count = model._sample_block_source(batch["candidate_mask"], generator=generator)
    assert ids.unique(dim=0).shape[0] > 1
    labels = (0, 1, model.void_id, 0)
    expected_times = (1.0, 0.75, 0.5, 0.25)
    expected = source
    traced: list[torch.Tensor] = []

    def predict(_spectra: torch.Tensor, fields: torch.Tensor, t: torch.Tensor, **kwargs: object) -> torch.Tensor:
        torch.testing.assert_close(fields, expected)
        torch.testing.assert_close(t, torch.full_like(t, expected_times[len(traced)]))
        return fields.new_zeros((*fields.shape, model.num_materials), dtype=torch.float32)

    def assert_state(step: int, _time: float, endpoint: torch.Tensor, state: torch.Tensor) -> None:
        nonlocal expected
        flip_prob = 1 - ((len(labels) - step - 1) / (len(labels) - step)) ** 2
        flip = (torch.rand((len(source), count), generator=generator) < flip_prob).gather(1, ids)
        expected = torch.where(flip, labels[step], expected)
        torch.testing.assert_close(state, expected)
        assert bool((endpoint == labels[step]).all())
        traced.append(state)

    def endpoint(*args: object, **kwargs: object) -> tuple[torch.Tensor, None, None]:
        return torch.full_like(source, labels[len(traced)]), None, None

    with patch.object(model, "_sample_block_source", wraps=model._sample_block_source) as partition, patch.object(
        model, "forward", side_effect=predict
    ), patch.object(model, "_sample_logits", side_effect=endpoint):
        sampled = model.sample(
            spectra=batch["target_spectrum"].transpose(1, 2),
            wavelengths_nm=batch["wavelengths_nm"],
            candidate_nk=batch["candidate_nk"],
            candidate_mask=batch["candidate_mask"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            polarization_id=batch["polarization_id"],
            steps=len(labels),
            generator=torch.Generator().manual_seed(20),
            trace_step=assert_state,
        )
    partition.assert_called_once()
    assert len(traced) == len(labels) and bool((sampled == labels[-1]).all())


def test_run_06_block_config_preserves_rng_and_checkpoint_metadata() -> None:
    """Legacy fixed-range source and corruption consume exactly the original random draws."""
    legacy = {
        "source_mode": "block", "source_mask_prob": 0.8, "source_void_prob": 0.1, "source_material_prob": 0.1,
        "block_min_bins": 8, "block_max_bins": 32,
    }
    cfg = DiscreteFlowConfig.from_dict(legacy)
    assert cfg.to_dict() == legacy
    model = tiny_flow_model(cfg, depth_patch_size=4, depth_bins=2000)
    candidates = model_batch(batch_size=5)["candidate_mask"]
    clean = torch.zeros((5, model.depth_bins), dtype=torch.long)
    rng = torch.Generator().manual_seed(37)
    width = torch.randint(2, 9, (5, 1), generator=rng)
    offset = (torch.rand((5, 1), generator=rng) * width).long()
    ids = (torch.arange(2000)[None] // 4 + offset) // width
    block_count = 2000 // 8 + 1
    u = torch.rand((5, block_count), generator=rng)
    material = torch.multinomial(candidates.float(), block_count, replacement=True, generator=rng)
    source = torch.where(u < 0.8, model.mask_id, torch.where(u < 0.8 + 0.1, model.void_id, material)).gather(1, ids)
    expected = (torch.rand((5, block_count), generator=rng) < 0.25).gather(1, ids)
    actual_rng = torch.Generator().manual_seed(37)
    noisy, corrupted = model.corrupt(clean, torch.full((5,), 0.5), candidates, generator=actual_rng)
    torch.testing.assert_close(corrupted, expected)
    torch.testing.assert_close(noisy, torch.where(expected, source, clean))
    torch.testing.assert_close(actual_rng.get_state(), rng.get_state())
    multiscale = tiny_flow_model(multiscale_flow(), depth_patch_size=4, depth_bins=2000)
    multiscale.load_state_dict(model.state_dict(), strict=True)


def test_legacy_flow_config_defaults_to_iid() -> None:
    """Old checkpoint metadata still resolves to the original IID process."""
    saved = {"source_mask_prob": 0.7, "source_void_prob": 0.2, "source_material_prob": 0.1}
    flow = DiscreteFlowConfig.from_dict(saved)
    assert flow.source_mode == "iid"
    assert flow.source_mask_prob == 0.7
    assert DiscreteFlowConfig.from_dict(flow.to_dict()) == flow


def test_flow_path_endpoints_and_gradients() -> None:
    """The path reaches its endpoints and supports posterior training."""
    torch.manual_seed(2)
    model = tiny_flow_model()
    batch = model_batch()
    source, from_source = model.corrupt(batch["clean_fields"], torch.ones(2), batch["candidate_mask"])
    assert bool(from_source.all())
    assert bool((source != batch["clean_fields"]).any())
    clean, from_source = model.corrupt(batch["clean_fields"], torch.zeros(2), batch["candidate_mask"])
    assert not bool(from_source.any())
    torch.testing.assert_close(clean, batch["clean_fields"])
    train_cfg = {
        "VOID_LOSS_WEIGHT": 1.0,
        "RANDOM_REPLACE_PROB": 0.0,
        "CORRUPTED_LOSS_WEIGHT": 1.0,
        "UNCORRUPTED_LOSS_WEIGHT": 1.0,
        "BOUNDARY_LOSS": {"ENABLED": False},
    }
    from optollama.model import DepthFieldCorruptionConfig

    direct = model.training_loss(batch, timesteps=torch.tensor([0.25, 0.75]))
    assert torch.isfinite(direct["loss"])
    result = compute_loss(model, batch, train_cfg, DepthFieldCorruptionConfig())
    assert torch.isfinite(result["loss"])
    assert result["flow_diagnostics"].shape == (8, 3)
    result["loss"].backward()
    assert model.pointer_query.weight.grad is not None


def test_validation_aggregates_flow_diagnostics() -> None:
    """Validation history includes state and time-band diagnostics."""
    from optollama.model import DepthFieldCorruptionConfig

    model = tiny_flow_model()
    batch = model_batch()
    batch["sample_mask"] = torch.ones(2, dtype=torch.bool)
    metrics = run_epoch(
        model,
        [batch],
        torch.device("cpu"),
        {"RANDOM_REPLACE_PROB": 0.0, "VOID_LOSS_WEIGHT": 1.0, "BOUNDARY_LOSS": {"ENABLED": False}},
        DepthFieldCorruptionConfig(),
        optimizer=None,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        amp_dtype=None,
        epoch=0,
        epochs=1,
        max_steps=1,
        base_lr=0.0,
        lr_schedule=None,
        global_samples_seen=0,
    )
    assert metrics["samples"] == 2
    assert metrics["mask_bins"] + metrics["wrong_material_bins"] + metrics["wrong_void_bins"] <= 24
    assert sum(metrics[f"t_{a}_{b}_bins"] for a, b in (("0", "25"), ("25", "50"), ("50", "75"), ("75", "100"))) == 24


def test_sampler_can_revise_a_committed_material_and_void() -> None:
    """A changing posterior can revise material and void decisions."""
    model = tiny_flow_model().eval()
    batch = model_batch(batch_size=1)
    start = torch.full((1, model.depth_bins), 0, dtype=torch.long)
    labels = [1, model.void_id, 0]
    calls = iter(labels)

    def predict(*args: object, **kwargs: object) -> torch.Tensor:
        label = next(calls)
        logits = torch.full((1, model.depth_bins, model.num_materials), -100.0)
        logits[..., label] = 100.0
        return logits

    traced: list[torch.Tensor] = []
    with patch.object(model, "sample_source", return_value=start), patch.object(model, "forward", side_effect=predict):
        result = model.sample(
            spectra=batch["target_spectrum"].transpose(1, 2),
            wavelengths_nm=batch["wavelengths_nm"],
            candidate_nk=batch["candidate_nk"],
            candidate_mask=batch["candidate_mask"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            polarization_id=batch["polarization_id"],
            steps=3,
            deterministic=True,
            trace_step=lambda _step, _time, _endpoint, state: traced.append(state),
        )
    assert [int(state[0, 0]) for state in traced] == [1, model.void_id, 0]
    assert bool((result == 0).all())
    assert not bool((result == model.mask_id).any())
