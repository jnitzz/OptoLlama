from unittest.mock import patch

import torch

from optollama.model.open_vocab_depth_field import OpenVocabularyDepthFieldConfig
from optollama.model.open_vocab_depth_field_dfm import DiscreteFlowConfig, OpenVocabularyDepthFieldDFM
from scripts.train_open_vocab_depth_field import compute_loss, run_epoch


def tiny_flow_model(
    flow_config: DiscreteFlowConfig | None = None, *, depth_patch_size: int = 1
) -> OpenVocabularyDepthFieldDFM:
    """Build a CPU-sized Patch4/DFM model."""
    config = OpenVocabularyDepthFieldConfig(
        spectrum_shape=(3, 8),
        depth_bins=12,
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
