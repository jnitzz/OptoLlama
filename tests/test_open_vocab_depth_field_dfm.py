from unittest.mock import patch

import torch

from optollama.model.open_vocab_depth_field import OpenVocabularyDepthFieldConfig
from optollama.model.open_vocab_depth_field_dfm import DiscreteFlowConfig, OpenVocabularyDepthFieldDFM
from scripts.train_open_vocab_depth_field import compute_loss, run_epoch


def tiny_flow_model() -> OpenVocabularyDepthFieldDFM:
    """Build a CPU-sized Patch4/DFM model."""
    config = OpenVocabularyDepthFieldConfig(
        spectrum_shape=(3, 8),
        depth_bins=12,
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
    return OpenVocabularyDepthFieldDFM(config, DiscreteFlowConfig())


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
