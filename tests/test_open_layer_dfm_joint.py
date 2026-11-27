"""Joint layer DFM geometry, coupled predictions, and training resume."""

from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from optollama.data.open_layer import MaterialCatalog, OpenLayerJointDFMCollator
from optollama.model.open_layer_dfm_joint import OpenLayerJointDFM, OpenLayerJointDFMConfig
from optollama.model.open_vocab_depth_field_dfm import dfm_step_flip_probability
from scripts import train_open_vocab_depth_field as trainer

TOKENS = ["A_17.25", "B_37.5", "<EOS>", "<PAD>", "<MSK>"]


@pytest.fixture
def example():
    """Build a batch with native lengths, adjacent equal materials and an empty stack."""
    wavelength = np.array([350.0, 650.0])
    catalog = MaterialCatalog(
        ("A", "B"),
        (wavelength, wavelength),
        (np.array([1.3, 1.5]), np.array([2.4, 2.7])),
        (np.zeros(2), np.ones(2) * 0.04),
    )
    collate = OpenLayerJointDFMCollator(
        wavelengths_nm=torch.linspace(350, 650, 9),
        catalog=catalog,
        idx_to_token=dict(enumerate(TOKENS)),
        eos_idx=2,
        pad_idx=3,
        msk_idx=4,
        max_layers=5,
        max_candidates=4,
        channels=("R", "A", "T"),
        min_layer_nm=1.0,
        max_total_nm=10000.0,
        min_query_points=8,
        max_query_points=8,
        query_sampling="coordinate_full",
        incidence_angle_deg=12,
        polarization="p",
        include_source_spectrum=True,
    )
    rat = torch.ones(3, 9) / 3
    model = OpenLayerJointDFM(
        OpenLayerJointDFMConfig(
            d_model=16,
            n_heads=4,
            n_blocks=2,
            ffn_multiplier=2,
            max_layers=5,
            max_candidates=4,
            query_points=8,
            query_encoder_blocks=1,
        )
    )
    return model, collate([(rat, torch.tensor([0, 0, 1, 2, 3]), 7), (rat, torch.tensor([2, 3, 3, 3, 3]), 8)])


def condition(batch):
    """Pack the coordinate/optical features used by both training and sampling."""
    return {
        "spectra": batch["target_spectrum"].transpose(1, 2),
        **{
            key: batch[key]
            for key in (
                "wavelengths_nm",
                "candidate_mask",
                "candidate_nk",
                "incidence_angle_deg",
                "polarization_id",
            )
        },
    }


@pytest.mark.parametrize("visible_weight", [1.0, 0.25, 0.0])
def test_native_layers_and_empty_stacks(example, visible_weight):
    """Do not merge or rasterize native targets, and do not discard empty stacks."""
    model, batch = example
    mask = batch["layer_mask"]
    assert mask.sum(1).tolist() == [3, 0]
    assert batch["sample_mask"].tolist() == [True, True]
    assert batch["thickness_nm"][0, :3].tolist() == [17.25, 17.25, 37.5]
    assert (batch["material_targets"][~mask] == -100).all()
    assert batch["source_spectrum_rat"].shape == (2, 9, 3)
    assert batch["wavelengths_nm"].shape == (2, 8)
    assert batch["incidence_angle_deg"].tolist() == [12, 12]
    assert batch["polarization_id"].tolist() == [1, 1]

    empty = {key: value[1:] for key, value in batch.items()}
    loss = trainer.compute_joint_loss(model, model, empty, {"CORRECT_VISIBLE_MATERIAL_WEIGHT": visible_weight})
    assert loss["material_loss"].item() == loss["thickness_loss"].item() == 0.0
    torch.testing.assert_close(loss["loss"], loss["count_loss"])
    loss["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("visible_weight", [0.25, 0.10, 0.0])
def test_material_loss_weights_masked_and_wrong_slots(example, monkeypatch, visible_weight):
    """Weight recovery slots but not padding, retaining the original validation metric."""
    model, batch = example
    clean = batch["material_targets"]
    end = model.training_state(batch, torch.ones(clean.shape[0]))
    masked, wrong = torch.zeros_like(clean, dtype=torch.bool), torch.zeros_like(clean, dtype=torch.bool)
    masked[0, 0] = True
    wrong[0, 1] = True
    material = clean.masked_fill(masked, model.mask_id)
    monkeypatch.setattr(model, "training_state", lambda *_: (material, end[1], end[2], masked))
    monkeypatch.setattr(
        model, "replace_visible_materials",
        lambda *args, **kwargs: (torch.where(wrong, 1 - material, material), wrong),
    )
    logits = torch.randn((*clean.shape, model.config.max_candidates), generator=torch.Generator().manual_seed(8))
    logits.requires_grad_()
    monkeypatch.setattr(
        model, "forward",
        lambda **kwargs: {
            "material_logits": logits, "thickness_velocity": torch.zeros_like(end[2]),
            "count_logits": torch.zeros((clean.shape[0], model.config.max_layers + 1)),
        },
    )
    config = {"CORRECT_VISIBLE_MATERIAL_WEIGHT": visible_weight, "VISIBLE_MATERIAL_REPLACE_PEAK": 0.10}
    loss = trainer.compute_joint_loss(model, model, batch, config, replacement_generator=torch.Generator())
    slot_loss = torch.nn.functional.cross_entropy(logits[0, :3], clean[0, :3], reduction="none")
    expected = (slot_loss[0] + slot_loss[1] + visible_weight * slot_loss[2]) / (2 + visible_weight)
    torch.testing.assert_close(loss["material_loss"], expected)
    loss["loss"].backward()
    assert not logits.grad[~batch["layer_mask"]].any()

    model.eval()
    validation = trainer.compute_joint_loss(model, model, batch, config)
    torch.testing.assert_close(validation["material_loss"], slot_loss.mean())
    for component in ("count_loss", "thickness_loss"):
        torch.testing.assert_close(loss[component], validation[component])


def test_thickness_geometry_and_flow_endpoints(example):
    """Keep the latent path finite at empty, minimum, and exactly capped endpoints."""
    model, batch = example
    mask = torch.tensor([[True, True], [True, True], [False, False]])
    values = torch.tensor([[1.0, 9999.0], [1.0, 1.0], [0.0, 0.0]])
    state = model.encode_thickness(values, mask)
    assert torch.isfinite(state).all()
    decoded = model.decode_thickness(state, mask)
    torch.testing.assert_close(decoded, values, rtol=0, atol=0.03)
    extreme = torch.tensor([[1000.0, 1000.0, -1000.0], [-1000.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    decoded = model.decode_thickness(extreme, mask)
    assert (decoded[mask] >= 1).all() and not decoded[~mask].any()
    assert (decoded.double().sum(1) <= 10000).all()

    layer_mask = batch["layer_mask"]
    source = model.training_state(batch, torch.zeros(2), generator=torch.Generator().manual_seed(8))
    end = model.training_state(batch, torch.ones(2), generator=torch.Generator().manual_seed(8))
    assert (source[0][layer_mask] == model.mask_id).all()
    assert (source[0][~layer_mask] == model.pad_id).all()
    assert torch.equal(end[0], batch["material_targets"])
    assert torch.equal(source[3], layer_mask) and not end[3].any()
    assert torch.isfinite(source[1]).all() and torch.isfinite(end[2]).all()
    torch.testing.assert_close(end[1] - source[1], end[2])
    for steps in (1, 4, 16):
        survival = 1.0
        for step in range(steps):
            survival *= 1 - dfm_step_flip_probability(step, steps)
            assert survival == pytest.approx((1 - (step + 1) / steps) ** 2)


def test_cross_conditioning_and_padding(example):
    """Both prediction heads read the noisy joint state and ignore padding."""
    torch.manual_seed(3)
    model, batch = example
    mask = batch["layer_mask"]
    with torch.no_grad():
        ids, state, _, _ = model.training_state(batch, torch.full((2,), 0.5))
        inputs = dict(layer_mask=mask, flow_time=torch.full((2,), 0.5), **condition(batch))
        original = model(material_ids=ids, thickness_state=state, **inputs)
        changed_state = state.clone()
        changed_state[0, 1] += 2
        changed_ids = ids.clone()
        changed_ids[0, 0] = model.mask_id if ids[0, 0] >= 0 else 0
        geometry = model(material_ids=ids, thickness_state=changed_state, **inputs)
        material = model(material_ids=changed_ids, thickness_state=state, **inputs)
        assert not torch.allclose(original["material_logits"][0, 0], geometry["material_logits"][0, 0])
        assert not torch.allclose(original["thickness_velocity"][0, 1], material["thickness_velocity"][0, 1])
        torch.testing.assert_close(geometry["count_logits"], material["count_logits"])
        noisy_pad = state.masked_fill(~model.state_mask(mask), 1000)
        padded = model(material_ids=ids.masked_fill(~mask, 999), thickness_state=noisy_pad, **inputs)
        torch.testing.assert_close(original["material_logits"][mask], padded["material_logits"][mask])
        torch.testing.assert_close(original["thickness_velocity"], padded["thickness_velocity"])


def test_reproducible_generation_uses_predicted_counts(example):
    """Spend one joint forward per step, using condition-only counts even when empty."""
    model, batch = example
    model.eval()
    count_projection = model.count_head[-1]
    with torch.no_grad():
        count_projection.weight.zero_()
        count_projection.bias.fill_(-100)
        count_projection.bias[2] = 100
    calls = []
    handle = model.register_forward_pre_hook(lambda _, __, kwargs: calls.append(kwargs["flow_time"]), with_kwargs=True)
    args = condition(batch)
    first = model.sample(steps=4, generator=torch.Generator().manual_seed(8), **args)
    handle.remove()
    second = model.sample(steps=4, generator=torch.Generator().manual_seed(8), **args)
    assert [call[0].item() for call in calls] == [0.0, 0.25, 0.5, 0.75]
    for first_tensor, second_tensor in zip(first, second, strict=True):
        torch.testing.assert_close(first_tensor, second_tensor, atol=0, rtol=0)
    assert first.layer_counts.tolist() == [2, 2]
    assert args["candidate_mask"].gather(1, first.material_ids).all()
    assert torch.isfinite(first.thickness_nm).all() and (first.thickness_nm >= 1).all()
    assert (first.thickness_nm.double().sum(1) <= 10000).all()

    with torch.no_grad():
        count_projection.bias[0] = 200
    empty = model.sample(steps=2, generator=torch.Generator().manual_seed(9), **args)
    assert empty.layer_counts.tolist() == [0, 0]
    assert (empty.material_ids == model.pad_id).all() and not empty.thickness_nm.any()


def test_reference_count_ablation_pairs_sampling(example):
    """A reference count overrides only the active slots without shifting the random stream."""
    model, batch = example
    model.eval()
    with torch.no_grad():
        model.count_head[-1].weight.zero_()
        model.count_head[-1].bias.fill_(-100)
        model.count_head[-1].bias[2] = 100
    settings = dict(steps=4, fixed_layer_slots=True, **condition(batch))
    predicted = model.sample(generator=torch.Generator().manual_seed(8), **settings)
    reference = model.sample(
        generator=torch.Generator().manual_seed(8), reference_layer_counts=torch.tensor([2, 0]), **settings,
    )
    assert predicted.layer_counts.tolist() == [2, 2]
    assert reference.layer_counts.tolist() == [2, 0]
    assert reference.material_ids.shape[1] == predicted.material_ids.shape[1] == model.config.max_layers
    for without_override, with_override in zip(predicted, reference, strict=True):
        torch.testing.assert_close(without_override[0], with_override[0], rtol=0, atol=0)
    assert not reference.layer_mask[1].any()
    assert (reference.material_ids[1] == model.pad_id).all()
    assert not reference.thickness_nm[1].any()
    with pytest.raises(ValueError, match="between zero and MAX_LAYERS"):
        model.sample(reference_layer_counts=torch.tensor([1, model.config.max_layers + 1]), **settings)


def test_reference_thickness_follows_forward_path(example):
    """Give the treatment only the noised reference geometry at each denoising step."""
    model, batch = example
    model.eval()
    mask = batch["layer_mask"]
    lengths = batch["thickness_nm"]
    latent = model.encode_thickness(lengths, mask)
    traces = []

    def capture(_, __, inputs):
        traces.append((inputs["thickness_state"].clone(), inputs["material_ids"].clone()))

    handle = model.register_forward_pre_hook(capture, with_kwargs=True)
    settings = dict(steps=4, fixed_layer_slots=True, reference_layer_counts=mask.sum(dim=1), **condition(batch))
    model.sample(generator=torch.Generator().manual_seed(8), **settings)
    control = traces.copy()
    traces.clear()
    reference = model.sample(generator=torch.Generator().manual_seed(8), reference_thickness_nm=lengths, **settings)
    handle.remove()
    torch.testing.assert_close(traces[0][0], control[0][0], rtol=0, atol=0)
    torch.testing.assert_close(traces[0][1], control[0][1], rtol=0, atol=0)
    torch.testing.assert_close(traces[1][1], control[1][1], rtol=0, atol=0)
    for step, (state, _) in enumerate(traces):
        torch.testing.assert_close(state, torch.lerp(control[0][0], latent, step / 4))
    torch.testing.assert_close(reference.thickness_nm, lengths, rtol=0, atol=0)
    assert (reference.material_ids[mask] >= 0).all()
    assert (reference.material_ids[~mask] == model.pad_id).all()
    with pytest.raises(ValueError, match="within the thickness budget"):
        model.sample(reference_thickness_nm=lengths * 1000, **settings)


def test_reference_materials_follow_scheduled_jumps(example):
    """Reveal reference labels on the original jump schedule while predicting thickness."""
    model, batch = example
    model.eval()
    mask = batch["layer_mask"]
    labels = batch["material_targets"]
    traces = []

    def capture(_, __, inputs):
        traces.append((inputs["thickness_state"].clone(), inputs["material_ids"].clone()))

    handle = model.register_forward_pre_hook(capture, with_kwargs=True)
    settings = dict(steps=4, fixed_layer_slots=True, reference_layer_counts=mask.sum(dim=1), **condition(batch))
    model.sample(generator=torch.Generator().manual_seed(17), **settings)
    control = traces.copy()
    traces.clear()
    reference = model.sample(generator=torch.Generator().manual_seed(17), reference_material_ids=labels, **settings)
    handle.remove()
    torch.testing.assert_close(traces[0][0], control[0][0], rtol=0, atol=0)
    torch.testing.assert_close(traces[1][0], control[1][0], rtol=0, atol=0)
    assert (traces[0][1][mask] == model.mask_id).all()
    for (_, state), (_, control_state) in zip(traces, control, strict=True):
        torch.testing.assert_close(state == model.mask_id, control_state == model.mask_id)
        assert (state[mask & (state != model.mask_id)] == labels[mask & (state != model.mask_id)]).all()
    torch.testing.assert_close(reference.material_ids, labels)
    assert (reference.thickness_nm[mask] >= model.config.min_thickness_nm).all()
    assert not torch.equal(reference.thickness_nm, batch["thickness_nm"])
    bad_labels = labels.masked_fill(mask, model.config.max_candidates - 1)
    with pytest.raises(ValueError, match="active candidate bank"):
        model.sample(reference_material_ids=bad_labels, **settings)


def test_visible_wrong_materials_preserve_masks_and_rng():
    """Replace only visible slots with valid alternatives, using an independent RNG."""
    ids = torch.tensor([[0, 2, -1, -100], [1, 1, -1, -100]]).repeat(3, 1)
    bank = torch.tensor([[True, False, True], [False, True, False]]).repeat(3, 1)
    times = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    state = torch.get_rng_state()
    changed, wrong = OpenLayerJointDFM.replace_visible_materials(
        ids, times, bank, peak=1.0, generator=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert changed[2].tolist() == [2, 0, -1, -100]
    torch.testing.assert_close(changed[~wrong], ids[~wrong])
    torch.testing.assert_close(wrong, ids != changed)
    assert wrong.sum() == 2


def test_joint_flow_time_mixture_and_legacy_random_stream():
    """Check the mixture distribution, determinism and disabled RNG compatibility."""
    cpu = torch.device("cpu")
    first = trainer.sample_joint_flow_time(40_000, cpu, (0.2, 0.35), generator=torch.Generator().manual_seed(7))
    duplicate = trainer.sample_joint_flow_time(40_000, cpu, (0.2, 0.35), generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(first, duplicate, rtol=0, atol=0)
    assert float((first < 0.35).float().mean()) == pytest.approx(0.48, abs=0.009)
    assert float((1 - first).square().mean()) == pytest.approx(0.4048, abs=0.008)
    assert bool(((first >= 0) & (first < 1)).all())
    legacy, current = torch.Generator().manual_seed(17), torch.Generator().manual_seed(17)
    actual = trainer.sample_joint_flow_time(9, cpu, trainer.joint_flow_time_mix({}), generator=current)
    torch.testing.assert_close(actual, torch.rand(9, generator=legacy), rtol=0, atol=0)
    torch.testing.assert_close(current.get_state(), legacy.get_state())


@pytest.mark.parametrize(("fraction", "max_s"), [(-0.1, 0.35), (1.1, 0.35), (0.2, 0.0), (0.2, 1.01), (float("nan"), 0.35)])
def test_joint_flow_time_rejects_invalid_config(fraction, max_s):
    """Reject invalid mixture bounds and probabilities before starting training."""
    with pytest.raises(ValueError, match="FLOW_TIME_SAMPLING"):
        trainer.joint_flow_time_mix({"FLOW_TIME_SAMPLING": {"HIGH_NOISE_FRACTION": fraction, "HIGH_NOISE_MAX_S": max_s}})


def test_joint_flow_time_mixture_does_not_change_validation(example):
    """Use the training mixture only in train mode, with the same validation RNG."""
    model, batch = example
    settings = {"FLOW_TIME_SAMPLING": {"HIGH_NOISE_FRACTION": 1.0, "HIGH_NOISE_MAX_S": 0.01}}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6)
        train = trainer.compute_joint_loss(model, model, batch, settings)
        assert bool((train["diagnostic_timesteps"] < 0.01).all())
        model.eval()
        torch.manual_seed(6)
        plain = trainer.compute_joint_loss(model, model, batch, {})
        torch.manual_seed(6)
        mixed = trainer.compute_joint_loss(model, model, batch, settings)
    for key in ("diagnostic_timesteps", "loss", "material_loss", "thickness_loss", "count_loss"):
        torch.testing.assert_close(plain[key], mixed[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("replacement_peak", "visible_weight", "high_noise_fraction"),
    [(0.0, 1.0, 0.0), (0.10, 1.0, 0.0), (0.10, 0.25, 0.0), (0.10, 1.0, 0.2)],
)
def test_mid_epoch_checkpoint_resume(tmp_path, monkeypatch, replacement_peak, visible_weight, high_noise_fraction):
    """Restore optimizer, RNG, loader cursor and EMA and reproduce the completed epoch."""
    data = tmp_path / "data"
    (data / "materials").mkdir(parents=True)
    (data / "tokens.json").write_text(json.dumps(TOKENS), encoding="utf-8")
    for name, index in (("A", 1.4), ("B", 2.4)):
        (data / "materials" / f"{name}.csv").write_text(
            f"nm,n,k\n350,{index},0\n650,{index + 0.1},0\n",
            encoding="utf-8",
        )
    tensors = {
        "thin_films": torch.tensor([[0, 1, 2], [2, 3, 3], [1, 0, 2], [0, 2, 3], [1, 2, 3], [0, 0, 2]]),
        "spectra": torch.rand(6, 3, 7),
    }
    for split in ("train", "test"):
        save_file(tensors, data / f"{split}.safetensors")
    config = {
        "EXPERIMENT": {"MODEL": "open_layer_dfm_joint", "RUN": "unused", "SEED": 17},
        "PATHS": {
            "DATA_PATH": str(data),
            "MATERIALS_PATH": str(data / "materials"),
            "TOKENS_PATH": str(data / "tokens.json"),
            "OUTPUT_PATH": str(data / "output"),
            "TRAIN": [str(data / "train.safetensors")],
            "TEST": [str(data / "test.safetensors")],
        },
        "DATA_LOADING": {
            "NUM_WORKERS": 0,
            "SHARDED_LOADING": True,
            "NUM_SAMPLES_TRAIN": 6,
            "NUM_SAMPLES_TEST": 2,
            "TRAIN_BATCH_SIZE": 2,
            "TEST_BATCH_SIZE": 2,
        },
        "SPECTRAL_GRID": {"WAVELENGTH_MIN": 350, "WAVELENGTH_MAX": 650, "WAVELENGTH_STEPS": 50},
        "OPEN_LAYER_DFM_JOINT": {
            "MAX_LAYERS": 5,
            "QUERY": {"ENABLED": True, "POINTS": 8, "TRAIN_MODE": "mixed", "MAX_BAND_NM": 250},
            "MATERIAL_BANK": {"MAX_CANDIDATES": 4},
            "MODEL": {"D_MODEL": 16, "N_HEADS": 4, "N_BLOCKS": 1, "QUERY_ENCODER_BLOCKS": 1},
            "TRAIN": {"EPOCHS": 1, "AMP": False, "CHECKPOINT_EVERY_STEPS": 2, "EMA": {"UPDATE_EVERY_STEPS": 1}},
        },
    }
    train = config["OPEN_LAYER_DFM_JOINT"]["TRAIN"]
    if high_noise_fraction:
        train["FLOW_TIME_SAMPLING"] = {"HIGH_NOISE_FRACTION": high_noise_fraction, "HIGH_NOISE_MAX_S": 0.35}
        train["KEEP_EPOCH_CHECKPOINTS"] = [1]
    path = tmp_path / "joint.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    output = data / "arm"
    args = [
        "train", "--config", str(path), "--device", "cpu", "--output-dir", str(output),
        "--visible-material-replace-peak", str(replacement_peak),
    ]
    if visible_weight != 1.0:
        args += ["--correct-visible-material-weight", str(visible_weight)]
    monkeypatch.setattr(sys, "argv", args)
    trainer.main()
    progress = tmp_path / "progress.pt"
    shutil.copyfile(output / "open-layer-dfm-joint-progress.pt", progress)
    completed = torch.load(output / "open-layer-dfm-joint-last.pt", weights_only=False)
    checkpoint = torch.load(progress, weights_only=False)
    assert checkpoint["extra"]["resume_cursor"]["next_batch"] == 2
    assert checkpoint["extra"]["ema"]["updates"] == 2
    assert checkpoint["extra"]["visible_material_replace_peak"] == replacement_peak
    assert checkpoint["extra"]["correct_visible_material_weight"] == visible_weight
    assert checkpoint["extra"]["joint_flow_time_mix"] == (high_noise_fraction, 0.35)

    if high_noise_fraction:
        archived = output / "open-layer-dfm-joint-epoch-0001.pt"
        archived_bytes = archived.read_bytes()
        assert archived_bytes == (output / "open-layer-dfm-joint-last.pt").read_bytes()
        train["FLOW_TIME_SAMPLING"]["HIGH_NOISE_FRACTION"] = 0.0
        path.write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [*args, "--resume", str(progress)])
        with pytest.raises(ValueError, match="same joint flow-time mixture"):
            trainer.main()
        train["FLOW_TIME_SAMPLING"]["HIGH_NOISE_FRACTION"] = high_noise_fraction
        path.write_text(json.dumps(config), encoding="utf-8")
    elif visible_weight == 1.0:
        del checkpoint["extra"]["correct_visible_material_weight"]
        del checkpoint["extra"]["joint_flow_time_mix"]
        torch.save(checkpoint, progress)
    else:
        monkeypatch.setattr(sys, "argv", [*args, "--resume", str(progress), "--correct-visible-material-weight", "0.10"])
        with pytest.raises(ValueError, match="same correctly visible material weight"):
            trainer.main()
    monkeypatch.setattr(sys, "argv", [*args, "--resume", str(progress)])
    trainer.main()
    resumed = torch.load(output / "open-layer-dfm-joint-last.pt", weights_only=False)
    assert resumed["extra"]["global_optimizer_steps"] == completed["extra"]["global_optimizer_steps"] == 3
    assert resumed["extra"]["history"][0]["train"]["samples"] == 6
    assert ("wrong_visible_accuracy" in resumed["extra"]["history"][0]["train"]) == bool(replacement_peak)
    assert "wrong_visible_accuracy" not in resumed["extra"]["history"][0]["val"]
    if high_noise_fraction:
        assert archived.read_bytes() == archived_bytes and not archived.samefile(output / "open-layer-dfm-joint-last.pt")
    for key, tensor in completed["model_state"].items():
        torch.testing.assert_close(resumed["model_state"][key], tensor, rtol=0, atol=0)
        torch.testing.assert_close(
            resumed["extra"]["ema"]["shadow"][key],
            completed["extra"]["ema"]["shadow"][key],
            rtol=0,
            atol=0,
        )
