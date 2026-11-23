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


def test_native_layers_and_empty_stacks(example):
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
    loss = trainer.compute_joint_loss(model, model, empty, {})
    assert loss["material_loss"].item() == loss["thickness_loss"].item() == 0.0
    torch.testing.assert_close(loss["loss"], loss["count_loss"])
    loss["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


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


def test_mid_epoch_checkpoint_resume(tmp_path, monkeypatch):
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
    path = tmp_path / "joint.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--device", "cpu"])
    trainer.main()
    output = data / "output"
    progress = tmp_path / "progress.pt"
    shutil.copyfile(output / "open-layer-dfm-joint-progress.pt", progress)
    completed = torch.load(output / "open-layer-dfm-joint-last.pt", weights_only=False)
    checkpoint = torch.load(progress, weights_only=False)
    assert checkpoint["extra"]["resume_cursor"]["next_batch"] == 2
    assert checkpoint["extra"]["ema"]["updates"] == 2

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume", str(progress)])
    trainer.main()
    resumed = torch.load(output / "open-layer-dfm-joint-last.pt", weights_only=False)
    assert resumed["extra"]["global_optimizer_steps"] == completed["extra"]["global_optimizer_steps"] == 3
    assert resumed["extra"]["history"][0]["train"]["samples"] == 6
    for key, tensor in completed["model_state"].items():
        torch.testing.assert_close(resumed["model_state"][key], tensor, rtol=0, atol=0)
        torch.testing.assert_close(
            resumed["extra"]["ema"]["shadow"][key],
            completed["extra"]["ema"]["shadow"][key],
            rtol=0,
            atol=0,
        )
