"""Recovery and mixed-precision checks for open-vocabulary DFM training."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch

from optollama.data.dataset import ShardedSpectraDataset
from optollama.model.open_vocab_depth_field import OpenVocabularyDepthFieldConfig
from optollama.model.open_vocab_depth_field_dfm import DiscreteFlowConfig, OpenVocabularyDepthFieldDFM
from scripts.train_open_vocab_depth_field import (
    LocalGradientDiagnostic,
    capture_rank_state,
    diagnostic_allreduce,
    run_epoch,
)


def test_sharded_resume_seeks_past_complete_shards() -> None:
    """Batch seeking preserves the epoch's per-shard shuffled order."""
    paths = [f"shard_{shard}.safetensors" for shard in range(3)]

    def shape(_path, key):
        return (9, 3, 4) if key == "spectra" else (9, 1)

    def load(path, device):
        shard = paths.index(path)
        return {
            "spectra": torch.full((9, 3, 4), float(shard)),
            "thin_films": torch.arange(shard * 9, (shard + 1) * 9).reshape(9, 1),
        }

    with mock.patch("optollama.data.dataset.collect_safetensor_paths", return_value=paths), mock.patch(
        "optollama.data.dataset.safetensor_shape", side_effect=shape
    ), mock.patch("optollama.data.dataset.safetensors.torch.load_file", side_effect=load) as loader:
        dataset = ShardedSpectraDataset(paths, split="train", rank=0, world_size=1, seed=7)
        dataset.set_epoch(2)
        complete = [index for _, _, index in dataset]
        dataset.set_resume_batch(4, 3)
        resumed = [index for _, _, index in dataset]
        assert resumed == complete[12:]
        assert loader.call_count < 6
        dataset.set_epoch(3)
        assert len(list(dataset)) == 27


def test_fp32_attention_under_bf16_autocast() -> None:
    """Attention's projection inputs remain FP32 while the model uses BF16 AMP."""
    config = OpenVocabularyDepthFieldConfig(
        spectrum_shape=(3, 8),
        depth_bins=8,
        max_candidates=3,
        d_model=16,
        n_blocks=1,
        n_heads=2,
        hybrid_dilations=(1,),
        spectrum_patch_size=2,
        spectrum_patch_stride=2,
        spectrum_encoder_blocks=1,
        spectrum_encoder_heads=2,
        fp32_attention=True,
    )
    model = OpenVocabularyDepthFieldDFM(config, DiscreteFlowConfig())
    recorded = []

    def record_dtype(_module, args):
        recorded.append(args[0].dtype)

    handles = [
        module.register_forward_pre_hook(record_dtype)
        for module in model.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
    ]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(
                torch.rand(1, 3, 8),
                torch.full((1, 8), model.mask_id),
                torch.tensor([0.5]),
                wavelengths_nm=torch.linspace(300, 700, 8).unsqueeze(0),
                candidate_nk=torch.rand(1, 3, 8, 2),
                candidate_mask=torch.tensor([[True, True, False]]),
                incidence_angle_deg=torch.zeros(1),
                polarization_id=torch.zeros(1, dtype=torch.long),
            )
            loss = logits.float().square().mean()
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert recorded and all(dtype == torch.float32 for dtype in recorded)
    assert torch.isfinite(model.pointer_query.weight.grad).all()


def test_bf16_autocast_without_fp32_attention() -> None:
    """The configurable fast attention path supports a finite BF16 step."""
    config = OpenVocabularyDepthFieldConfig(
        spectrum_shape=(3, 8),
        depth_bins=8,
        max_candidates=3,
        d_model=16,
        n_blocks=1,
        n_heads=2,
        hybrid_dilations=(1,),
        spectrum_patch_size=2,
        spectrum_patch_stride=2,
        spectrum_encoder_blocks=1,
        spectrum_encoder_heads=2,
        fp32_attention=False,
    )
    model = OpenVocabularyDepthFieldDFM(config, DiscreteFlowConfig())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(
            torch.rand(1, 3, 8),
            torch.full((1, 8), model.mask_id),
            torch.tensor([0.5]),
            wavelengths_nm=torch.linspace(300, 700, 8).unsqueeze(0),
            candidate_nk=torch.rand(1, 3, 8, 2),
            candidate_mask=torch.tensor([[True, True, False]]),
            incidence_angle_deg=torch.zeros(1),
            polarization_id=torch.zeros(1, dtype=torch.long),
        )
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.num_materials).float(), torch.zeros(8, dtype=torch.long)
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.pointer_query.weight.grad).all()


def test_local_gradient_diagnostic_precedes_reduction() -> None:
    """The hook names a rank-local bad parameter without modifying mean reduction."""
    model = torch.nn.Linear(2, 2)
    diagnostic = LocalGradientDiagnostic(model)
    good = torch.ones_like(model.weight)
    bad = torch.full_like(model.bias, float("nan"))
    buffer = torch.cat((good.flatten(), bad.flatten()))
    bucket = SimpleNamespace(buffer=lambda: buffer, parameters=lambda: [model.weight, model.bias], gradients=lambda: [good, bad])
    future = torch.futures.Future()
    future.set_result([buffer])
    with mock.patch("torch.distributed.get_world_size", return_value=1), mock.patch(
        "torch.distributed.all_reduce", return_value=SimpleNamespace(get_future=lambda: future)
    ):
        result = diagnostic_allreduce(diagnostic, bucket).wait()
    assert diagnostic.bad_parameters == ["bias"]
    torch.testing.assert_close(result[:4], good.flatten())


def test_run_epoch_checkpoint_state_resumes_metrics_and_weights() -> None:
    """A partial epoch and its cursor reproduce uninterrupted training."""
    def make_components():
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return model, optimizer

    loader = [
        {"sample_mask": torch.ones(1, dtype=torch.bool), "clean_fields": torch.zeros((1, 1), dtype=torch.long)}
        for _ in range(4)
    ]

    def fake_loss(model, _batch, _cfg, _corruption):
        loss = model.weight.square().mean() * torch.rand(())
        metric = loss.detach()
        return {
            "loss": loss,
            "accuracy": metric,
            "corrupted_accuracy": metric,
            "noise_probability": metric,
            "corrupted_fraction": metric,
            "condition_dropped": metric,
        }

    def train(model, optimizer, batches, **kwargs):
        return run_epoch(
            model,
            batches,
            torch.device("cpu"),
            {},
            mock.Mock(),
            optimizer=optimizer,
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            amp_dtype=None,
            epoch=0,
            epochs=1,
            max_steps=kwargs.pop("max_steps", None),
            base_lr=0.1,
            lr_schedule=None,
            global_samples_seen=0,
            log_every_steps=10,
            **kwargs,
        )

    uninterrupted, uninterrupted_optimizer = make_components()
    with mock.patch("scripts.train_open_vocab_depth_field.compute_loss", side_effect=fake_loss):
        torch.manual_seed(123)
        full = train(uninterrupted, uninterrupted_optimizer, loader)
        first, first_optimizer = make_components()
        saved = {}
        torch.manual_seed(123)

        def checkpoint(_step, _samples, _next_batch, state):
            saved.update(state)

        train(
            first,
            first_optimizer,
            loader,
            max_steps=2,
            checkpoint_every_steps=2,
            on_checkpoint_step=checkpoint,
        )
        saved["rng"] = capture_rank_state(SimpleNamespace(calls=2), SimpleNamespace(calls=0))
        resumed, resumed_optimizer = make_components()
        resumed.load_state_dict(first.state_dict())
        resumed_optimizer.load_state_dict(first_optimizer.state_dict())
        partial = train(resumed, resumed_optimizer, loader[2:], resume_state=saved)
    torch.testing.assert_close(resumed.weight, uninterrupted.weight)
    assert partial["optimizer_steps"] == full["optimizer_steps"] == 4
    assert partial["loss"] == full["loss"]
