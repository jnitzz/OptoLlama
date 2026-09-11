from __future__ import annotations

from unittest import mock

import torch

from scripts.train_open_vocab_depth_field import ModelEma, run_epoch, unwrap_model


class CompiledLikeWrapper(torch.nn.Module):
    """Minimal wrapper exposing the attribute used by torch.compile."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = model

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._orig_mod(value)


def test_unwrap_model_and_interval_ema_decay() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEma(model, decay=0.9)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(CompiledLikeWrapper(model), decay=0.9**2)

    assert unwrap_model(CompiledLikeWrapper(model)) is model
    torch.testing.assert_close(ema.shadow["weight"], torch.tensor([[1.38]]))
    assert ema.updates == 1


def test_run_epoch_triggers_global_step_evaluation_and_throttles_ema() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loader = [
        {
            "sample_mask": torch.ones(2, dtype=torch.bool),
            "clean_fields": torch.zeros((2, 1), dtype=torch.long),
        }
        for _ in range(4)
    ]
    callbacks: list[tuple[int, int]] = []
    ema = ModelEma(model, decay=0.9)

    def fake_compute_loss(current_model, batch, _train_cfg, _corruption):
        loss = current_model.weight.square().mean()
        scalar = loss.detach().new_tensor(0.5)
        return {
            "loss": loss,
            "accuracy": scalar,
            "corrupted_accuracy": scalar,
            "noise_probability": scalar,
            "corrupted_fraction": scalar,
            "condition_dropped": scalar,
        }

    with mock.patch("scripts.train_open_vocab_depth_field.compute_loss", side_effect=fake_compute_loss):
        metrics = run_epoch(
            model,
            loader,  # type: ignore[arg-type]
            torch.device("cpu"),
            {},
            mock.Mock(),
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=None,
            epoch=0,
            epochs=1,
            max_steps=None,
            base_lr=0.01,
            lr_schedule=None,
            global_samples_seen=100,
            ema=ema,
            global_optimizer_steps=6,
            log_every_steps=50,
            ema_update_every_steps=2,
            eval_every_steps=2,
            on_evaluation_step=lambda step, samples, _running: callbacks.append((step, samples)),
        )

    assert callbacks == [(8, 104), (10, 108)]
    assert metrics["optimizer_steps"] == 4
    assert metrics["global_optimizer_steps"] == 10
    assert metrics["global_samples_seen"] == 108
    assert ema.updates == 2
