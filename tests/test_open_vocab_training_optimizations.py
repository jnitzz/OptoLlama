from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import torch

from scripts.train_open_vocab_depth_field import ModelEma, run_epoch, unwrap_model


class CompiledLikeWrapper(torch.nn.Module):
    """Minimal wrapper exposing the attribute used by torch.compile."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = model

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Delegate to the wrapped module."""
        return self._orig_mod(value)


def test_unwrap_model_and_interval_ema_decay() -> None:
    """Compiled wrappers unwrap and EMA intervals compound decay."""
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
    """Step callbacks and EMA updates follow successful optimizer steps."""
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


def test_run_epoch_skips_one_nonfinite_loss_and_keeps_training() -> None:
    """A bad forward loss does not poison the next optimizer step."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [
        {
            "sample_mask": torch.ones(1, dtype=torch.bool),
            "sample_indices": torch.tensor([index]),
            "clean_fields": torch.zeros((1, 1), dtype=torch.long),
        }
        for index in (17, 18)
    ]
    calls = 0

    def fake_compute_loss(current_model, _batch, _train_cfg, _corruption):
        nonlocal calls
        calls += 1
        loss = current_model.weight.square().sum()
        if calls == 1:
            loss = loss * float("nan")
        scalar = loss.detach().new_tensor(0.5)
        return {
            "loss": loss,
            "accuracy": scalar,
            "corrupted_accuracy": scalar,
            "noise_probability": scalar,
            "corrupted_fraction": scalar,
            "condition_dropped": scalar,
        }

    output = StringIO()
    with mock.patch("scripts.train_open_vocab_depth_field.compute_loss", side_effect=fake_compute_loss):
        with redirect_stdout(output):
            metrics = run_epoch(
                model,
                loader,  # type: ignore[arg-type]
                torch.device("cpu"),
                {"MAX_NONFINITE_STEPS_PER_EPOCH": 1, "MAX_CONSECUTIVE_NONFINITE_STEPS": 1},
                mock.Mock(),
                optimizer=optimizer,
                scaler=torch.amp.GradScaler("cpu", enabled=False),
                amp_dtype=None,
                epoch=0,
                epochs=1,
                max_steps=None,
                base_lr=0.1,
                lr_schedule=None,
                global_samples_seen=0,
            )

    assert "sample_indices=[17]" in output.getvalue()
    assert metrics["nonfinite_loss_skips"] == 1
    assert metrics["nonfinite_grad_skips"] == 0
    assert metrics["optimizer_steps"] == 1
    torch.testing.assert_close(model.weight, torch.tensor([[0.9]]))


def test_run_epoch_skips_one_nonfinite_gradient_and_keeps_training() -> None:
    """A bad gradient norm does not update weights or EMA."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    backward_calls = 0

    def inject_bad_gradient(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal backward_calls
        backward_calls += 1
        return gradient.new_full(gradient.shape, float("nan")) if backward_calls == 1 else gradient

    model.weight.register_hook(inject_bad_gradient)
    loader = [
        {
            "sample_mask": torch.ones(1, dtype=torch.bool),
            "sample_indices": torch.tensor([index]),
            "clean_fields": torch.zeros((1, 1), dtype=torch.long),
        }
        for index in (27, 28)
    ]

    def fake_compute_loss(current_model, _batch, _train_cfg, _corruption):
        loss = current_model.weight.square().sum()
        scalar = loss.detach().new_tensor(0.5)
        return {
            "loss": loss,
            "accuracy": scalar,
            "corrupted_accuracy": scalar,
            "noise_probability": scalar,
            "corrupted_fraction": scalar,
            "condition_dropped": scalar,
        }

    ema = ModelEma(model, decay=0.9)
    output = StringIO()
    with mock.patch("scripts.train_open_vocab_depth_field.compute_loss", side_effect=fake_compute_loss):
        with redirect_stdout(output):
            metrics = run_epoch(
                model,
                loader,  # type: ignore[arg-type]
                torch.device("cpu"),
                {"MAX_NONFINITE_STEPS_PER_EPOCH": 1, "MAX_CONSECUTIVE_NONFINITE_STEPS": 1},
                mock.Mock(),
                optimizer=optimizer,
                scaler=torch.amp.GradScaler("cpu", enabled=False),
                amp_dtype=None,
                epoch=0,
                epochs=1,
                max_steps=None,
                base_lr=0.1,
                lr_schedule=None,
                global_samples_seen=0,
                ema=ema,
            )

    assert "sample_indices=[27]" in output.getvalue()
    assert metrics["nonfinite_loss_skips"] == 0
    assert metrics["nonfinite_grad_skips"] == 1
    assert metrics["optimizer_steps"] == 1
    assert ema.updates == 1
    torch.testing.assert_close(model.weight, torch.tensor([[0.9]]))


def test_run_epoch_aborts_after_nonfinite_skip_limit() -> None:
    """Repeated bad losses exceed the guard's per-epoch budget."""
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [
        {"sample_mask": torch.ones(1, dtype=torch.bool), "clean_fields": torch.zeros((1, 1), dtype=torch.long)} for _ in range(2)
    ]

    def fake_compute_loss(current_model, _batch, _train_cfg, _corruption):
        loss = current_model.weight.sum() * float("nan")
        scalar = loss.detach().new_zeros(())
        return {
            "loss": loss,
            "accuracy": scalar,
            "corrupted_accuracy": scalar,
            "noise_probability": scalar,
            "corrupted_fraction": scalar,
            "condition_dropped": scalar,
        }

    with mock.patch("scripts.train_open_vocab_depth_field.compute_loss", side_effect=fake_compute_loss):
        with redirect_stdout(StringIO()):
            try:
                run_epoch(
                    model,
                    loader,  # type: ignore[arg-type]
                    torch.device("cpu"),
                    {"MAX_NONFINITE_STEPS_PER_EPOCH": 1},
                    mock.Mock(),
                    optimizer=optimizer,
                    scaler=torch.amp.GradScaler("cpu", enabled=False),
                    amp_dtype=None,
                    epoch=0,
                    epochs=1,
                    max_steps=None,
                    base_lr=0.1,
                    lr_schedule=None,
                    global_samples_seen=0,
                )
            except FloatingPointError as error:
                assert "skipped 2 steps this epoch" in str(error)
            else:
                raise AssertionError("Repeated non-finite losses should abort training.")
