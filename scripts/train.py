# train.py
"""
Distributed Trainer for Optical-GPT.

All state-handling (model, optimiser, scheduler, gradient-scaler,
checkpointing, …) is encapsulated inside `OpticalGPTTrainer`.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Dict, List, Mapping, Tuple
import functools
import socket
import tempfile
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from data import (
    SinglePTDataset,
    build_dataloaders,
    create_masks,
    pad_collate_fn,
)
from model import OptoLlama
from utils import seed_everything

__all__ = ["OpticalGPTTrainer"]


class OpticalGPTTrainer:
    """
    Wraps the complete training loop (one `.train()` call → multi-epoch run).
    """

    # --------------------------------------------------------------------- #
    # Init                                                                  #
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        *,
        cfg,
        num_classes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_length: int,
        input_dim: int,
        dropout: float,
        epsilon: float,
        pad_idx: int,
        sos_idx: int,
        eos_idx: int,
        pt_file: str,
        token_string_map: Mapping[int, str],
        material_to_id: Mapping[str, int],
    ) -> None:
        self.cfg = cfg 
        self._rank, self._world_size, self._local_rank = self._setup_dist()
        self.device = torch.device("cuda", self._local_rank)

        # Core hyper-parameters
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.pad_idx, self.sos_idx, self.eos_idx = pad_idx, sos_idx, eos_idx
        self.epsilon = epsilon

        # Seed
        if self._rank == 0:
            Path(self.cfg.PATH_RUN).mkdir(parents=True, exist_ok=True)
        dist.barrier(device_ids=[self._local_rank])
        seed_everything(42)

        # ---------------------------------------------------------------- #
        # Model                                                            #
        # ---------------------------------------------------------------- #
        self.model = OptoLlama(
            num_classes=num_classes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_length=max_seq_length,
            input_dim=input_dim,
            dropout=dropout,
        ).to(self.device)
        self.model = DDP(
            self.model,
            device_ids=[self._local_rank],
            output_device=self._local_rank,
            broadcast_buffers=False,
        )

        # Optimiser, scheduler, loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=400
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.scaler = GradScaler("cuda")

        # ---------------------------------------------------------------- #
        # Data                                                             #
        # ---------------------------------------------------------------- #
        dataset = SinglePTDataset(pt_file)
        collate = functools.partial(
            pad_collate_fn,
            max_seq_length=max_seq_length,
            pad_idx=pad_idx,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            token_string_map=token_string_map,
            material_to_id=material_to_id,
        )
        self.train_dl, self.val_dl = build_dataloaders(
            dataset,
            batch_size=self.cfg.BATCH_SIZE,
            world_size=self._world_size,
            rank=self._rank,
            collate_fn=collate,
        )
        self.train_len, self.val_len = len(self.train_dl), len(self.val_dl)

        # Resume?
        if self.cfg.RESUME_EPOCH > 0:
            self._load_checkpoint(self.cfg.RESUME_EPOCH)

    # --------------------------------------------------------------------- #
    # Public interface                                                      #
    # --------------------------------------------------------------------- #

    def train(self) -> None:
        """
        Full training run across `self.cfg.EPOCHS`.
        """
        for epoch in range(max(0, self.cfg.RESUME_EPOCH), self.cfg.EPOCHS):
            tr_loss, tr_ce, tr_mse = self._train_epoch(epoch)
            v_loss, v_ce, v_mse, v_acc = self._validate_epoch()

            if self._rank == 0:
                print(
                    f"📈 Epoch {epoch+1:>3}/{self.cfg.EPOCHS} "
                    f"train={tr_loss:.5f} (ce={tr_ce:.5f}|mse={tr_mse:.5f})   "
                    f"val={v_loss:.5f} (ce={v_ce:.5f}|mse={v_mse:.5f})  "
                    f"acc={v_acc:.3%}"
                )
            self._save_checkpoint(
                epoch, tr_loss, tr_ce, tr_mse, v_loss, v_ce, v_mse, v_acc
            )
            self.scheduler.step()
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _setup_dist() -> tuple[int, int, int]:
        """
        Single-GPU launch on Windows / Spyder **without** fixed ports.
        Re-uses an existing process-group if one is already initialised.
        """
        # 1) Re-use if already initialised (second run in same kernel)
        if dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            local = int(os.environ.get("LOCAL_RANK", "0"))
            return rank, world, local
    
        # 2) Multi-GPU via launch utility → environment vars already set
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            dist.init_process_group(backend="nccl", init_method="env://",
                                    timeout=datetime.timedelta(hours=1))
        else:
            # 3) Single GPU / CPU fallback — choose a **free**, random port
            free_port = random.randint(30000, 40000)
            os.environ.update({
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(free_port),
            })
            # Use a file-based rendez-vous → no socket at all
            init_file = tempfile.NamedTemporaryFile(delete=False).name
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{init_file}",
                rank=0,
                world_size=1,
            )
    
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local if torch.cuda.is_available() else 0)
        return rank, world, local

    # --------------------------- 1) Train --------------------------------- #

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        sum_loss = sum_ce = sum_mse = 0.0

        for step, batch in enumerate(self.train_dl, start=1):
            self.optimizer.zero_grad(set_to_none=True)

            spec, mat_seq, thick_seq = (x.to(self.device) for x in batch)
            bsz, t = mat_seq.shape  # noqa: N806

            # Containers for per-time-step predictions
            mat_preds = torch.zeros(bsz, t - 1, self.num_classes, device=self.device)
            thick_preds = torch.zeros(bsz, t - 1, device=self.device)

            with autocast("cuda"):
                input_seq = mat_seq[:, :1]  # start token
                for t_idx in range(t - 1):
                    attn_mask, pad_mask = create_masks(
                        input_seq, self.pad_idx, n_heads=self.model.module.n_heads
                    )
                    attn_mask, pad_mask = attn_mask.to(self.device), pad_mask.to(
                        self.device
                    )

                    mat_logit, thick_logit = self.model(
                        spec, input_seq, attn_mask=attn_mask, key_padding_mask=pad_mask
                    )
                    mat_preds[:, t_idx] = mat_logit[:, -1]
                    thick_preds[:, t_idx] = thick_logit[:, -1]

                    if t_idx + 1 < t:  # teacher forcing
                        input_seq = torch.cat(
                            (input_seq, mat_seq[:, t_idx + 1 : t_idx + 2]), dim=1
                        )

                #  ------ loss
                loss_ce = self._cross_entropy_loss(
                    mat_preds.reshape(-1, self.num_classes),  # <- reshape
                    mat_seq[:, 1:]
                )
                loss_mse = self._thickness_loss(
                    thick_preds.reshape(-1),                  # <- reshape
                    thick_seq[:, 1:]
                )
                loss_mse = self._thickness_loss(
                    thick_preds,                      # [B, T-1]
                    thick_seq[:, 1:]                  # [B, T-1]
                )
                loss = loss_ce + loss_mse

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            sum_loss += loss.item()
            sum_ce += loss_ce.item()
            sum_mse += loss_mse.item()

            if step % 10 == 0 and self._rank == 0:
                print(
                    f"  • epoch {epoch+1:>3} | "
                    f"step {step:>4}/{self.train_len:<4} | "
                    f"loss {sum_loss/step:.5f}"
                )

        n = self.train_len
        return sum_loss / n, sum_ce / n, sum_mse / n

    # ------------------------- 2) Validation ------------------------------ #

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, float, float, float]:
        self.model.eval()
        sum_loss = sum_ce = sum_mse = 0.0
        correct, total = 0, 0

        for spec, mat_seq, thick_seq in self.val_dl:
            spec, mat_seq, thick_seq = (x.to(self.device) for x in (spec, mat_seq, thick_seq))
            bsz = spec.size(0)

            # greedy autoregressive generation w/ teacher forcing for loss
            input_seq = mat_seq[:, :1]
            finished = torch.zeros(bsz, dtype=torch.bool, device=self.device)
            all_preds: List[torch.Tensor] = []

            with autocast("cuda"):
                step_losses: List[Tuple[torch.Tensor, torch.Tensor]] = []

                for _ in range(self.max_seq_length - 1):
                    attn_mask, pad_mask = create_masks(
                        input_seq, self.pad_idx, self.model.module.n_heads
                    )
                    attn_mask, pad_mask = (
                        attn_mask.to(self.device),
                        pad_mask.to(self.device),
                    )

                    mat_logit, thick_logit = self.model(
                        spec, input_seq, attn_mask=attn_mask, key_padding_mask=pad_mask
                    )

                    pred_token = mat_logit[:, -1].argmax(dim=-1)  # [B]
                    all_preds.append(pred_token)

                    # teacher-forcing losses
                    gold_token = mat_seq[:, input_seq.size(1)]
                    gold_thick = thick_seq[:, input_seq.size(1)]

                    step_losses.append(
                        (
                            self._cross_entropy_loss(
                                mat_logit[:, -1], gold_token.unsqueeze(1)
                            ),
                            self._thickness_loss(thick_logit[:, -1], gold_thick),
                        )
                    )

                    # stopping
                    finished |= pred_token.eq(self.eos_idx)
                    if finished.all():
                        break

                    # next input
                    input_seq = torch.cat((input_seq, gold_token.unsqueeze(1)), dim=1)

                # aggregate
                ce_stack, mse_stack = zip(*step_losses) if step_losses else ((), ())
                loss_ce = torch.stack(list(ce_stack)).mean() if ce_stack else torch.tensor(0.0)
                loss_mse = torch.stack(list(mse_stack)).mean() if mse_stack else torch.tensor(0.0)
                loss = loss_ce + loss_mse

            sum_loss += loss.item()
            sum_ce += loss_ce.item()
            sum_mse += loss_mse.item()

            # preds = torch.stack(all_preds, dim=1)  # [B, T-1]
            preds = torch.stack(all_preds, dim=1).contiguous()  # keep it contiguous
            gold = mat_seq[:, 1 : 1 + preds.size(1)]
            mask = gold.ne(self.pad_idx)
            correct += preds[mask].eq(gold[mask]).sum().item()
            total += mask.sum().item()

        n = self.val_len
        return (
            sum_loss / n if n else 0.0,
            sum_ce / n if n else 0.0,
            sum_mse / n if n else 0.0,
            correct / total if total else 0.0,
        )

    # ---------------------- 3) Loss helpers -------------------------------- #

    def _cross_entropy_loss(self, logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        gold   = gold.reshape(-1)
        logits = logits.reshape(-1, self.num_classes)
        gold_1hot = F.one_hot(gold, self.num_classes).float()
        smoothed = gold_1hot * (1 - self.epsilon) + self.epsilon / self.num_classes
        mask = gold.ne(self.pad_idx)
        return self.criterion(F.log_softmax(logits[mask], -1), smoothed[mask])

    def _thickness_loss(self, pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        # Flatten so shapes always match, no matter if we pass [B] or [B,T]
        pred = pred.reshape(-1)
        gold = gold.reshape(-1)
        return self.mse_loss(pred / 50.0, gold / 50.0)/4

    # ---------------------- 4) Checkpointing ------------------------------- #

    def _save_checkpoint(
        self,
        epoch: int,
        tr_loss: float,
        tr_ce: float,
        tr_mse: float,
        v_loss: float,
        v_ce: float,
        v_mse: float,
        v_acc: float,
    ) -> None:
        if self._rank != 0:
            return
        torch.save(self.model.state_dict(), f"{self.cfg.PATH_RUN}/model_epoch_{epoch}.pth")
        torch.save(
            self.optimizer.state_dict(), f"{self.cfg.PATH_RUN}/optimizer_epoch_{epoch}.pth"
        )

        # append metrics
        import pandas as pd

        df = pd.DataFrame(
            {
                "epoch": [epoch],
                "train_loss": [tr_loss],
                "train_loss_ce": [tr_ce],
                "train_loss_mse": [tr_mse],
                "val_loss": [v_loss],
                "val_loss_ce": [v_ce],
                "val_loss_mse": [v_mse],
                "val_accuracy": [v_acc],
            }
        )
        csv_path = Path(self.cfg.PATH_RUN) / "training_metrics.csv"
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    def _load_checkpoint(self, epoch_to_resume: int) -> None:
        mpath = Path(self.cfg.PATH_RUN) / f"model_epoch_{epoch_to_resume}.pth"
        opath = Path(self.cfg.PATH_RUN) / f"optimizer_epoch_{epoch_to_resume}.pth"
        map_loc = {"cuda:0": f"cuda:{self._local_rank}"}
        self.model.load_state_dict(torch.load(mpath, map_location=map_loc))
        self.optimizer.load_state_dict(torch.load(opath, map_location=map_loc))
        if self._rank == 0:
            print(f"▶️  Resumed from epoch {epoch_to_resume}")
