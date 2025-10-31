#!/usr/bin/env python
import argparse
import os, sys

import numpy as np
import torch
import tqdm

import cli as cli
from utils_runner import setup_run, _is_ddp
from utils_data import make_loader, SpectraDataset
from utils_model import build_model
from utils_eval import build_tmm_context, token_accuracy, validate_model


# ------------------------------- training loop -------------------------------
def train_loop(arguments: argparse.Namespace, device: str, rank: int, world_size: int, slurm_localid: int) -> None:
    # --- token maps (keep your old behavior) ---
    from utils import init_tokenmaps
    tokens, token_to_idx, idx_to_token, \
        EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, \
        eos_idx, pad_idx, msk_idx = init_tokenmaps(cfg.PATH_DATA)

    # utils_data.make_loader uses cfg.BATCH_SIZE; keep your separate TRAIN/VALID batch sizes by overriding before each call
    # (This mutates the cfg module in-place; simple and effective.)
    setattr(cfg, "BATCH_SIZE", getattr(cfg, "TRAIN_BATCH", getattr(cfg, "BATCH_SIZE", 256)))
    train_subset_n = getattr(cfg, "NUM_SAMPLES_TRAIN", 0) or None
    train_ds, train_loader, train_sampler = make_loader(
        cfg,
        split="train",
        subset_n=train_subset_n,
        ddp=_is_ddp(),
    )

    setattr(cfg, "BATCH_SIZE", getattr(cfg, "VALID_BATCH", getattr(cfg, "BATCH_SIZE", 64)))
    valid_subset_n = getattr(cfg, "NUM_SAMPLES_VALID", 0) or None
    valid_ds, valid_loader, valid_sampler = make_loader(
        cfg,
        split="valid",
        subset_n=valid_subset_n,
        ddp=_is_ddp(),
    )

    # --- physics / TMM (centralized) ---
    try:
        tmm_ctx = build_tmm_context(cfg=cfg, idx_to_token=idx_to_token, device=device)
    except:
        tmm_ctx = None

    # --- model ---
    vocab_size = len(idx_to_token)
    example_spectrum = train_ds.spectra[0] if isinstance(train_ds, SpectraDataset) else train_ds.dataset.spectra[0]
    model = build_model(
        model_type=getattr(cfg, "ARCH", getattr(cfg, "OL_MODEL", "dit")),
        sample_spectrum=example_spectrum,            # [W,3] example
        vocab_size=vocab_size,
        max_stack_depth=train_ds.maximum_depth if isinstance(train_ds, SpectraDataset) else train_ds.dataset.maximum_depth,
        d_model=cfg.D_MODEL,
        n_blocks=cfg.N_BLOCKS,
        n_heads=cfg.N_HEADS,
        timesteps=cfg.STEPS,
        dropout=cfg.DROPOUT,
        idx_to_token=idx_to_token,
        mask_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
    ).to(device)

    # --- DDP wrapper (only when actually in DDP) ---
    is_ddp = _is_ddp()
    if is_ddp and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[slurm_localid],
            output_device=slurm_localid,
            find_unused_parameters=False,
        )

    # --- optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- metric buffers / resume bookkeeping ---
    E = cfg.EPOCHS
    train_losses     = torch.zeros(E)
    train_losses_CE  = torch.zeros(E)
    train_acc        = torch.zeros(E)
    valid_acc          = torch.zeros(E)
    valid_MAE          = torch.ones(E) * np.inf
    start_epoch = 0

    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint.pt"
    best_valid_acc = 0.0
    best_valid_mae = np.inf

    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start_epoch = int(state.get("epoch", -1)) + 1
        # The file stores {'model_state', 'optimizer_state', ...}; load back
        if "model_state" in state:
            model.load_state_dict(state["model_state"], strict=True)
        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if "train_losses" in state: train_losses = state["train_losses"]
        if "train_acc" in state:    train_acc    = state["train_acc"]
        if "valid_acc" in state:      valid_acc      = state["valid_acc"]
        if "valid_MAE" in state:      valid_MAE      = state["valid_MAE"]
        # robust bests
        acc_arr = valid_acc.detach().cpu().numpy()
        if np.any(np.isfinite(acc_arr)):
            best_valid_acc = float(np.nanmax(acc_arr[np.isfinite(acc_arr)]))
        mae_arr = valid_MAE.detach().cpu().numpy()
        if np.any(np.isfinite(mae_arr)):
            best_valid_mae = float(np.nanmin(mae_arr[np.isfinite(mae_arr)]))

    # ------------------------------ epochs ------------------------------
    for epoch in range(start_epoch, cfg.EPOCHS):
        # DDP epoch seeds
        if is_ddp and getattr(train_loader, "sampler", None) is not None:
            train_loader.sampler.set_epoch(epoch)
        if is_ddp and getattr(valid_loader, "sampler", None) is not None:
            valid_loader.sampler.set_epoch(epoch)

        # ------------------------------ train ------------------------------
        if not args.notrain:
            model.train()
            if rank == 0:
                pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.EPOCHS} train", leave=True)

            for i, batch in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)

                spectra, stacks = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                logits = model(spectra, stacks)  # teacher forcing path → [B,S,V]

                # --- CE ---
                if cfg.CE:
                    log_probs = torch.nn.functional.log_softmax(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
                    loss_CE = torch.nn.NLLLoss(ignore_index=pad_idx)(
                        log_probs.view(-1, vocab_size), stacks.view(-1)
                    )
                else:
                    loss_CE = torch.tensor(0.0, device=device)

                # --- total loss ---
                loss = 0.0
                if cfg.CE:  loss = loss + loss_CE

                # Guard for NaNs, DDP-synchronized
                ce_bad  = torch.tensor([float(cfg.CE and (not torch.isfinite(loss_CE)))],  device=device)
                if is_ddp:
                    torch.distributed.all_reduce(ce_bad,  op=torch.distributed.ReduceOp.MAX)
                if ce_bad.item() > 0.0:
                    if rank == 0:
                        what = "CE" if ce_bad.item() > 0 else ""
                        print(f"⚠️ non-finite loss ({what}) at epoch {epoch}, step {i} — skipping batch")
                    continue

                loss.backward()
                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        print('bad_grad, scipping')
                        break
                if bad_grad:
                    optimizer.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()

                # --- logging (DDP-avg for losses) ---
                with torch.no_grad():
                    log_loss = loss.detach().clone()
                    log_ce   = loss_CE.detach().clone()
                    if is_ddp:
                        for t in (log_loss, log_ce):
                            torch.distributed.all_reduce(t)
                        log_loss /= world_size; log_ce /= world_size;

                    train_losses[epoch]     += log_loss.item()
                    train_losses_CE[epoch]  += log_ce.item()

                    acc, _ = token_accuracy(stacks, logits.argmax(dim=-1), eos_idx, pad_idx, msk_idx)
                    train_acc[epoch] += acc

                if rank == 0:
                    pbar.set_postfix(
                        loss=f"{train_losses[epoch] / (i + 1):.4f}",
                        loss_CE=f"{train_losses_CE[epoch] / (i + 1):.4f}",
                        acc=f"{train_acc[epoch] / (i + 1) * 100:.4f}%"
                    )
                    pbar.update()

            if rank == 0:
                pbar.close()

        # ------------------------------ validation ------------------------------
        mc_samples = int(getattr(cfg, "MC_SAMPLES", 1))
        mode = (args.validsim if args.validsim else "NOSIM").upper()
        
        val_out = validate_model(
            model,
            valid_loader,
            mode=mode,
            eos=eos_idx, pad=pad_idx, msk=msk_idx,
            device=device,
            idx_to_token=idx_to_token,
            tmm_ctx=tmm_ctx,
            mc_samples=mc_samples,
            rank=rank,
            world_size=world_size,
            gather=True,
        )
        
        # update trackers
        valid_acc[epoch] = float(val_out['mean_acc'])
        if mode == "TMM_FAST":
            valid_MAE[epoch] = float(val_out['mean_mae'])
        
        # save per-example results (rank 0 only)
        if rank == 0 and 'results' in val_out:
            os.makedirs(cfg.PATH_SAVED, exist_ok=True)
            out_name = f"results_{cfg.RUN_NAME}"
            from utils import save_JSONPICKLE
            save_JSONPICKLE(cfg.PATH_SAVED, val_out['results'], out_name)
            print(f"💾 [rank 0] Saved {len(val_out['results'])} samples → …/{out_name}.json")
            if mode == "TMM_FAST":
                print(f" min valid MAE: {float(torch.min(valid_MAE).item()) if torch.isfinite(valid_MAE).any() else valid_MAE[epoch]:.4f}")
                print(f"last valid MAE: {valid_MAE[epoch]:.4f}")
            else:
                print(f"Validation accuracy: {valid_acc[epoch]:.2f}%")
                if rank == 0:
                    print(f"⚠️ No validation at epoch {epoch}")
                    continue

        # ------------------------------ checkpointing ------------------------------
        if not args.notrain and rank == 0:
            os.makedirs(cfg.PATH_SAVED, exist_ok=True)
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "valid_MAE": valid_MAE,
            }

            if not args.validsim:
                # accuracy-based checkpointing
                cur_best = float(np.nanmax(valid_acc.numpy()))
                if cur_best >= best_valid_acc:
                    best_valid_acc = cur_best
                    print(f"Checkpoint saved (ACC): best={best_valid_acc:.6f} [epoch {epoch}]")
                    torch.save(state, checkpoint)
            else:
                # MAE-based checkpointing (lower is better); use small epsilon to force strict improvement
                new_mae = float(valid_MAE[epoch].item())
                if new_mae < (best_valid_mae - 1e-7):
                    print(f"Checkpoint saved (MAE): new={new_mae:.6f} < best={best_valid_mae:.6f} [epoch {epoch}]")
                    best_valid_mae = new_mae
                    torch.save(state, checkpoint)


# -------------------------------- entry point --------------------------------

def train(args: argparse.Namespace, cfg) -> None:
    # distributed + torch setup
    device, slurm_localid, rank, world_size = setup_run(cfg, make_dirs=False)

    # main loop
    train_loop(args, device, rank, world_size, slurm_localid)

    # graceful shutdown
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Make repeated runs in IDEs safe by cleaning up any stale DDP state
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass

    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "config_MD60.py"])                             #TODO rename to better name

    # Parse args and build final config (applies --ckpt/--mc-samples/--validsim and --set)
    args = cli.parse_arguments()
    cfg = cli.load_config_with_overrides(args)

    # Optional: support `--print-config` from here too (no run)
    if getattr(args, "print_config", True):
        for k in sorted([k for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))]):
            print(f"{k} = {getattr(cfg, k)!r}")

    try:
        train(args, cfg)
    finally:
        # Always clean up DDP
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

