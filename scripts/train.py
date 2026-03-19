#!/usr/bin/env python

import torch
import tqdm

import optollama
import optollama.cli
import optollama.runner

# ruff: noqa: N806


def train(cfg: dict) -> None:
    """
    Train and optionally validate OptoLlama (or OptoGPT).

    Args
    ----
    cfg:
        Configuration object (SimpleNamespace or similar) as returned by
        `cli.load_config_with_overrides`. Must contain at least the keys used
        below (paths, model hyperparameters, batch sizes, etc.).
    """
    device, slurm_localid, rank, world_size = optollama.runner.setup_run(cfg, make_dirs=True)
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = init_tokenmaps(cfg.PATH_DATA)

    train_ds, train_loader, train_sampler = make_loader(
        cfg,
        split="train",
        subset_n=cfg.NUM_SAMPLES_TRAIN,
        ddp=_is_ddp(),
    )

    valid_ds, valid_loader, valid_sampler = make_loader(
        cfg,
        split="valid",
        subset_n=cfg.NUM_SAMPLES_VALID,
        ddp=_is_ddp(),
    )

    # --- physics / TMM (centralized) ---
    # TMM context
    tmm_ctx = None
    if cfg.VALIDSIM == "TMM_FAST":
        tmm_ctx = build_tmm_context(cfg=cfg, idx_to_token=idx_to_token, device=device)

    # --- model ---
    vocab_size = len(idx_to_token)
    example_spectrum = train_ds.spectra[0] if isinstance(train_ds, SpectraDataset) else train_ds.dataset.spectra[0]
    model = build_model(
        model_type=getattr(cfg, "MODEL_KEY", "optollama"),
        sample_spectrum=example_spectrum,  # [W,3] example
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
        temperature=cfg.TEMPERATURE,
        top_k=cfg.TOP_K,
        top_p=cfg.TOP_P,
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
    train_losses = torch.zeros(cfg.EPOCHS)
    train_acc = torch.zeros(cfg.EPOCHS)
    valid_acc = torch.zeros(cfg.EPOCHS)
    valid_mae = torch.ones(cfg.EPOCHS) * torch.inf

    checkpoint = cfg.PATH_CKPT

    best_valid_acc = 0.0
    best_valid_mae = torch.inf
    start_epoch = 0

    if checkpoint and os.path.exists(checkpoint):
        start_epoch, blob = load_checkpoint(checkpoint, model, optimizer=optimizer, map_location="cpu", strict=True)
        start_epoch = start_epoch or 0
        # recover metric buffers if present
        train_losses = blob.get("train_losses", train_losses)
        train_acc = blob.get("train_acc", train_acc)
        valid_acc = blob.get("valid_acc", valid_acc)
        valid_mae = blob.get("valid_mae", valid_mae)

        # robust bests
        if torch.any(torch.isfinite(valid_acc)):
            best_valid_acc = float(torch.max(valid_acc[torch.isfinite(valid_acc)]))
        if torch.any(torch.isfinite(valid_mae)):
            best_valid_mae = float(torch.min(valid_mae[torch.isfinite(valid_mae)]))

    # ------------------------------ epochs ------------------------------
    for epoch in range(start_epoch, cfg.EPOCHS):
        # DDP epoch seeds
        if is_ddp and getattr(train_loader, "sampler", None) is not None:
            train_loader.sampler.set_epoch(epoch)
        if is_ddp and getattr(valid_loader, "sampler", None) is not None:
            valid_loader.sampler.set_epoch(epoch)

        # ------------------------------ train ------------------------------
        if not cfg.NOTRAIN:
            model.train()
            if rank == 0:
                pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{cfg.EPOCHS} train", leave=True)

            for i, batch in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)

                spectra, stacks = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                logits = model(spectra, stacks)  # teacher forcing path → [B,S,V]

                log_probs = torch.nn.functional.log_softmax(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
                loss_CE = torch.nn.NLLLoss(ignore_index=pad_idx)(log_probs.view(-1, vocab_size), stacks.view(-1))

                # Guard for NaNs, DDP-synchronized
                ce_bad = torch.tensor([float(not torch.isfinite(loss_CE))], device=device)
                if is_ddp:
                    torch.distributed.all_reduce(ce_bad, op=torch.distributed.ReduceOp.MAX)
                if ce_bad.item() > 0.0:
                    if rank == 0:
                        what = "CE" if ce_bad.item() > 0 else ""
                        print(f"non-finite loss ({what}) at epoch {epoch}, step {i} — skipping batch")
                    continue

                loss_CE.backward()
                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        print("bad_grad, scipping")
                        break
                if bad_grad:
                    optimizer.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()

                # --- logging (DDP-avg for losses) ---
                with torch.no_grad():
                    log_ce = loss_CE.detach().clone()
                    if is_ddp:
                        torch.distributed.all_reduce(log_ce)
                        log_ce /= world_size

                    train_losses[epoch] += log_ce.item()

                    acc, _ = token_accuracy(stacks, logits.argmax(dim=-1), eos_idx, pad_idx, msk_idx)
                    train_acc[epoch] += acc

                if rank == 0:
                    pbar.set_postfix(
                        loss_CE=f"{train_losses[epoch] / (i + 1):.4f}",
                        acc=f"{train_acc[epoch] / (i + 1) * 100:.4f}%",
                    )
                    pbar.update()

            if rank == 0:
                pbar.close()

        # ------------------------------ validation ------------------------------
        val_out = validate_model(
            model,
            valid_loader,
            mode=cfg.VALIDSIM,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            device=device,
            idx_to_token=idx_to_token,
            tmm_ctx=tmm_ctx,
            mc_samples=cfg.MC_SAMPLES,
            rank=rank,
            world_size=world_size,
            gather=True,
            track_step_mae=False,
        )

        # update trackers
        valid_acc[epoch] = float(val_out["mean_acc"])
        if cfg.VALIDSIM == "TMM_FAST":
            valid_mae[epoch] = float(val_out["mean_mae"])

        # save per-example results (rank 0 only)
        if rank == 0 and "results" in val_out:
            os.makedirs(cfg.PATH_SAVED, exist_ok=True)
            out_name = f"results_{cfg.RUN_NAME}"
            save_as_json(cfg.PATH_SAVED, val_out["results"], out_name)
            print(f"[rank 0] Saved {len(val_out['results'])} samples → …/{out_name}.json")
            if cfg.VALIDSIM == "TMM_FAST":
                print(
                    f" min valid MAE: {float(torch.min(valid_mae).item()) if torch.isfinite(valid_mae).any() else valid_mae[epoch]:.4f}"
                )
                print(f"last valid MAE: {valid_mae[epoch]:.4f}")
            else:
                print(f"Validation accuracy: {valid_acc[epoch]:.2f}%")
                if rank == 0:
                    print(f"No validation at epoch {epoch}")
                    continue

        # ------------------------------ checkpointing ------------------------------
        if not cfg.NOTRAIN and rank == 0:
            if cfg.VALIDSIM != "TMM_FAST":
                # accuracy-based checkpointing
                new_acc = float(valid_acc[epoch].item())
                if new_acc > (best_valid_acc + 1e-7):
                    print(f"Checkpoint saved (ACC): new={new_acc:.6f} > best={best_valid_acc:.6f} [epoch {epoch}]")
                    best_valid_acc = new_acc
                    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint_best.pt"
                else:
                    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint.pt"
            else:
                # MAE-based checkpointing
                new_mae = float(valid_mae[epoch].item())
                if new_mae < (best_valid_mae - 1e-7):
                    print(f"Checkpoint saved (MAE): new={new_mae:.6f} < best={best_valid_mae:.6f} [epoch {epoch}]")
                    best_valid_mae = new_mae
                    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint_best.pt"
                else:
                    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint.pt"

            os.makedirs(cfg.PATH_SAVED, exist_ok=True)
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_losses=train_losses,
                train_acc=train_acc,
                valid_acc=valid_acc,
                valid_mae=valid_mae,
                extra={
                    "sampling": {
                        "temperature": float(getattr(cfg, "TEMPERATURE", 0.0)),
                        "top_k": int(getattr(cfg, "TOP_K", 0)),
                        "top_p": float(getattr(cfg, "TOP_P", 0.0)),
                    }
                },
            )


if __name__ == "__main__":
    optollama.runner.stop_ddp() # clean up old ddp sesssion in interactive mode

    # parse args and build final config
    args = optollama.cli.parse_arguments()
    cfg = optollama.cli.load_config(args)

    try:
        train(cfg)
    finally:
       optollama.runner.stop_ddp()
