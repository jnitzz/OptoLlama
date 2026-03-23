#!/usr/bin/env python

import os

import torch
import torch.utils.data
import tqdm

import optollama
import optollama.data
import optollama.evaluation
import optollama.model
import optollama.utils

# ruff: noqa: N806


def train(cfg: dict) -> None:
    """
    Train and optionally validate OptoLlama (or OptoGPT).

    Args
    ----
    cfg: dict
        Configuration object
    """
    # --- distributed computation setup ---
    device, local_rank, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=True)
    ddp = optollama.utils.is_ddp()
    
    # --- data loading and preprocessing ---
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])

    train_ds, train_loader, train_sampler = optollama.data.SpectraDataset.make_loader(
        cfg, 
        split="train", 
        subset_n=cfg["NUM_SAMPLES_TRAIN"], 
        ddp=ddp
    )
    test_ds, test_loader, test_sampler = optollama.data.SpectraDataset.make_loader(
        cfg, 
        split="test", 
        subset_n=cfg["NUM_SAMPLES_TEST"], 
        ddp=ddp
    )

    # --- TMM simulation ---
    tmm_ctx = optollama.evaluation.simulation.build_tmm_context(
        cfg=cfg, 
        idx_to_token=idx_to_token, 
        device=device
    ) if cfg["VALID_SIM"] == "TMM_FAST" else None

    # --- model ---
    vocab_size = len(idx_to_token)
    example_spectrum = train_ds.dataset.spectra[0] if isinstance(train_ds, torch.utils.data.Subset) else train_ds.spectra[0]
    
    model = optollama.model.build_model(
        model_type=cfg["MODEL"],
        sample_spectrum=example_spectrum,  # [W,3] example
        vocab_size=vocab_size,
        max_stack_depth=cfg["MAX_SEQ_LEN"],
        d_model=cfg["D_MODEL"],
        n_blocks=cfg["N_BLOCKS"],
        n_heads=cfg["N_HEADS"],
        timesteps=cfg.get("DIFFUSION_STEPS", None),
        dropout=cfg["DROPOUT"],
        idx_to_token=idx_to_token,
        mask_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
        temperature=cfg["TEMPERATURE"],
        top_k=cfg["TOP_K"],
        top_p=cfg["TOP_P"],
    ).to(device)

    # --- DDP wrapper ---
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    # --- optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- metric buffers / resume bookkeeping ---
    train_losses = torch.zeros(cfg["EPOCHS"])
    train_acc = torch.zeros(cfg["EPOCHS"])
    test_acc = torch.zeros(cfg["EPOCHS"])
    test_mae = torch.full((cfg["EPOCHS"],), torch.inf)

    checkpoint = cfg["CHECKPOINT_PATH"]

    best_test_acc = 0.0
    best_test_mae = torch.inf
    start_epoch = 0

    if checkpoint and os.path.exists(checkpoint):
        start_epoch, blob = load_checkpoint(checkpoint, model, optimizer=optimizer, map_location="cpu", strict=True)
        
        # recover metric buffers if present
        train_losses = blob.get("train_losses", train_losses)
        train_acc = blob.get("train_acc", train_acc)
        test_acc = blob.get("test_acc", test_acc)
        test_mae = blob.get("test_mae", test_mae)

    # ------------------------------ epochs ------------------------------
    epochs = cfg["EPOCHS"]
    for epoch in range(start_epoch, epochs):
        # DDP epoch seeds
        if ddp:
            train_loader.sampler.set_epoch(epoch)
            test_loader.sampler.set_epoch(epoch)

        # ------------------------------ train ------------------------------
        model.train()
        if rank == 0:
            pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs} train", leave=True)

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            spectra, stacks = batch[0].to(device), batch[1].to(device)
            logits = model(spectra, stacks)  # teacher forcing path → [B,S,V]

            log_probs = torch.nn.functional.log_softmax(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
            loss = torch.nn.NLLLoss(ignore_index=pad_idx)(log_probs.view(-1, vocab_size), stacks.view(-1))

            loss.backward()
            optimizer.step()

            # --- logging (DDP-avg for losses) ---
            with torch.no_grad():
                log_ce = loss.detach().clone()
                if ddp:
                    torch.distributed.all_reduce(log_ce)
                    log_ce /= world_size

                train_losses[epoch] += log_ce.item()

                acc, _ = optollama.evaluation.metrics.token_accuracy(stacks, logits.argmax(dim=-1), eos_idx, pad_idx, msk_idx)
                # train_acc[epoch] += acc

            if rank == 0:
                pbar.set_postfix(
                    loss_CE=f"{train_losses[epoch] / (i + 1):.4f}",
                    acc=f"{train_acc[epoch] / (i + 1) * 100:.4f}%",
                )
                pbar.update()

        if rank == 0:
            pbar.close()

        # ------------------------------ validation ------------------------------
        model.eval()
        test_output = optollama.evaluation.evaluate_model(
            model,
            test_loader,
            device=device,
            mode=cfg["VALID_SIM"],
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            idx_to_token=idx_to_token,
            tmm_ctx=tmm_ctx,
            mc_samples=cfg["MC_SAMPLES"],
            rank=rank,
            world_size=world_size,
            gather=True,
            track_step_mae=False,
        )

        # update trackers
        test_acc[epoch] = test_output["mean_acc"].item()
        if cfg["VALID_SIM"] == "TMM_FAST":
            test_mae[epoch] = test_output["mean_mae"].item()

        # save per-example results (rank 0 only)
        if rank == 0 and "results" in val_out:
            os.makedirs(cfg.PATH_SAVED, exist_ok=True)
            out_name = f"results_{cfg.RUN_NAME}"
            save_as_json(cfg.PATH_SAVED, val_out["results"], out_name)
            print(f"[rank 0] Saved {len(val_out['results'])} samples → …/{out_name}.json")
            if cfg.VALIDSIM == "TMM_FAST":
                print(
                    f" min valid MAE: {float(torch.min(test_mae).item()) if torch.isfinite(test_mae).any() else test_mae[epoch]:.4f}"
                )
                print(f"last valid MAE: {test_mae[epoch]:.4f}")
            else:
                print(f"Validation accuracy: {test_acc[epoch]:.2f}%")
                if rank == 0:
                    print(f"No validation at epoch {epoch}")
                    continue
        
        # ------------------------------ checkpointing ------------------------------
        if not cfg.NOTRAIN and rank == 0:
            if cfg.VALIDSIM != "TMM_FAST":
                # accuracy-based checkpointing
                new_acc = float(test_acc[epoch].item())
                if new_acc > (best_test_acc + 1e-7):
                    print(f"Checkpoint saved (ACC): new={new_acc:.6f} > best={best_test_acc:.6f} [epoch {epoch}]")
                    best_test_acc = new_acc
                    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint_best.pt"
                else:
                    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint.pt"
            else:
                # MAE-based checkpointing
                new_mae = float(test_mae[epoch].item())
                if new_mae < (best_test_mae - 1e-7):
                    print(f"Checkpoint saved (MAE): new={new_mae:.6f} < best={best_test_mae:.6f} [epoch {epoch}]")
                    best_test_mae = new_mae
                    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint_best.pt"
                else:
                    checkpoint = f"{cfg.PATH_SAVED}/ol3l-checkpoint.pt"
        
            os.makedirs(cfg["OUTPUT_PATH"], exist_ok=True)
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_losses=train_losses,
                train_acc=train_acc,
                test_acc=test_acc,
                test_mae=test_mae,
                extra={
                    "sampling": {
                        "temperature": cfg["TEMPERATURE"],
                        "top_k": cfg["TOP_K"],
                        "top_p": cfg["TOP_P"],
                    }
                },
            )


if __name__ == "__main__":
    optollama.utils.stop_ddp() # clean up old ddp sesssion in interactive mode

    # parse args and build final config
    args = optollama.utils.parse_arguments()
    cfg = optollama.utils.load_config(args)

    try:
        train(cfg)
    finally:
       optollama.utils.stop_ddp()
