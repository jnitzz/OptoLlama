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


def _set_loader_epoch(loader: torch.utils.data.DataLoader, epoch: int) -> None:
    """Set deterministic epoch state for either sampler-based or iterable datasets."""
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)

    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _example_spectrum(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """Return one representative spectrum without assuming eager dataset storage."""
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        first_idx = int(dataset.indices[0])
        if hasattr(base, "spectra"):
            return base.spectra[first_idx]
        sample = base[first_idx]
        return sample[0]

    if hasattr(dataset, "example_spectrum"):
        return dataset.example_spectrum()

    if hasattr(dataset, "spectra"):
        return dataset.spectra[0]

    sample = next(iter(dataset))
    return sample[0]


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

    # --- configuration checks
    ddp = optollama.utils.is_ddp()
    tmm = cfg["VALID_SIM"] == "TMM_FAST"
    
    # --- data loading and preprocessing ---
    _, _, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])

    train_ds, train_loader, _ = optollama.data.SpectraDataset.make_loader(
        cfg, 
        split="train", 
        subset_n=cfg["NUM_SAMPLES_TRAIN"], 
        ddp=ddp
    )
    _, test_loader, _ = optollama.data.SpectraDataset.make_loader(
        cfg, 
        split="test", 
        subset_n=cfg["NUM_SAMPLES_TEST"], 
        ddp=ddp
    )

    # --- TMM simulation ---
    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(
        cfg=cfg,
        idx_to_token=idx_to_token,
        device=device
    ) if tmm else None

    # --- model ---
    vocab_size = len(idx_to_token)
    example_spectrum = _example_spectrum(train_ds)
    
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
        spectrum_latent=cfg.get("SPECTRUM_LATENT"),
    ).to(device)

    # --- DDP wrapper ---
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    # --- optional initialization from an older checkpoint for fine-tuning ---
    init_checkpoint = cfg.get("INIT_CHECKPOINT_PATH")
    init_report = None
    resume_enabled = bool(cfg.get("RESUME_CHECKPOINT", True))
    if init_checkpoint and (not resume_enabled or not os.path.exists(cfg["LAST_CHECKPOINT_PATH"])):
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(f"INIT_CHECKPOINT_PATH does not exist: {init_checkpoint}")
        if rank == 0:
            print(f"Initializing model weights from {init_checkpoint}")
        init_report = optollama.utils.load_checkpoint_weights_for_init(
            init_checkpoint,
            model,
            map_location="cpu",
            strict=bool(cfg.get("INIT_CHECKPOINT_STRICT", True)),
            fallback_filter=bool(cfg.get("INIT_CHECKPOINT_FALLBACK_FILTER", True)),
        )
        if rank == 0:
            print(
                "Initialization load: "
                f"mode={init_report['mode']}, "
                f"fallback={init_report['fallback_used']}, "
                f"loaded={init_report['loaded_keys']}, "
                f"skipped={len(init_report['skipped_keys'])}"
            )

    # --- optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("LEARNING_RATE", 1e-4)))

    # --- metric buffers / resume bookkeeping ---
    train_losses = torch.zeros(cfg["EPOCHS"])
    train_acc = torch.zeros(cfg["EPOCHS"])
    test_acc = torch.zeros(cfg["EPOCHS"])
    test_mae = torch.full((cfg["EPOCHS"],), torch.inf)

    checkpoint = cfg["LAST_CHECKPOINT_PATH"]

    best_test_acc = 0.0
    best_test_mae = torch.inf
    start_epoch = 0

    if resume_enabled and checkpoint and os.path.exists(checkpoint):
        print(f"Resuming from checkpoint {checkpoint}")
        start_epoch, blob = optollama.utils.load_checkpoint(
            checkpoint, 
            model, 
            optimizer=optimizer, 
            map_location="cpu", 
            strict=True
        )
        
        # recover metric buffers if present
        train_losses = blob.get("train_losses", train_losses)
        train_acc = blob.get("train_acc", train_acc)
        test_acc = blob.get("test_acc", test_acc)
        test_mae = blob.get("test_mae", test_mae)

    if rank == 0:
        os.makedirs(cfg["OUTPUT_PATH"], exist_ok=True)

    # ------------------------------ epochs ------------------------------
    epochs = cfg["EPOCHS"]
    for epoch in range(start_epoch, epochs):
        epoch_loss_sum = 0.0
        epoch_correct = 0.0
        epoch_tokens = 0.0

        # DDP epoch seeds
        _set_loader_epoch(train_loader, epoch)
        _set_loader_epoch(test_loader, epoch)

        # ------------------------------ train ------------------------------
        model.train()
        if rank == 0:
            pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} train", leave=True)

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            spectra, stacks = batch[0].to(device), batch[1].to(device)
            logits = model(spectra, stacks)

            log_probs = torch.nn.functional.log_softmax(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
            loss = torch.nn.NLLLoss(ignore_index=pad_idx)(
                log_probs.view(-1, vocab_size), 
                stacks.view(-1)
            )

            loss.backward()
            optimizer.step()

            # --- logging (DDP-avg for losses) ---
            with torch.no_grad():
                total_loss = loss.detach().clone()
                if ddp:
                    torch.distributed.all_reduce(total_loss)
                    total_loss /= world_size

                epoch_loss_sum += total_loss.item()

                correct_count, total_count, _, _ = optollama.evaluation.token_accuracy_counts(
                    stacks, 
                    logits.argmax(dim=-1), 
                    eos_idx, 
                    pad_idx, 
                    msk_idx
                )
                batch_totals = torch.tensor(
                    [float(correct_count.item()), float(total_count.item())],
                    device=device,
                    dtype=torch.float64,
                )
                if ddp:
                    torch.distributed.all_reduce(batch_totals)

                epoch_correct += float(batch_totals[0].item())
                epoch_tokens += float(batch_totals[1].item())

                train_losses[epoch] = epoch_loss_sum / (i + 1)
                train_acc[epoch] = epoch_correct / max(epoch_tokens, 1.0)

            if rank == 0:
                pbar.set_postfix(
                    loss_CE=f"{train_losses[epoch]:.4f}",
                    acc=f"{train_acc[epoch] * 100:.4f}%",
                )
                pbar.update()

        if rank == 0:
            pbar.close()

        # ------------------------------ validation ------------------------------
        model.eval()
        test_output = optollama.evaluation.model_prediction(
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
        test_acc[epoch] = test_output["mean_acc"]
        if tmm:
            test_mae[epoch] = test_output["mean_mae"]

        # save per-example results (rank 0 only)
        if rank == 0:
            samples = len(test_output["results"])
            optollama.utils.save_as_json(cfg["SAMPLES_PATH"], test_output["results"])
            print(f"[rank 0] Saved {samples} samples -> {cfg['SAMPLES_PATH']}")

            if tmm:
                print(f"\tmin test MAE: {torch.min(test_mae).item():.6f}")
                print(f"\tlast test MAE: {test_mae[epoch]:.6f}")
            else:
                print(f"\ttest accuracy: {test_acc[epoch]:.2f}%")
        
        # ------------------------------ checkpointing ------------------------------
        if rank == 0:
            checkpoint_paths = [cfg["LAST_CHECKPOINT_PATH"],]

            if tmm:
                # MAE-based checkpointing
                new_mae = test_mae[epoch].item()
                if new_mae < best_test_mae:
                    print(f"Saving best checkpoint (MAE): new={new_mae:.6f} < best={best_test_mae:.6f} [epoch {epoch + 1}]")
                    best_test_mae = new_mae
                    checkpoint_paths.append(cfg["BEST_CHECKPOINT_PATH"])
            else:
                # accuracy-based checkpointing
                new_acc = test_acc[epoch].item()
                if new_acc > best_test_acc:
                    print(f"Saving best checkpoint (ACC): new={new_acc:.2f} > best={best_test_acc:.2f} [epoch {epoch + 1}]")
                    best_test_acc = new_acc
                    checkpoint_paths.append(cfg["BEST_CHECKPOINT_PATH"])
        
            for path in checkpoint_paths:
                optollama.utils.save_checkpoint(
                    path,
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
                        },
                        "init_checkpoint": init_report,
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
