from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import optollama.scripts.cli as cli
import torch
from optollama.dataloader.dataset import SpectraDataset, make_loader, make_repeated_spec_loader
from optollama.scripts.evaluate import validate_model
from optollama.scripts.inference import load_spectra_from_json_or_csv
from optollama.model.model import build_model
from optollama.scripts.runner import _is_ddp, setup_run
from optollama.utils.simulation_TMM_FAST import build_tmm_context
from optollama.utils.utils import init_tokenmaps, load_checkpoint, save_as_json, wl_mask

# ruff: noqa: N806


@torch.no_grad()
def run_inference(
    cfg: Any,
    ckpt: Optional[str] = None,
    mc_samples: Optional[int] = None,
    target: Optional[str] = None,
    n_targets: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[int, str], int, int, int]:
    """
    Run model inference / validation for a given configuration.

    This function sets up the runtime, builds the model, optionally loads a
    checkpoint, constructs the appropriate data loader (validation split or a
    user-provided target spectrum), and finally calls :func:`evaluate.validate_model`.

    Parameters
    ----------
    cfg:
        Configuration object with attributes such as
        ``PATH_DATA``, ``WAVELENGTHS``, ``PATH_CKPT``, ``PATH_SAVED``,
        ``MODEL_KEY``, ``D_MODEL``, ``N_BLOCKS``, ``N_HEADS``,
        ``STEPS``, ``DROPOUT``, ``MC_SAMPLES``, ``N_TARGETS``,
        ``NUM_SAMPLES_VALID``, ``VALID_BATCH``, ``MISMATCH_FILL_ORDER``, etc.
    ckpt:
        Optional checkpoint path overriding ``cfg.PATH_CKPT``. If provided and
        the file (or its ``*_best`` variant) exists, the model weights are
        loaded from there.
    mc_samples:
        Number of Monte-Carlo samples used in :func:`validate_model` (best-of-N
        decoding). If ``None``, falls back to ``cfg.MC_SAMPLES`` (default 1).
    target:
        Optional path to a JSON/CSV RAT file describing a *target* spectrum
        [R, A, T] over wavelength. When provided, the function builds a loader
        that repeats this base spectrum instead of using the validation set.
        When ``None``, the standard validation loader is used.
    n_targets:
        Number of repeated target samples to generate when ``target`` is given.
        If ``None``, falls back to ``cfg.N_TARGETS`` (default 1). Must be
        strictly positive when ``target`` is not ``None``.

    Returns
    -------
    Dict[str, Any]
        A dictionary with at least the following keys:

        - ``"mean_acc"`` (float): mean token accuracy over the (sub)dataset.
        - ``"mean_mae"`` (float | None): mean spectrum MAE (only for
          simulator modes that produce spectra, e.g. ``"TMM_FAST"``).
        - ``"results"`` (list[dict]): list of per-example records, including
          token stacks and RAT spectra. Present only on rank 0
          in DDP runs; in single-process runs it is always present.

    Notes
    -----
    - The function handles both single-process and DDP runs via
      :func:`runner.setup_run` and :func:`runner._is_ddp`.
    - Per-example results are saved as JSON in ``cfg.PATH_SAVED`` (if set),
      with a filename of the form ``results_{cfg.RUN_NAME}_{tag}.json`` where
      ``tag`` is ``"valid"`` or ``"target"``.
    """
    if cfg is None:
        raise ValueError("cfg must not be None; pass a loaded config object.")

    # Set up device / (optional) DDP. Single-process path works great in IDEs.
    device, local_rank, rank, world_size = setup_run(cfg, make_dirs=False)
    is_ddp = _is_ddp()

    # Tokens
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = init_tokenmaps(cfg.PATH_DATA)

    # --- always build model from a dataset sample to lock dims like training ---
    valid_subset_n = getattr(cfg, "NUM_SAMPLES_VALID")
    valid_ds, valid_loader, _ = make_loader(cfg, split="valid", subset_n=valid_subset_n, ddp=is_ddp)

    # figure out W (number of wavelength samples) from cfg or dataset
    if hasattr(cfg, "WAVELENGTHS"):
        w = len(cfg.WAVELENGTHS)
    else:
        ex = valid_ds.spectra[0] if isinstance(valid_ds, SpectraDataset) else valid_ds.dataset.spectra[0]
        w = int(ex.shape[0])  # fallback

    # ALWAYS use a stub [3, W] as the example for model sizing
    example_spec_for_build = torch.zeros((3, w), dtype=torch.float32, device=device)

    vocab_size = len(idx_to_token)
    max_stack = valid_ds.maximum_depth if isinstance(valid_ds, SpectraDataset) else valid_ds.dataset.maximum_depth

    model = build_model(
        model_type=getattr(cfg, "MODEL_KEY"),
        sample_spectrum=example_spec_for_build,  # [3,W]
        vocab_size=vocab_size,
        max_stack_depth=max_stack,
        d_model=getattr(cfg, "D_MODEL"),
        n_blocks=getattr(cfg, "N_BLOCKS"),
        n_heads=getattr(cfg, "N_HEADS"),
        timesteps=getattr(cfg, "STEPS"),
        dropout=getattr(cfg, "DROPOUT"),
        idx_to_token=idx_to_token,
        mask_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
        temperature=getattr(cfg, "TEMPERATURE"),
        top_k=getattr(cfg, "TOP_K"),
        top_p=getattr(cfg, "TOP_P"),
    )

    # Load weights (ckpt argument wins; else try cfg.PATH_CKPT)
    ckpt_path = cfg.PATH_CKPT
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        _, _ = load_checkpoint(ckpt_path, model, map_location="cpu", strict=True)
    else:
        print("No valid ckpt path set")

    # How many designed targets?
    n_tar = cfg.N_TARGETS
    if n_tar <= 0:
        print(f"use n_tar > 0. Got {n_tar}. n_tar = 1 used.")
        n_tar = 1

    # Build loader depending on branch
    if target is None:
        loader = valid_loader
    else:
        # Load base [3,W] spectrum with noise/smoothing DISABLED here;
        # fresh noise (if enabled) will be drawn per item by the repeated dataset.
        if target == "random":
            base_spec = torch.rand([3, 171], device=device)
        else:
            base_spec = load_spectra_from_json_or_csv(
                target,
                expect_shape="3xW",
                cfg=cfg,
                noise_cfg={"enabled": False},
                smooth_cfg={"enabled": False},
                mismatch_order=cfg.MISMATCH_FILL_ORDER,
            ).to(device)

        _, loader, _ = make_repeated_spec_loader(
            base_spec,
            n_tar,
            max_stack_depth=max_stack,
            pad_idx=pad_idx,
            wavelengths=getattr(cfg, "WAVELENGTHS"),
            cfg=cfg,
            batch_size=min(n_tar, getattr(cfg, "VALID_BATCH", 64)),
        )

    # Optional TMM context
    tmm_ctx = None
    if cfg.VALIDSIM == "TMM_FAST":
        try:
            tmm_ctx = build_tmm_context(cfg=cfg, idx_to_token=idx_to_token, device=device)
        except Exception as e:
            print(f"Could not initialize TMM context, falling back to NOSIM: {e}")
            cfg.VALIDSIM = "NOSIM"

    # Run validate_model — for a single item, it will still return the generated stack(s)
    out = validate_model(
        model,
        loader,
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
        track_step_mae=cfg.TRACK_STEP_MAE,
        roi_mask=wl_mask(cfg.WAVELENGTHS, cfg.ROI_MIN, cfg.ROI_MAX, device),
    )

    # Persist results on rank 0 (or single-process)
    if (not is_ddp) or rank == 0:
        path_saved = getattr(cfg, "PATH_SAVED", None)
        if path_saved:
            os.makedirs(path_saved, exist_ok=True)
        try:
            base = getattr(cfg, "RUN_NAME", "infer")
            tag = "target" if target else "valid"
            save_as_json(cfg.PATH_SAVED, out.get("results", []), f"results_{base}_{tag}")
            print(f"💾 Saved {len(out.get('results', []))} record(s) to {cfg.PATH_SAVED}")
        except Exception as e:
            print(f"Could not save results JSON: {e}")

    # Summary (accuracy/MAE may be None for single target without ground-truth)
    if out.get("mean_acc") is not None:
        print(f"✔ mean token accuracy: {100.0 * float(out['mean_acc']):.2f}%")
    if out.get("mean_mae") is not None:
        print(f"✔ mean spectrum MAE: {float(out['mean_mae']):.6f}")

    return out, idx_to_token, eos_idx, pad_idx, msk_idx


# ----------------------------- CLI entry -----------------------------
if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "./configs/config_OG_LOCAL.yaml"])
        sys.argv.extend(["--config", "./configs/config_OL_LOCAL.yaml"])
        # sys.argv.extend(["--config", "./OptoLlama/scripts/config_OL_HPCZ1.yaml"])

    # Parse args and build final config (applies --ckpt/--mc-samples/--validsim and --set)
    args = cli.parse_arguments()
    cfg = cli.load_config_with_overrides(args)

    out, idx_to_token, eos_idx, pad_idx, msk_idx = run_inference(
        cfg=cfg,
        ckpt=cfg.PATH_CKPT,
        mc_samples=cfg.MC_SAMPLES,
        target=getattr(cfg, "TARGET", None),
        n_targets=cfg.N_TARGETS,
    )

    # %%
    from metrics import masked_mae_roi

    tar = torch.tensor([out["results"][0]["rat_target"]])
    valkey = min(
        [
            [
                masked_mae_roi(
                    torch.tensor([out["results"][i]["rat_pred"]]), tar, wl_mask(cfg.WAVELENGTHS, cfg.ROI_MIN, cfg.ROI_MAX, "cpu")
                ),
                i,
            ]
            for i, item in enumerate(out["results"])
        ]
    )
    key = valkey[1]
    # key = 1

    # key = 0
    target_spec = torch.tensor(out["results"][key]["rat_target"])
    train_paths = sorted([getattr(cfg, k) for k in dir(cfg) if k.startswith("PATH_TRAIN")])

    from match_test_to_train import find_best_train_for_target, load_train_sample_by_global_id

    result = find_best_train_for_target(
        target_spec,
        train_paths=train_paths,  # or a list of dirs/files
        train_chunk_size=4 * 2048,
        wl_range=wl_mask(cfg.WAVELENGTHS, cfg.ROI_MIN, cfg.ROI_MAX, "cuda"),
    )
    print(result)
    nn_spectrum, nn_seq_ids, shard_path, local_idx = load_train_sample_by_global_id(
        global_id=result["best_global_index"], train_paths=train_paths
    )
    nn_sequence = [idx_to_token[int(t)] for t in nn_seq_ids[: cfg.MAX_SEQ_LEN] if int(t) not in (eos_idx, pad_idx, msk_idx)]

    from plots import plot_samples_clean_NN

    plot_samples_clean_NN(
        cfg,
        RAT_pred=torch.tensor(out["results"][key]["rat_pred"]),
        RAT_tar=target_spec,
        stack_pred=out["results"][key]["stack_pred_tokens"],
        stack_tar=out["results"][key]["stack_target_tokens"],
        ACC=out["results"][key]["acc"],
        number=cfg.MC_SAMPLES,
        RAT_nn=nn_spectrum,
        stack_nn=nn_sequence,
        nn_global_id=result["best_global_index"],
    )

# %%
# from utils import load_as_json
# save_path = r"d:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\runs\MD67\results"
# out = {}
# val_results = load_as_json('D:/Profile/a3536/Eigene Dateien/Github/OptoLlama/runs/MD67/results_MD67_valid_3k_steps_final.json')
# b = [1691, 1879, 2067, 2255, 2443, 2631, 2819, 3007]
# b.reverse()
# for ind in b:
#     val_results.pop(ind)
# # out = load_as_json('D:/Profile/a3536/Eigene Dateien/Github/OptoLlama/runs/MD67/results_MD67_valid_1k_steps_final.json')
# out['results'] = val_results
# from plots import plot_mae_trajectory, plot_mae_band
# example = out["results"][0]          # pick an example
# mae_traj = example["mae_traj"]       # list of length = steps

# plot_mae_trajectory(mae_traj, title=f"Example {example['dataset_index']}")

# mae_trajs = [
#     rec["mae_traj"]
#     for rec in out["results"]
#     if "mae_traj" in rec
#     ]

# plot_mae_band(mae_trajs, save_path, mode="percentile", title="MAE trajectory band")

# %% OptoGPT
# from utils import load_as_json
# data = load_as_json(r'd:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\runs\MD68\results\results_MD68_valid.json')
# # b = [1691, 1879, 2067, 2255, 2443, 2631, 2819, 3007]
# # b.reverse()
# # for ind in b:
#     # data.pop(ind)
# import numpy as np
# ll = [data[i]['mae'] for i in range(len(data))]
# mean = np.mean(ll)
# lower = np.percentile(ll, 10, axis=0)
# upper = np.percentile(ll, 90, axis=0)
# print(mean, lower, upper)

# %% Template
# from utils import unique_length_int_generator
# a = unique_length_int_generator(0e0, 1000000 - 1, 3000)
# nn_matches = load_as_json(r"d:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\data\TF_safetensors\test_to_train_nn.json")  # list with test_index, best_train_index, mae
# nnll = [nn_matches[i]['mae'] for i in a]
# nnmean = np.mean(nnll)
# lower = np.percentile(nnll, 10, axis=0)
# upper = np.percentile(nnll, 90, axis=0)
# print(mean, lower, upper)

# %%
# from utils import load_as_json
# from plots import plot_model_vs_nn_scatter

# # 1) Validation results from training/inference
# val_results = load_as_json(r"d:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\runs\MD67\results_MD67_valid_3k_steps_final.json")    # list of dicts from validate_model
# # val_results = load_as_json(r"d:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\runs\MD67\results\results_MD67_valid.json")    # list of dicts from validate_model
# # val_results = load_as_json(r"/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/Diffusion/runs/MD65/results/results_MD65_valid_1m.json")     # list of dicts from validate_model
# # val_results = val_results[:3000]
# b = [1691, 1879, 2067, 2255, 2443, 2631, 2819, 3007]
# b.reverse()
# for ind in b:
#     val_results.pop(ind)
# # 2) Nearest-neighbor mapping JSON you created
# nn_matches = load_as_json(r"d:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\data\TF_safetensors\test_to_train_nn.json")  # list with test_index, best_train_index, mae
# # nn_matches = load_as_json(r"/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/OptoLlama/data/TF_safetensors/test_to_train_nn.json")  # list with test_index, best_train_index, mae

# max_points=3000
# save_path = r"d:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\runs\MD67\results"
# # save_path = r"/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/Diffusion/runs/MD65/results"

# # 3) Plot ~1000 points
# plot_model_vs_nn_scatter(val_results, nn_matches, save_path, max_points)

# %%

# import pandas as pd
# path = r'D:/Profile/a3536/Eigene Dateien/Github/OptoLlama/data/targets'
# fileT = 'T_Morpho_txt_side.Probe.Rohdaten.csv'
# fileR = 'R_Morpho_txt_side.Probe.Rohdaten.csv'
# Rdata = pd.read_csv(rf'{path}/{fileR}', sep=';')
# Tdata = pd.read_csv(rf'{path}/{fileT}', sep=';')
# data = pd.DataFrame(pd.concat([Rdata['R'][::-1], 100-Rdata['R'][::-1]-Tdata['T'][::-1],Tdata['T'][::-1]], axis=1).values[::2]/100,index=Rdata['nm'][::-1].values[::2], columns=['R_measured','A_measured','T_measured'])
# # torch.load(rf'{path}/{fileR}')
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# data.plot(ax=ax)

# path_d = r'D:\Profile\a3536\Nextcloud\PhD - HEIBRiDS\Conferences\20250801_NatureMachineIntelligence\content paper\fig3\colorfilter\example-morphocolor.csv'
# data2 = pd.read_csv(path_d, sep=',')
# data2.index = Rdata['nm'][::-1].values[::2]
# data2[['RAT_tarR','RAT_tarA','RAT_tarT']].plot(ax=ax)
# # data_all = pd.concat([data.values,data2[['RAT_tarR','RAT_tarT','RAT_tarT']].values])
