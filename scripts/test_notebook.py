# %%
"""
Notebook-style plotting consumer for OptoLlama inference outputs.

Preferred reproducible entrypoint:
    python scripts/plot_results.py dashboard --config ./configs/optollama.yaml

This file is kept for notebook-style inspection and uses the same plotting API
as the CLI.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import optollama.data
import optollama.evaluation
import optollama.plotting
import optollama.utils


# %%
# 1) Config
DEFAULT_CONFIG = "./configs/optollama.yaml"
NN_MATCHES_PATH = None
COMPUTE_NN_FOR_SELECTED_SAMPLE = False
NN_CHUNK_SIZE = 4096

if "--config" not in sys.argv:
    sys.argv.extend(["--config", DEFAULT_CONFIG])

args = optollama.utils.parse_arguments()
cfg = optollama.utils.load_config(args)

results_path = args.results or cfg["SAMPLES_PATH"]
bundle_path = cfg["PLOT_BUNDLE_PATH"]

print("Config:", args.config)
print("Results:", results_path)
print("Plot bundle:", bundle_path)
print("NN matches:", NN_MATCHES_PATH)


# %%
# 2) Load saved inference outputs
results = optollama.plotting.load_results(results_path)
bundle = optollama.plotting.load_plot_bundle(bundle_path)
tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])

print("Loaded results:", len(results))
print("Has MAE grid:", bundle.mae_grid is not None)
print("Has predicted spectra grid:", bundle.pred_spectra_grid is not None)
print("Has MAE trajectory grid:", bundle.mae_traj_grid is not None)


# %%
# 3) Sample plot
sample_index = optollama.plotting.select_best_result_index(results)
sample = results[sample_index]
wavelengths = bundle.wavelengths if bundle.wavelengths is not None else cfg["WAVELENGTHS"].detach().cpu().numpy()
roi_mask = optollama.data.wavelength_mask(wavelengths, cfg.get("ROI_MIN"), cfg.get("ROI_MAX"), "cpu")

fig_sample = optollama.plotting.plot_sample_comparison(
    wavelengths=wavelengths,
    target_spectrum=sample["rat_target"],
    predicted_spectrum=sample["rat_pred"],
    predicted_tokens=sample.get("stack_pred_tokens", []),
    target_tokens=sample.get("stack_target_tokens", []),
    sample_acc=sample.get("acc"),
    sample_mae=sample.get("mae"),
    mc_samples=cfg.get("MC_SAMPLES"),
    roi_min=cfg.get("ROI_MIN"),
    roi_max=cfg.get("ROI_MAX"),
    title=f"Sample {sample_index}",
)
plt.show()


# %%
# 4) Dashboard plot
if bundle.mae_grid is not None:
    pred_tokens_grid = (
        optollama.plotting.build_pred_tokens_grid(bundle.ids_grid, idx_to_token, eos_idx, pad_idx, msk_idx)
        if bundle.ids_grid is not None
        else None
    )

    target_spec = optollama.plotting.results_target_spectra(results)
    fig_dashboard = optollama.plotting.plot_mc_dashboard(
        mae_grid=bundle.mae_grid,
        target_spec=target_spec,
        pred_spec_grid=bundle.pred_spectra_grid,
        pred_tokens_grid=pred_tokens_grid,
        wavelengths=wavelengths,
        topk=3,
        title=f"{cfg['MODEL']} {cfg['RUN']} dashboard",
    )
    plt.show()


# %%
# 5) Optional MAE trajectory for the globally best (target, mc) cell
if bundle.mae_traj_grid is not None and bundle.mae_grid is not None:
    best_target_idx, best_mc_idx = np.unravel_index(np.argmin(bundle.mae_grid), bundle.mae_grid.shape)
    fig_step_traj = optollama.plotting.plot_mae_trajectory(
        bundle.mae_traj_grid[best_target_idx, best_mc_idx],
        title=f"Target {best_target_idx}, MC {best_mc_idx} MAE trajectory",
    )
    plt.show()


# %%
# 6) Optional MAE trajectory band across MC samples for one target
if bundle.mae_traj_grid is not None:
    best_target_idx = int(bundle.mae_grid.mean(axis=1).argmin()) if bundle.mae_grid is not None else 0
    fig_traj = optollama.plotting.plot_mae_band(
        bundle.mae_traj_grid[best_target_idx],
        mode="percentile",
        title=f"Target {best_target_idx} MAE trajectory band",
    )
    plt.show()


# %%
# 7) Optional nearest-neighbor comparison on the sample plot
if COMPUTE_NN_FOR_SELECTED_SAMPLE:
    from template_matching import find_best_train_for_target, load_train_sample_by_global_id

    train_paths = sorted([cfg[key] for key in cfg.keys() if key.startswith("DATA_PATH_TRAIN")])
    nn_result = find_best_train_for_target(
        torch.tensor(sample["rat_target"]),
        train_paths=train_paths,
        train_chunk_size=NN_CHUNK_SIZE,
        wl_range=roi_mask,
    )
    nn_spectrum, nn_ids, shard_path, local_idx = load_train_sample_by_global_id(
        nn_result["best_global_index"],
        train_paths=train_paths,
    )
    nn_tokens = [
        idx_to_token[int(token_id)]
        for token_id in nn_ids.tolist()
        if int(token_id) not in (eos_idx, pad_idx, msk_idx)
    ]

    fig_sample_nn = optollama.plotting.plot_sample_comparison(
        wavelengths=wavelengths,
        target_spectrum=sample["rat_target"],
        predicted_spectrum=sample["rat_pred"],
        predicted_tokens=sample.get("stack_pred_tokens", []),
        target_tokens=sample.get("stack_target_tokens", []),
        sample_acc=sample.get("acc"),
        sample_mae=sample.get("mae"),
        mc_samples=cfg.get("MC_SAMPLES"),
        roi_min=cfg.get("ROI_MIN"),
        roi_max=cfg.get("ROI_MAX"),
        nn_spectrum=nn_spectrum,
        nn_tokens=nn_tokens,
        nn_id=nn_result["best_global_index"],
        nn_mae=nn_result["best_mae"],
        title=f"Sample {sample_index} with NN comparison",
    )
    plt.show()


# %%
# 8) Optional model-vs-nearest-neighbor scatter from a saved matches JSON
if NN_MATCHES_PATH:
    nn_matches = optollama.utils.load_as_json(NN_MATCHES_PATH)
    fig_nn_scatter = optollama.plotting.plot_model_vs_nn_scatter(
        results,
        nn_matches,
        title=f"{cfg['MODEL']} {cfg['RUN']} vs nearest-neighbor baseline",
    )
    plt.show()
