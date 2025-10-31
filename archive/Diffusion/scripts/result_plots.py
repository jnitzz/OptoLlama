# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 10:01:50 2025

@author: a3536
"""


from utils import load_JSONPICKLE_NEW
import torch
import matplotlib.pyplot as plt
import os

##### plot denoisnig steps #####
import config_MD49 as cfg
data = load_JSONPICKLE_NEW(cfg.PATH_RUN, 'MD49_inference_N100_MC10_K1_P0.0') #/Diffusion/runs/MD49/MD49_inference_N10_MC3_K5.0_P0.0.json
dn_tensor_DiT = torch.full((500,len(data)), 0, dtype=torch.float)
for i in range(len(data)):
    dn_tensor_DiT[:,i] = torch.tensor(data[i]['denoise_steps'])

import config_MD50 as cfg
data = load_JSONPICKLE_NEW(cfg.PATH_RUN, 'MD50_inference_N1000_MC10_K3_P0.0_summary')
# dn_tensor_DoT = torch.full((1,len(data)), 0, dtype=torch.float)
# for i in range(len(data)):
dn_tensor_DoT = torch.tensor([data[i]['mae'] for i in range(len(data))])

group2_mean = dn_tensor_DiT.mean(dim=1).cpu().detach().numpy()
print(group2_mean[-1])
group1_mean = dn_tensor_DoT.mean(dim=0).cpu().detach().numpy().repeat(500)
print(group1_mean[-1])

import pickle

# Path to your pickle file
# file_path = os.path.join(cfg.PATH_RUN, 'MD50_inference_N1000000_MC10_K3_P0.0.pkl')

# Load the pickle file
# with open(file_path, "rb") as file:
#     data = pickle.load(file)

#%% plot denoising steps
fig, ax = plt.subplots(figsize=(10, 5))
plt.rc("font", family="serif")

# Group 1: Remaining columns BO10
group2_mean = dn_tensor_DiT.mean(dim=1).cpu().detach().numpy()
group2_std = dn_tensor_DiT.std(dim=1).cpu().detach().numpy()

# Group 2: Remaining columns PBO10
group1_mean = dn_tensor_DoT.mean(dim=0).cpu().detach().numpy().repeat(500)
group1_std = dn_tensor_DoT.std(dim=0).cpu().detach().numpy().repeat(500)

# # Group 3: Remaining columns DE10
# group3_mean = df_convVals.iloc[:, 20:30].mean(axis=1)
# group3_std = df_convVals.iloc[:, 20:30].std(axis=1)

# # Group 4: Remaining columns SM10
# group4_mean = df_convVals.iloc[:, 30:40].mean(axis=1)
# group4_std = df_convVals.iloc[:, 30:40].std(axis=1)

# # Group 5: Remaining columns CM10
# group5_mean = df_convVals.iloc[:, 40:50].mean(axis=1)
# group5_std = df_convVals.iloc[:, 40:50].std(axis=1)

data_names = ['decoder-only transformer', 'masked diffusion transformer', ]

# Plot the mean lines
colmap = "tab10" # colmap[i]
cmap = plt.get_cmap('tab10')
hgfgray = (90/255, 105/255, 110/255)
hgfblue = (0/255, 90/255, 160/255)
hgfgreen= (140/255, 180/255, 35/255)
hgfhighlight=(205/255, 238/255, 251/255)
hgfmint = (5/255, 229/255, 186/255)
hgfenergy= (255/255, 210/255, 40/255)
ax.plot(group1_mean, color=hgfenergy, linewidth=1.0, label=data_names[0], zorder=3)
ax.plot(group2_mean, color=hgfblue, linewidth=1.0, label=data_names[1], zorder=3)
# group1_mean.plot(ax=ax, color=cmap(0), linewidth=1.0, label=data_names[0], zorder=3)
# group2_mean.plot(ax=ax, color=cmap(1), linewidth=1.0, label=data_names[1], zorder=3)
# group3_mean.plot(ax=ax, color=cmap(2), linewidth=1.0, label=data_names[2], zorder=3)
# group4_mean.plot(ax=ax, color=cmap(3), linewidth=1.0, label=data_names[3], zorder=3)
# group5_mean.plot(ax=ax, color=cmap(4), linewidth=1.0, label=data_names[4], zorder=3)

# Fill the area between the mean and standard deviation
ax.fill_between(range(dn_tensor_DiT.size(0)), group1_mean - group1_std, group1_mean + group1_std, color=hgfenergy, alpha=0.5, zorder=2)#, label='MC-BO range')
ax.fill_between(range(dn_tensor_DiT.size(0)), group2_mean - group2_std, group2_mean + group2_std, color=hgfblue, alpha=0.5, zorder=2)#, label='SC-BO range')
# ax.fill_between(df_convVals.index, group3_mean - group3_std, group3_mean + group3_std, color=cmap(2), alpha=0.2, zorder=2)#, label='CE range')
# ax.fill_between(df_convVals.index, group4_mean - group4_std, group4_mean + group4_std, color=cmap(3), alpha=0.2, zorder=2)#, label='DE range')
# ax.fill_between(df_convVals.index, group5_mean - group5_std, group5_mean + group5_std, color=cmap(4), alpha=0.2, zorder=2)#, label='DS range')

# Add legend and labels

# ax.set_yscale('log')
ax.grid('both')
ax.set_ylim(bottom=0)
# ax.set_ylim(top=0.25)
# ax.set_title("Performance of multi-channel Bayesian Optimization (MC-BO), single-channel BO (SC-BO), \nCMA-ES (CE), differential evolution (DE) and downhill simplex (DS)", fontsize=10)
# ax.set_title(f'Mean Absolute Error (MAE) per Denoising Step of Inference and Groundtruth [model: {cfg.RUN_NAME}]')
ax.legend(loc='upper center', bbox_to_anchor=(0.50, 1.17),
          ncol=5, fancybox=True, shadow=True,fontsize=15)
ax.set_xlabel("denoising step", fontsize=18)
ax.set_ylabel("MAE (inference vs groundtruth)", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.show()
# plt.savefig(r"D:\Profile\a3536\Nextcloud\PhD - HEIBRiDS\Conferences\20250801_NatureMachineIntelligence\denoising_steps_N1000_MC3_K5.0_P0.0.pdf", dpi=600, bbox_inches='tight')

# %% plot scatter comparison training data

import pickle
import numpy as np
from utils import init_tokenmaps, save_JSONPICKLE_NEW, load_JSONPICKLE_NEW
from call_rayflare import Call_RayFlare_with_dict
from plots import plot_samples, plot_mae, train_data_comp, plot_mae_comparison
from typing import Any, Dict, List, Mapping, Tuple, Union, Optional
import config_MD44 as cfg

out_name = f'{cfg.RUN_NAME}_valid_XX_inference_results'
results_MD03 = load_JSONPICKLE_NEW(cfg.PATH_RUN, out_name)

MAEs = [j['mae'] for j in results_MD03]
np.mean(MAEs)
# plot_mae(cfg, MAEs)

ds_search_path = os.path.join(cfg.PATH_DATA, r'data_test_crop_10k_closest_mae.pt')

MAE_scatter = train_data_comp(cfg,
                results_MD03,
                # ds_search=ds_search_path,
                ds_search_precalc=ds_search_path,
                sample_plots=False)

# all for comp plot
MAEs_trainset = [MAE_scatter[j]['mae_closest_crop'] for j in MAE_scatter]
# temp = cfg.RUN_NAME
# cfg.RUN_NAME = 'trainset'
# plot_mae(cfg, MAEs_trainset)

# temp = cfg.RUN_NAME
# cfg.RUN_NAME = 'OL15'
# plot_mae(cfg, [i['mae'] for i in per_sample_results][:])

# abs_errs = [np.abs(i[1]['spectrum_mae'] - i[1]['mae_closest']) for i in MAE_scatter.items()]
# temp = cfg.RUN_NAME
# cfg.RUN_NAME = 'trainset'
# plot_mae(cfg, abs_errs)

# cfg.RUN_NAME = temp
# with open(cfg.PATH_RES_COMP, 'rb') as f:   
#     per_example_results = pickle.load(f)
# per_sample_results = load_JSONPICKLE(r"d:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\results\OL15", 'val_sample_results_test_OL15_E331')
# comp_dict_all = {'trainset': MAEs_trainset, f'{cfg.RUN_NAME}': MAEs, 'N67_GPT26': [i[3] for i in per_example_results][::10], 'OL15': [i['mae'] for i in per_sample_results][:1000]} #N67_GPT26_Epoch207
# comp_dict = {k: comp_dict_all[k] for k in (f'{cfg.RUN_NAME}', 'OL15')}
# plot_mae_comparison(cfg, comp_dict)

MAE_scatter_sorted = sorted(MAE_scatter.items(), key=lambda x: np.abs(x[1]['spectrum_mae']-x[1]['mae_closest']))
MAE_scatter_sorted = sorted(MAE_scatter.items(), key=lambda x: x[1]['mae_closest'])
MAE_scatter_sorted = sorted(MAE_scatter.items(), key=lambda x: x[1]['spectrum_mae'])
# MAE_scatter_sorted = [item for item in MAE_scatter_sorted if item[1]['spectrum_mae']<=item[1]['mae_closest']+0.01]
# MAE_scatter_sorted = [item for item in MAE_scatter_sorted if item[1]['spectrum_mae']>=item[1]['mae_closest']]
# MAE_scatter_sorted = [item for item in MAE_scatter_sorted if item[1]['spectrum_mae']<=item[1]['mae_closest']+0.005]
# MAE_scatter_sorted = [item for item in MAE_scatter_sorted if item[1]['spectrum_mae']<=item[1]['mae_closest']]

# %%
import config_MD42 as cfg

out_name = f'{cfg.RUN_NAME}_valid_XX_inference_results'
results_MD03 = load_JSONPICKLE_NEW(cfg.PATH_RUN, out_name)

MAEs = [j['mae'] for j in results_MD03]
np.mean(MAEs)
# plot_mae(cfg, MAEs)

ds_search_path = os.path.join(cfg.PATH_DATA, r'data_test_crop_10k_closest_mae.pt')

MAE_scatter = train_data_comp(cfg,
                results_MD03,
                # ds_search=ds_search_path,
                ds_search_precalc=ds_search_path,
                sample_plots=False)

# all for comp plot
MAEs_trainset = [MAE_scatter[j]['mae_closest_crop'] for j in MAE_scatter]
MAE_scatter_sorted2 = sorted(MAE_scatter.items(), key=lambda x: np.abs(x[1]['spectrum_mae']-x[1]['mae_closest']))
MAE_scatter_sorted2 = sorted(MAE_scatter.items(), key=lambda x: x[1]['mae_closest'])
MAE_scatter_sorted2 = sorted(MAE_scatter.items(), key=lambda x: x[1]['spectrum_mae'])


# %%

print(len(MAE_scatter_sorted))
index = [_ for _, inner in MAE_scatter_sorted]
index = np.arange(len(index))
y = [inner['mae_closest']  for _, inner in MAE_scatter_sorted]
x2 = [inner['spectrum_mae'] for _, inner in MAE_scatter_sorted2]
x = [inner['spectrum_mae'] for _, inner in MAE_scatter_sorted]

# 2) Make the scatter plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(index, x, label='inference DiT',zorder=3)
plt.scatter(index, x2, label='inference T',zorder=2)
plt.scatter(index, y, label='trainset',zorder=1)

# 3) Labeling
plt.xlabel('index (sorted by trainset)')
plt.ylabel('MAE spectrum')
plt.title(f"Index vs MAE Spectrum of [Inference, Trainset] [model: {out_name.split('_')[0]}]")
# plt.title(f'Index vs [Spectrum MAE, MAE Closest] [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]')

# 4) (Optional) add a grid and show
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
