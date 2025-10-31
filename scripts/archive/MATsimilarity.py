import os
import random
import datetime
import numpy as np
import pickle
import pandas as pd
import torch
import torch.distributed as dist
from utils import load_JSONPICKLE, seed_everything
import config_OL19 as cfg
tokens = load_JSONPICKLE(cfg.PATH_DATA, "tokens")
for s in ("<PAD>", "<SOS>", "<EOS>"):
    if s not in tokens:
        tokens.append(s)

from main import _prepare_token_maps

idx_to_token, mat2id, pad_idx, sos_idx, eos_idx = _prepare_token_maps(cfg)
from solcore import material

import seaborn as sns
from sklearn.gaussian_process.kernels import RBF, Matern


df_n_mae = pd.DataFrame(index=mat2id.keys(),columns=mat2id.keys())
df_k_mae = pd.DataFrame(index=mat2id.keys(),columns=mat2id.keys())
df_nk_mae = pd.DataFrame(index=mat2id.keys(),columns=mat2id.keys())
df_n_mse = pd.DataFrame(index=mat2id.keys(),columns=mat2id.keys())
df_k_mse = pd.DataFrame(index=mat2id.keys(),columns=mat2id.keys())
df_nk_mse = pd.DataFrame(index=mat2id.keys(),columns=mat2id.keys())
df_nk_RBF = pd.DataFrame(index=mat2id.keys(),columns=mat2id.keys())
df_nk_Matern = pd.DataFrame(index=mat2id.keys(),columns=mat2id.keys())
for item in mat2id.keys():
    for item2 in mat2id.keys():
        pred_str = item
        true_str = item2
        
        # check material similarities
        pred_mat = material(pred_str)()
        true_mat = material(true_str)()
        pred_mat_intp_n = pred_mat.n_interpolated(np.arange(300, 2001, 10) * 1e-9)
        true_mat_intp_n = true_mat.n_interpolated(np.arange(300, 2001, 10) * 1e-9)
        pred_mat_intp_k = pred_mat.k_interpolated(np.arange(300, 2001, 10) * 1e-9)
        true_mat_intp_k = true_mat.k_interpolated(np.arange(300, 2001, 10) * 1e-9)
        
        n_mae = np.mean(np.absolute(pred_mat_intp_n - true_mat_intp_n))
        k_mae = np.mean(np.absolute(pred_mat_intp_k - true_mat_intp_k))
        nk_mae = np.mean(np.absolute(np.concatenate([pred_mat_intp_n,pred_mat_intp_k]) - np.concatenate([true_mat_intp_n,true_mat_intp_k])))
        df_n_mae.loc[item,item2] = n_mae
        df_k_mae.loc[item,item2] = k_mae
        df_nk_mae.loc[item,item2] = nk_mae
        
        n_mse = np.mean(np.square(pred_mat_intp_n - true_mat_intp_n))
        k_mse = np.mean(np.square(pred_mat_intp_k - true_mat_intp_k))
        nk_mse = np.mean(np.square(np.concatenate([pred_mat_intp_n,pred_mat_intp_k]) - np.concatenate([true_mat_intp_n,true_mat_intp_k])))
        df_n_mse.loc[item,item2] = n_mse
        df_k_mse.loc[item,item2] = k_mse
        df_nk_mse.loc[item,item2] = nk_mse
        
        # Stack curves as vectors
        curve1n = pred_mat_intp_n.reshape(-1, 1)
        curve1k = pred_mat_intp_k.reshape(-1, 1)
        curve2n = true_mat_intp_n.reshape(-1, 1)
        curve2k = true_mat_intp_k.reshape(-1, 1)
        
        # Define kernels
        rbf_kernel = RBF(length_scale=1.0)
        matern_kernel = Matern(length_scale=1.0, nu=1.5)
        
        # Compute similarity between curves using kernels
        rbf_similarity_n = rbf_kernel(curve1n, curve2n)
        rbf_similarity_k = rbf_kernel(curve1k, curve2k)
        matern_similarity_n = matern_kernel(curve1n, curve2n)
        matern_similarity_k = matern_kernel(curve1k, curve2k)
        
        # Convert similarity to a scalar measure (e.g., mean similarity)
        rbf_distance = np.mean(np.concatenate([rbf_similarity_n,rbf_similarity_k]))
        matern_distance = np.mean(np.concatenate([matern_similarity_n,matern_similarity_k]))
        
        df_nk_RBF.loc[item,item2] = rbf_distance
        df_nk_Matern.loc[item,item2] = matern_distance
        
        if item == item2:
            if float(rbf_distance) != 1.0 or float(matern_distance) != 1.0:
                df_nk_RBF.loc[item,item2] = 1.0
                df_nk_Matern.loc[item,item2] = 1.0
        
        # print(f'RBF Kernel Distance: {rbf_distance:.4f}')
        # print(f'Matérn Kernel Distance: {matern_distance:.4f}')
        
        # # Plot curves
        # plt.plot(range(len(curve1n)), curve1n, label='Curve 1n')
        # plt.plot(range(len(curve1k)), curve1k, label='Curve 1k')
        # plt.plot(range(len(curve2n)), curve2n, label='Curve 2n')
        # plt.plot(range(len(curve2k)), curve2k, label='Curve 2k')
        # plt.legend()
        # plt.title('Comparison of Two Curves')
        # plt.show()
        
df_final = np.sqrt((np.square(df_n_mae) + np.square(df_k_mae)).astype('float'))
# sns.heatmap(df_final.astype('float'))
import matplotlib.pyplot as plt
# plt.show()

df_final2 = 1/df_final
df_final2.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final3 = df_final2/df_final2.max().max()
# sns.heatmap(df_final3.astype('float'))
# plt.show()

df_final3.replace(np.nan, 1, inplace=True)
df_final3 = df_nk_RBF.copy()
# df_final3 = df_nk_Matern.copy()
df_final3 = df_final3.astype('float')

sns.clustermap(df_final3)
plt.show()

# df_final2 = 1/df_nk_mse
# df_final2.replace([np.inf, -np.inf], np.nan, inplace=True)
# df_final3 = df_final2/df_final2.max().max()
# sns.heatmap(df_final3.astype('float'))
# plt.show()

# sns.heatmap(df_n_mae.astype('float'))
# plt.show()
# sns.heatmap(df_k_mae.astype('float'))
# plt.show()
# sns.heatmap(df_nk_mae.astype('float'))
# plt.show()
        
# sns.heatmap(df_n_mse.astype('float'))
# plt.show()
# sns.heatmap(df_k_mse.astype('float'))
# plt.show()
# sns.heatmap(df_nk_mse.astype('float'))
# plt.show()

# for item in [df_nk_Matern, df_nk_RBF, df_nk_mae, df_nk_mse]:
#     sns.heatmap(item.astype('float'))
#     plt.show()

# sns.heatmap(df_nk_mae.astype('float'))

#%%
df = df_nk_mae.astype(float)

def softmax_row(row):
    scores = -row
    scores -= scores.max()
    exps   = np.exp(scores)       # now this is a float64 Series
    return exps / exps.sum()

probs = df.apply(softmax_row, axis=0)
sns.heatmap(probs)

for alpha in np.arange(1,30,3):
    # 1) make sure your MSE’s are floats
    df = df_nk_mae.copy()
    df = df.astype(float)
    
    # 2) choose a “sharpening” factor α – larger means more one-hot-y
    alpha = 25.0
    
    # 3) build scores = –α·MSE
    scores = -df * alpha
    
    # 4) shift each row by its max (numerical stability)
    scores = scores.sub(scores.max(axis=1), axis=0)
    
    # 5) exponentiate and normalize per row
    exp_scores = np.exp(scores)
    probs = exp_scores.div(exp_scores.sum(axis=1), axis=0)
    sns.heatmap(probs)
    plt.show()

probs.sum(axis=1)
# torch.save(torch.Tensor(probs.to_numpy().tolist()),'sim_matrix.pth')
sim_matrix = torch.load('sim_matrix.pth')
rr
sns.heatmap(rr)