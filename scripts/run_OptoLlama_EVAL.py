import os
import random
import datetime
import numpy as np
import pickle
import torch
import torch.distributed as dist

from utils import seed_everything, load_JSONPICKLE, save_JSONPICKLE, generate_signal, plot_signal
from plots import plot_accuracy, plot_mse, plot_mae, plot_samples, plot_mae_comparison, plot_mse_comparison, plot_acc_comparison

from call_rayflare import DBinitNewMats, Call_RayFlare_with_dict
import config_EVAL as c

from pathlib import Path
import sys

# make sure the repo root is on sys.path
repo_root = Path(__file__).resolve().parent
sys.path.append(str(repo_root))             # if not already on PYTHONPATH

import main as optical_main         # adjust to your package layout

###############################################################################
# 8) Main
###############################################################################
def main():
    # If you're not using distributed, you can comment out these lines:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend='nccl', init_method='env://',
                                timeout=datetime.timedelta(seconds=1800))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    if local_rank == 0:
        os.makedirs(c.PATH_RUN, exist_ok=True)

    seed_everything(int(random.random()*1000))

    # ------------------------------------------------------------------
    # Load your token definitions (material tokens)
    # ------------------------------------------------------------------
    tokens = load_JSONPICKLE(c.PATH_DATA, 'tokens')

    # Insert special tokens if not present
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    for special_tk in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
        if special_tk not in tokens:
            tokens.append(special_tk)

    token_to_idx = {tk: i for i, tk in enumerate(tokens)}
    pad_idx = token_to_idx[PAD_TOKEN]
    sos_idx = token_to_idx[SOS_TOKEN]
    eos_idx = token_to_idx[EOS_TOKEN]
    idx_to_token = {i: tk for i, tk in enumerate(token_to_idx)}
    
    c.THICKNESSES = np.arange(c.THICKNESS_MIN,c.THICKNESS_MAX+1,c.THICKNESS_STEPS)
    c.WAVELENGTHS = np.arange(c.WAVELENGTH_MIN,c.WAVELENGTH_MAX+1,c.WAVELENGTH_STEPS)
    
    #%% target generation methods
    if c.TARGET == 'file':
        # target = {}
        # from numpy import genfromtxt
        # file_path = os.path.join(c.PATH_DATA, 'filter_10-2000-10_R.csv')
        # target['R'] = genfromtxt(file_path, delimiter=',')
        # file_path = os.path.join(c.PATH_DATA, 'filter_10-2000-10_T.csv')
        # target['T'] = genfromtxt(file_path, delimiter=',')
        # target['A'] = np.array([1-(float(a)+float(b)) for a,b in zip(target['R'],target['T'])]) #1 - (target['R'] + target['T'])
        # spectrum_values = np.concatenate([target['R'], target['A'], target['T']])
        stack_str_list = ['ARC_ISE__100','soda-lime_0-01__2000','9327_SiN_fit_formatted__142.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__203.0', '9333_TiO2_fit_formatted__164.5', '9327_SiN_fit_formatted__90.7', 'ITO__110.0','3Hal_LZ__550.0', 'Ag__100.0', '<EOS>']
        stack_str_list = ['ARC_ISE__100','soda-lime_0-01__2000','9327_SiN_fit_formatted__142.0', 'TiO2__165.5', '9327_SiN_fit_formatted__203.0', 'TiO2__164.5', '9327_SiN_fit_formatted__90.7', 'ITO__110.0', '3Hal_LZ__550.0', 'C60_HZB__23', 'BCP__8.0','Ag__100.0', '<EOS>']
        stack_str_list = ['ARC_ISE__100','soda-lime_0-01__2000','9327_SiN_fit_formatted__142.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__203.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__90.7', 'ITO__110.0', '<EOS>']
        stack_str_list = ['ARC_ISE__100','soda-lime_0-01__2000','9327_SiN_fit_formatted__142.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__203.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__90.7', '<EOS>']
        # stack_str_list = ['9327_SiN_fit_formatted__142.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__203.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__90.7', '<EOS>']
        # stack_str_list = []
        pure_signal = Call_RayFlare_with_dict(c, stack_str_list)
        
        wl0, wl1, step = 300, 2000, 10
        baseline_segments = [
            {"start": 300,  "end":  400, "value": 0.00, 'noise': 0.25},
            {"start": 400,  "end":  700, "value": 0.00, 'noise': 0.1},
            {"start": 700,  "end": 2000, "value": 0.00, 'noise': 0.25},
        ]
        wavelengths, signals, noisy_signals, spectrum_values = generate_signal(
            wl0, wl1, step,
            pure_signal=pure_signal[:171],
            baseline=baseline_segments,
            num_samples=500,
            noise_std_dev=0.05,
            smooth_signal=True,
            smooth_sigma=1.75,
        )
        plot_signal(wavelengths, signals, noisy_signals, spectrum_values)
        all_spectra = torch.tensor(spectrum_values.tolist())#.unsqueeze(0) # shape: [1, 513]
        all_labels = [[eos_idx]]*int(all_spectra.shape[0])
        spectra_tensor = [arr.clone().detach().to(torch.float16) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float16) for arr in all_spectra]
        label_tensor = [lbl.clone().detach().to(torch.long) if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long) for lbl in all_labels]
        torch_outfile = os.path.join(c.PATH_DATA, "my_dataset_file.pt")
        torch.save((spectra_tensor, label_tensor), torch_outfile)
    elif c.TARGET == 'custom':
        wl0, wl1, step = 300, 2000, 10
        peaks = [
            {"anchor": 550,  "fwhm": 200,  "amplitude": 0.8},
        ]
        baseline_segments = [
            {"start": 300,  "end":  400, "value": 0.05, 'noise': 0.05},
            {"start": 400,  "end":  950, "value": 0.05, 'noise': 0.05},
            {"start": 950,  "end": 2000, "value": 0.05, 'noise': 0.05},
            # {"start": 300,  "end": 400, "value": 0.50},
            # {"start": 400, "end": 1200, "value": 0.93},
            # {"start": 1200, "end": 2000, "value": 0.07},
        ]
    
        w, s, ns, rat = generate_signal(
            wl0, wl1, step,
            num_samples=1000,
            peaks=peaks,
            baseline=baseline_segments,
            noise_std_dev=0.05,
            smooth_signal=True,
            smooth_sigma=1.5,
        )
        
        plot_signal(w, s, ns, rat)
    
        wavelengths, signals, noisy_signals, spectrum_values = w, s, ns, rat
        
        all_spectra = torch.tensor(spectrum_values.tolist()) #.unsqueeze(0) # shape: [1, 513]
        all_labels = [[eos_idx]]*int(spectrum_values.shape[0])
        spectra_tensor = [arr.clone().detach().to(torch.float16) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float16) for arr in all_spectra]
        label_tensor = [lbl.clone().detach().to(torch.long) if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long) for lbl in all_labels]
        torch_outfile = os.path.join(c.PATH_DATA, "my_dataset_interact.pt")
        torch.save((spectra_tensor, label_tensor), torch_outfile)
    elif c.TARGET == 'validation':
        pass
    
    ckpt = rf'{c.PATH_RUN}/model_epoch_{c.RESUME_EPOCH}.pth'

    optical_main._main(argv=["test", 
                             "--ckpt", str(ckpt), 
                             "--mode", "ray"], 
                       cfg=c)
    
    # tgt_tokens, pred_tokens, target_spectrum, pred_spectrum, accuracy, mae = per_sample_results[0]
    
    if c.TARGET == 'file':
        per_sample_results = load_JSONPICKLE(c.PATH_RUN, f'val_sample_results_{c.TARGET}_{c.RUN_NAME}_E{c.RESUME_EPOCH}')
        sorted_by_first = sorted(per_sample_results, key=lambda x: np.mean(np.absolute(pure_signal[10:40]-x['pred_spectrum'][10:40])))
        # sorted_by_first = sorted(per_sample_results, key=lambda x: np.mean(np.absolute(pure_signal[:]-x['pred_spectrum'][:])))
        for i in np.arange(0,5,1):
            plot_samples(c, 
                         sorted_by_first[i]['pred_spectrum'], 
                         pure_signal,
                         # sorted_by_first[i]['target_spectrum'], 
                         sorted_by_first[i]['pred_seq'], 
                         sorted_by_first[i]['target_seq'], 
                         sorted_by_first[i]['accuracy'], 
                         sorted_by_first[i]['mae'], 
                         i)
    elif c.TARGET == 'validation':
        per_sample_results = load_JSONPICKLE(c.PATH_RUN, f'val_sample_results_{c.TARGET}_{c.RUN_NAME}_E{c.RESUME_EPOCH}')
        plot_accuracy(c, [i['accuracy'] for i in per_sample_results])
        plot_mae(c, [i['mae'] for i in per_sample_results])
        mses = [np.mean(np.square(np.array(i['target_spectrum']) - np.array(i['pred_spectrum']))) for i in per_sample_results]
        plot_mse(c, mses)
        sorted_by_first = sorted(per_sample_results, key=lambda x: x['mae'])
        for i in np.arange(0,10e3,1e3):
            i = int(i)
            plot_samples(c, 
                         sorted_by_first[i]['pred_spectrum'], 
                         sorted_by_first[i]['target_spectrum'], 
                         sorted_by_first[i]['pred_seq'], 
                         sorted_by_first[i]['target_seq'], 
                         sorted_by_first[i]['accuracy'], 
                         sorted_by_first[i]['mae'], 
                         i)
        # model comp
        with open(c.PATH_RES_COMP, 'rb') as f:   
            per_example_results = pickle.load(f)
        # per_example_results = load_JSONPICKLE(c.PATH_RUN, f'per_sample_results_validation_411')
        comp_dict = {f'{c.RUN_NAME}_Epoch{c.RESUME_EPOCH}': [i['mae'] for i in per_sample_results], 'N67_GPT26_Epoch207': [i[3] for i in per_example_results]} #N67_GPT26
        plot_mae_comparison(c, comp_dict)
        comp_dict = {f'{c.RUN_NAME}_Epoch{c.RESUME_EPOCH}': mses, 'N67_GPT26_Epoch207': [np.mean(np.square(i[4] - i[5])) for i in per_example_results]}
        plot_mse_comparison(c, comp_dict)
        comp_dict = {f'{c.RUN_NAME}_Epoch{c.RESUME_EPOCH}': [i['accuracy'] for i in per_sample_results], 'N67_GPT26_Epoch207': [i[0] for i in per_example_results]}
        plot_acc_comparison(c, comp_dict)
    elif c.TARGET == 'custom':
        per_sample_results = load_JSONPICKLE(c.PATH_RUN, f'val_sample_results_{c.TARGET}_{c.RUN_NAME}_E{c.RESUME_EPOCH}')
        plot_mae(c, [i['mae'] for i in per_sample_results])
        mses = [np.mean(np.square(np.array(i['target_spectrum']) - np.array(i['pred_spectrum']))) for i in per_sample_results]
        plot_mse(c, mses)
        target_comp = np.mean([i['target_spectrum'] for i in per_sample_results], axis=0)
        sorted_by_first = sorted(per_sample_results, key=lambda x: x['mae'])
        for i in np.arange(0,5,1):
            plot_samples(c, 
                         sorted_by_first[i]['pred_spectrum'], 
                         target_comp, 
                         sorted_by_first[i]['pred_seq'], 
                         sorted_by_first[i]['target_seq'], 
                         sorted_by_first[i]['accuracy'], 
                         sorted_by_first[i]['mae'], 
                         i)
    
    # exclude solutions
    # sorted_by_noAlorAg = [i for i in sorted_by_first if bool(all([True if j.split('__')[0] not in ['Ag','Al','TiN'][0:] else False for j in i[1] or i[2]]))]
    # sorted_by_noAlorAg = [i for i in sorted_by_first if bool(any([True if j.split('__')[0] in ['Ag','Al','TiN'][0:] else False for j in i[1] or i[2]]))]
    # for i in np.arange(0,3,1):
    #     plot_samples(c, sorted_by_noAlorAg[i][5], sorted_by_noAlorAg[i][4], sorted_by_noAlorAg[i][1], sorted_by_noAlorAg[i][2], sorted_by_noAlorAg[i][0], sorted_by_noAlorAg[i][3], i)
    # plot_mse(c, [np.mean(np.square(i[4] - i[5])) for i in sorted_by_noAlorAg])
    # plot_mae(c, [i[3] for i in sorted_by_noAlorAg])

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()