import os
import random
import datetime
import numpy as np
import pickle
import torch
import torch.distributed as dist

from utils import seed_everything, load_JSONPICKLE, save_JSONPICKLE, generate_signal, plot_signal
from plots import plot_accuracy, plot_mse, plot_mae, plot_samples, plot_samples2, MemmapSpectra, plot_mae_comparison, plot_mse_comparison, plot_acc_comparison

from call_rayflare import DBinitNewMats, Call_RayFlare_with_dict
import config_OL19 as c

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
    for peek in [1]:
    # for peek in range(360,830,10):
        if c.TARGET == 'custom':
            # SIGNAL FILE
            # target = {}
            # from numpy import genfromtxt
            # file_path = os.path.join(c.PATH_DATA, 'filter_10-2000-10_R.csv')
            # target['R'] = genfromtxt(file_path, delimiter=',')
            # file_path = os.path.join(c.PATH_DATA, 'filter_10-2000-10_T.csv')
            # target['T'] = genfromtxt(file_path, delimiter=',')
            # target['A'] = np.array([1-(float(a)+float(b)) for a,b in zip(target['R'],target['T'])]) #1 - (target['R'] + target['T'])
            # pure_signal = np.concatenate([target['R'], target['A'], target['T']])
            
            # SIGNAL RAYFLARE
            # stack_str_list = ['ARC_ISE__100','soda-lime_0-01__2000','9327_SiN_fit_formatted__142.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__203.0', '9333_TiO2_fit_formatted__164.5', '9327_SiN_fit_formatted__90.7', 'ITO__110.0','3Hal_LZ__550.0', 'Ag__100.0', '<EOS>']
            # stack_str_list = ['ARC_ISE__100','soda-lime_0-01__2000','9327_SiN_fit_formatted__142.0', 'TiO2__165.5', '9327_SiN_fit_formatted__203.0', 'TiO2__164.5', '9327_SiN_fit_formatted__90.7', 'ITO__110.0', '3Hal_LZ__550.0', 'C60_HZB__23', 'BCP__8.0','Ag__100.0', '<EOS>']
            # stack_str_list = ['ARC_ISE__100','soda-lime_0-01__2000','9327_SiN_fit_formatted__142.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__203.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__90.7', 'ITO__110.0', '<EOS>']
            # stack_str_list = ['ARC_ISE__100','soda-lime_0-01__2000','9327_SiN_fit_formatted__142.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__203.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__90.7', '<EOS>']
            # stack_str_list = ['9327_SiN_fit_formatted__142.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__203.0', '9333_TiO2_fit_formatted__165.5', '9327_SiN_fit_formatted__90.7', '<EOS>']
            # stack_str_list = []
            # pure_signal = Call_RayFlare_with_dict(c, stack_str_list)
            
            # SIGNAL CMF XYZ
            # xyz_bar = np.array(np.genfromtxt('CIE_xyz_1931_2deg.csv', delimiter=','))
            # pre = np.zeros((60,4))
            # pre[:,0] = np.arange(300, 360,1)
            # post = np.zeros((1170,4))
            # post[:,0] = np.arange(831, 2001,1)
            # pure_signals = np.concatenate([pre,xyz_bar,post])
            # pure_signal_R = pure_signals[::10,[3]].flatten()
            # pure_signal = np.clip(np.concatenate([pure_signal_R,np.zeros(171),1-pure_signal_R]), 0.0, 1.0)
            
            # SIGNAL PEAKS
            # peaks = [
            #     {"anchor": peek,  "fwhm": 120,  "amplitude": 0.7},
            # ]
            
            wl0, wl1, step = 300, 2000, 10
            baseline_segments = [
                {"start": 300,  "end":  600, "value": 0.0, 'noise': 0.0},
                {"start": 600,  "end":  900, "value": 1.0, 'noise': 0.0},
                {"start": 900,  "end": 2000, "value": 0.0, 'noise': 0.0},
                
                # {"start": 300,  "end":  500, "value": 0.0, 'noise': 0.0},
                # {"start": 500,  "end":  600, "value": 1.0, 'noise': 0.0},
                # {"start": 600,  "end": 800, "value": 0.0, 'noise': 0.0},
                # {"start": 800,  "end": 1000, "value": 1.0, 'noise': 0.0},
                # {"start": 1000,  "end": 2000, "value": 0.0, 'noise': 0.0},
            ]
            
            wavelengths, signals, noisy_signals, spectrum_values = generate_signal(
                wl0, wl1, step,
                # pure_signal=pure_signal[:171],
                # peaks=peaks,
                baseline=baseline_segments,
                num_samples=20,
                noise_std_dev=0.10,
                smooth_signal=True,
                smooth_sigma=0.1,
            )
            # plot_signal(wavelengths, signals, noisy_signals, spectrum_values)
            all_spectra = torch.tensor(spectrum_values.tolist())#.unsqueeze(0) # shape: [1, 513]
            all_labels = [[eos_idx]]*int(all_spectra.shape[0])
            spectra_tensor = [arr.clone().detach().to(torch.float16) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float16) for arr in all_spectra]
            label_tensor = [lbl.clone().detach().to(torch.long) if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long) for lbl in all_labels]
            torch_outfile = os.path.join(c.PATH_DATA, "my_dataset_interact.pt")
            torch.save((spectra_tensor, label_tensor), torch_outfile)
        elif c.TARGET == 'test':
            pass
        
        ckpt = rf'{c.PATH_RUN}/model_epoch_{c.RESUME_EPOCH}.pth'
    
        optical_main._main(argv=["test", 
                                 "--ckpt", str(ckpt), 
                                 "--mode", "ray"], 
                           cfg=c)
        
        # tgt_tokens, pred_tokens, target_spectrum, pred_spectrum, accuracy, mae = per_sample_results[0]
        per_sample_results = load_JSONPICKLE(c.PATH_RUN, f'val_sample_results_{c.TARGET}_{c.RUN_NAME}_E{c.RESUME_EPOCH}')
        # plot_mae(c, [i['mae'] for i in per_sample_results])
        # plot_accuracy(c, [i['accuracy'] for i in per_sample_results])
        # mses = [np.mean(np.square(np.array(i['target_spectrum']) - np.array(i['pred_spectrum']))) for i in per_sample_results]
        # plot_mse(c, mses)
        # model comp
        # with open(c.PATH_RES_COMP, 'rb') as f:   
        #     per_example_results = pickle.load(f)
        # # per_example_results = load_JSONPICKLE(c.PATH_RUN, f'per_sample_results_validation_411')
        # comp_dict = {f'{c.RUN_NAME}_Epoch{c.RESUME_EPOCH}': [i['mae'] for i in per_sample_results], 'N67_GPT26_Epoch207': [i[3] for i in per_example_results]} #N67_GPT26
        # plot_mae_comparison(c, comp_dict)
        # comp_dict = {f'{c.RUN_NAME}_Epoch{c.RESUME_EPOCH}': mses, 'N67_GPT26_Epoch207': [np.mean(np.square(i[4] - i[5])) for i in per_example_results]}
        # plot_mse_comparison(c, comp_dict)
        # comp_dict = {f'{c.RUN_NAME}_Epoch{c.RESUME_EPOCH}': [i['accuracy'] for i in per_sample_results], 'N67_GPT26_Epoch207': [i[0] for i in per_example_results]}
        # plot_acc_comparison(c, comp_dict)
        
        target = np.concatenate([np.zeros(30),np.ones(30),np.zeros(111),np.zeros(171),np.ones(30),np.zeros(30),np.ones(111)])
        
        sorted_by_first = sorted(per_sample_results, key=lambda x: x['mae'])
        # sorted_by_first = sorted(per_sample_results, key=lambda x: np.mean(np.absolute(np.mean(spectrum_values,axis=0)[6:44]-x['pred_spectrum'][6:44])))
        # sorted_by_first = sorted(per_sample_results, key=lambda i: np.mean(np.absolute(np.concatenate([target[10:81],target[171*1+10:171*1+81],target[171*2+10:171*2+81]])
        #                             -np.concatenate([i['pred_spectrum'][10:81],i['pred_spectrum'][171*1+10:171*1+81],i['pred_spectrum'][171*2+10:171*2+81]]))))
        # sorted_by_first = sorted(per_sample_results, key=lambda i: np.mean(np.absolute(np.concatenate([target[10:81],target[171*2+10:171*2+81]])
        #                             -np.concatenate([i['pred_spectrum'][10:81],i['pred_spectrum'][171*2+10:171*2+81]]))))
        # sorted_by_first = sorted(per_sample_results, key=lambda x: np.mean(np.absolute(pure_signal[:]-x['pred_spectrum'][:])))
        
        # mses = [np.mean(np.absolute(np.concatenate([target[10:81],target[171*1+10:171*1+81],target[171*2+10:171*2+81]])
        #                             -np.concatenate([i['pred_spectrum'][10:81],i['pred_spectrum'][171*1+10:171*1+81],i['pred_spectrum'][171*2+10:171*2+81]]))) for i in per_sample_results]
        ds_train = MemmapSpectra(rf"{c.PATH_DATA}/my_dataset_16m.npy")
        i=0
        # for i in np.arange(0,2000,1000):
        #     i = int(i)
        #     plot_samples2(c, 
        #                   sorted_by_first[i]['pred_spectrum'], 
        #                   sorted_by_first[i]['target_spectrum'], 
        #                   sorted_by_first[i]['pred_seq'], 
        #                   sorted_by_first[i]['target_seq'], 
        #                   sorted_by_first[i]['accuracy'], 
        #                   sorted_by_first[i]['mae'], 
        #                   i,
        #                   ds_search=rf"{c.PATH_DATA}/my_dataset_16m.npy")
            
        if c.TARGET == 'custom':
            for i in np.arange(0,1,1):
                plot_samples(c, 
                             sorted_by_first[i]['pred_spectrum'], 
                             # np.mean(spectrum_values,axis=0), 
                             target,
                             sorted_by_first[i]['pred_seq'], 
                             sorted_by_first[i]['target_seq'], 
                             sorted_by_first[i]['accuracy'], 
                             sorted_by_first[i]['mae'], 
                             i)
        elif c.TARGET == 'test':
            for i in np.arange(0,10,1):
                i = int(i)
                plot_samples(c, 
                             sorted_by_first[i]['pred_spectrum'], 
                             sorted_by_first[i]['target_spectrum'], 
                             sorted_by_first[i]['pred_seq'], 
                             sorted_by_first[i]['target_seq'], 
                             sorted_by_first[i]['accuracy'], 
                             sorted_by_first[i]['mae'], 
                             i)
        
        # exclude solutions
        # sorted_by_noAlorAg = [i for i in sorted_by_first if bool(all([True if j.split('__')[0] not in ['Ag','Al','TiN'][0:] else False for j in np.concatenate([i['pred_seq'],i['target_seq']])]))]
        # sorted_by_noAlorAg = [i for i in sorted_by_noAlorAg if len(i['pred_seq']) > 5]
        # sorted_by_noAlorAg = [i for i in sorted_by_first if len(i['pred_seq']) > 5]
        # sorted_by_noAlorAg = [i for i in sorted_by_first if bool(any([True if j.split('__')[0] in ['Ag','Al','TiN'][0:] else False for j in np.concatenate([i['pred_seq'],i['target_seq']])]))]
        # for i in np.arange(0,10,1):
        #     i = int(i)
        #     plot_samples(c, 
        #                  sorted_by_noAlorAg[i]['pred_spectrum'], 
        #                  sorted_by_noAlorAg[i]['target_spectrum'], 
        #                  sorted_by_noAlorAg[i]['pred_seq'], 
        #                  sorted_by_noAlorAg[i]['target_seq'], 
        #                  sorted_by_noAlorAg[i]['accuracy'], 
        #                  sorted_by_noAlorAg[i]['mae'], 
        #                  i)
        # # plot_mse(c, [np.mean(np.square(i[4] - i[5])) for i in sorted_by_noAlorAg])
        # plot_mae(c, [i['mae'] for i in sorted_by_noAlorAg])

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
