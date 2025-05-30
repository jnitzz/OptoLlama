# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:10:49 2025

@author: a3536
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracy(c, accuracies):
    # accuracies = [i[0] for i in samples]
    plt.figure(figsize=(8, 6))
    plt.hist(accuracies, bins=20, edgecolor='black')
    plt.title(f'Distribution of Per-Example Accuracy [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]')
    plt.xlabel('Accuracy score [20 bins]')
    plt.ylabel('Count')
    # Save the figure in the model directory
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_distribution_accuracy.svg')
    plt.savefig(hist_path)
    print(f"Accuracy distribution plot saved to: {hist_path}")

    # If running locally and you want to see the plot:
    plt.show()
    
def plot_mse(c, mses):
    # mses = [i[3] for i in samples]
    plt.figure(figsize=(8, 6))
    plt.hist(mses, bins=100,edgecolor='black')
    plt.title(f'Distribution of Mean Squared Error (MSE) [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]')#  [model: {model_folder}]')
    plt.xlabel('MSE [100 bins]')
    plt.ylabel('Count')
    plt.text(0.95, 0.95, f"mean MSE: {np.mean(mses):.3f}", ha='right', fontsize=10, transform=plt.gca().transAxes)
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_distribution_mse.svg')
    plt.savefig(hist_path)
    print(f"Accuracy distribution plot saved to: {hist_path}")
    plt.show()

def plot_mae(c, maes):
    # mses = [i[3] for i in samples]
    plt.figure(figsize=(8, 6))
    plt.hist(maes, bins=100,edgecolor='black')
    plt.title(f'Distribution of Mean Absolute Error (MAE) [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]')#  [model: {model_folder}]')
    plt.xlabel('MAE [100 bins]')
    plt.ylabel('Count')
    plt.text(0.95, 0.95, f"mean MAE: {np.mean(maes):.3f}", ha='right', fontsize=10, transform=plt.gca().transAxes)
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_distribution_mae.svg')
    plt.savefig(hist_path)
    print(f"Accuracy distribution plot saved to: {hist_path}")
    plt.show()
    
def plot_samples(c, RAT_pred, RAT_tar, stack_pred, stack_tar, ACC, MSE, number, RAT_tar_mean = None):
    def _strip(seq):
        return [t for t in seq if t not in ("<PAD>", "<SOS>", "<EOS>")]
    stack_tar = _strip(stack_tar)
    stack_pred = _strip(stack_pred)
    colormap = plt.cm.tab20(range(6))
    wls = np.arange(c.WAVELENGTH_MIN,c.WAVELENGTH_MAX+1,c.WAVELENGTH_STEPS)
    plt.plot(wls,RAT_pred[:171], label = 'Prediction (R)', color=colormap[1])
    plt.plot(wls,RAT_tar[:171], label = 'Target (R)', color=colormap[0])
    if RAT_tar_mean is not None:
        plt.plot(wls,RAT_tar_mean[:171], '--', label = 'Target mean(R)', color=colormap[0])
    plt.plot(wls,RAT_pred[171:2*171], label = 'Prediction (A)', color=colormap[3])
    plt.plot(wls,RAT_tar[171:2*171], label = 'Target (A)', color=colormap[2])
    if RAT_tar_mean is not None:
        plt.plot(wls,RAT_tar_mean[171:2*171], '--', label = 'Target mean(A)', color=colormap[2])
    plt.plot(wls,RAT_pred[2*171:], label = 'Prediction (T)', color=colormap[5])
    plt.plot(wls,RAT_tar[2*171:], label = 'Target (T)', color=colormap[4])
    if RAT_tar_mean is not None:
        plt.plot(wls,RAT_tar_mean[2*171:], '--', label = 'Target mean(T)', color=colormap[4])
    plt.title(f'Prediction using OptoLLama [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]', fontsize=10)
    plt.legend(loc = 'upper right')
    plt.xlabel('Wavelengths [nm]', fontsize=10)
    plt.ylabel('Reflectance (R), Transmissance (T), Absorptance (A)', fontsize=10)
    
    # Automatically position the target and prediction text    
    text = '\n'.join(f'{material.split("__")[0]}' for material in stack_tar)
    text2 = '\n'.join(f'{material.split("__")[1]}' for material in stack_tar)
    plt.text(0.02, 1.1, f"Target:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -" , ha='left', fontsize=8, transform=plt.gca().transAxes)
    plt.text(0.24, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
        
    text = '\n'.join(f'{material.split("__")[0]}' for material in stack_pred)
    text2 = '\n'.join(f'{material.split("__")[1]}' for material in stack_pred)
    plt.text(0.76, 1.1, f"Prediction:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -", ha='left', fontsize=8, transform=plt.gca().transAxes)
    plt.text(0.98, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
    
    # Automatically position the accuracy, MAE, and key text
    # spectrum_rss = np.sum(np.square(RAT_pred - RAT_tar))
    spectrum_mae = np.mean(np.absolute(RAT_pred - RAT_tar))
    plt.text(0.36, 1.10, "- - - - - - - - - - - - - -\nKey#:\nMAE:\nAccuracy:\n- - - - - - - - - - - - - -", ha='left', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.64, 1.10, f"\n{number}\n{spectrum_mae:.2f}\n{ACC:.2f}\n", ha='right', fontsize=10, transform=plt.gca().transAxes)
    
    plt.show()

def plot_mae_comparison(c, model_maes_dict):
    """
    Plot MAE distributions for multiple models.

    Parameters:
    - c: config object with RUN_NAME, RESUME_EPOCH, PATH_RUN
    - model_maes_dict: dict where keys are model names (str), values are lists or arrays of MAEs
    """
    plt.figure(figsize=(8, 6))
    max_vals = max([max(maes) for i, maes in model_maes_dict.items()])
    # Plot each model's MAE histogram
    for model_name, maes in model_maes_dict.items():
        plt.hist(maes, bins=np.linspace(0,max_vals,100), alpha=0.5, label=f'{model_name} (mean: {np.mean(maes):.3f})', edgecolor='black', linewidth=0.5)

    plt.title('MAE Distribution Comparison')
    plt.xlabel('MAE [100 bins]')
    plt.ylabel('Count')
    plt.legend(loc='upper right', fontsize=9)

    # Save and show
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_mae_comparison.svg')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=400)
    print(f"MAE comparison plot saved to: {hist_path}")
    plt.show()

def plot_mse_comparison(c, model_mses_dict):
    """
    Plot MSE distributions for multiple models.

    Parameters:
    - c: config object with RUN_NAME, RESUME_EPOCH, PATH_RUN
    - model_mses_dict: dict where keys are model names (str), values are lists or arrays of MSEs
    """
    plt.figure(figsize=(8, 6))
    max_vals = max([max(mses) for i, mses in model_mses_dict.items()])
    # Plot each model's MAE histogram
    for model_name, mses in model_mses_dict.items():
        plt.hist(mses, bins=np.linspace(0,max_vals,100), alpha=0.5, label=f'{model_name} (mean: {np.mean(mses):.3f})', edgecolor='black', linewidth=0.5)

    plt.title('MSE Distribution Comparison')
    plt.xlabel('MSE [100 bins]')
    plt.ylabel('Count')
    plt.legend(loc='upper right', fontsize=9)

    # Save and show
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_mse_comparison.svg')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=400)
    print(f"MSE comparison plot saved to: {hist_path}")
    plt.show()

def plot_acc_comparison(c, model_accs_dict):
    """
    Plot ACC distributions for multiple models.

    Parameters:
    - c: config object with RUN_NAME, RESUME_EPOCH, PATH_RUN
    - model_accs_dict: dict where keys are model names (str), values are lists or arrays of ACCs
    """
    plt.figure(figsize=(8, 6))
    max_vals = max([max(accs) for i, accs in model_accs_dict.items()])
    # Plot each model's MAE histogram
    for model_name, accs in model_accs_dict.items():
        plt.hist(accs, bins=np.linspace(0,max_vals,100), alpha=0.5, label=f'{model_name} (mean: {np.mean(accs):.3f})', edgecolor='black', linewidth=0.5)

    plt.title('Accuracy Distribution Comparison')
    plt.xlabel('Accuracy [100 bins]')
    plt.ylabel('Count')
    plt.legend(loc='upper right', fontsize=9)

    # Save and show
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_acc_comparison.svg')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=400)
    print(f"ACC comparison plot saved to: {hist_path}")
    plt.show()