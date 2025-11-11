import matplotlib.pyplot as plt
import numpy as np
def plot_samples(c, RAT_pred, RAT_tar, stack_pred, stack_tar, ACC, number, RAT_tar_mean = None):
    def _strip(seq):
        return [t for t in seq if t not in ("<PAD>", "<SOS>", "<EOS>", "<MSK>")]
    stack_tar = _strip(stack_tar)
    stack_pred = _strip(stack_pred)
    colormap = plt.cm.tab20(range(6))
    wls = c.WAVELENGTHS
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
    plt.title(f'Prediction using OptoLLama [model: {c.RUN_NAME}]', fontsize=10)
    plt.legend(loc = 'upper right')
    plt.xlabel('Wavelengths [nm]', fontsize=10)
    plt.ylabel('Reflectance (R), Transmissance (T), Absorptance (A)', fontsize=10)
    
    # Automatically position the target and prediction text    
    text = '\n'.join(f'{material.split("_")[0]}' for material in stack_tar)
    text2 = '\n'.join(f'{material.split("_")[1]}' for material in stack_tar)
    plt.text(0.02, 1.1, f"Target:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -" , ha='left', fontsize=8, transform=plt.gca().transAxes)
    plt.text(0.24, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
        
    text = '\n'.join(f'{material.split("_")[0]}' for material in stack_pred)
    text2 = '\n'.join(f'{material.split("_")[1]}' for material in stack_pred)
    plt.text(0.76, 1.1, f"Prediction:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -", ha='left', fontsize=8, transform=plt.gca().transAxes)
    plt.text(0.98, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
    
    # Automatically position the accuracy, MAE, and key text
    spectrum_mae = np.mean(np.absolute(np.array(RAT_pred) - np.array(RAT_tar)))
    plt.text(0.36, 1.10, "- - - - - - - - - - - - - -\nMC samples:\nMAE:\nAccuracy:\n- - - - - - - - - - - - - -", ha='left', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.64, 1.10, f"\n{number}\n{spectrum_mae:.4f}\n{ACC:.4f}\n", ha='right', fontsize=10, transform=plt.gca().transAxes)
    
    plt.show()