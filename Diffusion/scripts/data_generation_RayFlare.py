# -*- coding: utf-8 -*-

import os
import sys
import config_RF as c
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import itertools
import random
import torch
from collections import Counter
import tempfile

from call_rayflare import Call_RayFlare_with_dict, DBinitNewMats

from utils import load_JSONPICKLE_NEW, save_JSONPICKLE_NEW, init_tokenmaps

from typing import Any, Dict, List, Mapping, Tuple, Union, Optional

def parse_token(token):
    parts = token.split('__')
    if len(parts) != 2:
        raise ValueError(f"Invalid token format: {token}")
    material, thickness = parts
    return material, int(thickness)

def combine_tokens(tokens):
    combined_tokens = []
    current_material = None
    current_thickness = 0

    for token in tokens:
        material, thickness = parse_token(token)
        if material == current_material:
            current_thickness += thickness
        else:
            if current_material is not None:
                combined_tokens.extend(split_token(current_material, current_thickness))
            current_material = material
            current_thickness = thickness

    if current_material is not None:
        combined_tokens.extend(split_token(current_material, current_thickness))

    return combined_tokens

def split_token(material, thickness):
    tokens = []
    while thickness > 500:
        tokens.append(f'{material}__500')
        thickness -= 500
    tokens.append(f'{material}__{thickness}')
    return tokens

def generate_unique_sequences(tokens, num_samples, max_combination_length, sample_file_path=None, sample_file_name=None):
    """
    Generate unique sequences from a list of tokens.

    Args:
        tokens (list): List of tokens to generate sequences from.
        num_samples (int): Number of unique sequences to generate.
        max_combination_length (int): Maximum length of a sequence.
        sample_file_path (str): Path to the directory containing the sample file.
        sample_file_name (str): Name of the sample file.

    Returns:
        list: List of unique sequences.
    """

    # Load existing sequences from file if it exists
    try:
        existing_sequences = load_JSONPICKLE_NEW(sample_file_path, sample_file_name)
    except FileNotFoundError:
        existing_sequences = []

    sequences = set(tuple(seq) for seq in existing_sequences)
    while len(sequences) < num_samples:
        sequence = tuple(random.choices(tokens, k=random.randint(1, max_combination_length)))
        combined_sequence = tuple(combine_tokens(sequence))
        sequences.add(combined_sequence)

    # Convert the set back to a list
    unique_sequences = [list(sequence) for sequence in sequences]

    # Save the updated sequences to file
    # save_JSONPICKLE_NEW(sample_file_path, unique_sequences, sample_file_name)

    return unique_sequences

#%% new main function
def main():
    from solcore import config
    config.user_folder = rf'{c.PATH}/RayFlare/data/Solcore'
    config['Parameters','custom'] = f'{config.user_folder}/custom_parameters.txt'
    c.THICKNESSES = np.arange(c.THICKNESS_MIN,c.THICKNESS_MAX+1,c.THICKNESS_STEPS)
    c.WAVELENGTHS = np.arange(c.WAVELENGTH_MIN,c.WAVELENGTH_MAX+1,c.WAVELENGTH_STEPS)
    c.MATERIAL_LIST = [item[:-4] for item in 
                               sorted(os.listdir(c.PATH_MATERIALS)) 
                               if not item.startswith('XX')]
    os.makedirs(c.PATH_RUN, exist_ok=True)
    save_JSONPICKLE_NEW(c.PATH_RUN, c, 'config_RF')

    # config.user_folder = rf'{c.PATH}/RayFlare/data/Solcore'
    # config['Parameters','custom'] = f'{config.user_folder}/custom_parameters.txt'
    
    DBinitNewMats(c)
    
    # Combine the two lists into one list
    tokens = [str(x) + '_' + str(y) for x, y in itertools.product(c.MATERIAL_LIST, c.THICKNESSES)]
    
    save_JSONPICKLE_NEW(c.PATH_RUN, tokens, 'tokens')
    # tokens = load_JSONPICKLE_NEW(c.PATH_RUN, 'tokens')
    
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = init_tokenmaps(c.PATH_DATA)

    
    # Stochastically sample combinations of length 20
    # samples = generate_unique_sequences(tokens, int(c.NUM_SAMPLES), c.MAX_SEQ_LEN, sample_file_path=c.PATH_RUN, sample_file_name='samples')

    # save_JSONPICKLE_NEW(c.PATH_RUN, samples, 'samples')
    # samples = load_JSONPICKLE_NEW(c.PATH_RUN, 'samples')
    
    # ma_data = pd.read_csv(r'd:\Profile\a3536\Eigene Dateien\GitHub\ColorAppearanceToolbox\Diffusion\data\ma\data_test.csv', sep=',')
    for datafile in range(0,10):
        print(datafile)
        ma_data = pd.read_csv(rf'd:\Profile\a3536\Eigene Dateien\GitHub\ColorAppearanceToolbox\Diffusion\data\ma\data_test.csv', sep=',')
        import ast
        samples = ma_data.loc[:, 'structure']
        samples = samples.apply(ast.literal_eval)
        samples = samples.to_list()
        save_JSONPICKLE_NEW(c.PATH_RUN, samples, f'samples_train_{datafile}')
        # save_JSONPICKLE_NEW(c.PATH_RUN, samples, f'samples_test')
        samples = load_JSONPICKLE_NEW(c.PATH_RUN, f'samples_train_{datafile}')
        # samples = load_JSONPICKLE_NEW(c.PATH_RUN, 'samples_test')
            
        # We’ll collect all results in a structure suitable for conversion to Parquet
        # or for PyTorch. Typically you want columns like "spectrum" and "label".
        all_spectra = []
        all_labels = []
        # You could store angles or other metadata as well; let’s store each angle’s R.
        print('starting simulations')
        
        # try:
        #     i_store = load_JSONPICKLE_NEW(c.PATH_RUN, 'i')
        #     torch_outfile = os.path.join(c.PATH_RUN, f"data_train_{datafile}.pt")
        #     # torch_outfile = os.path.join(c.PATH_RUN, "data_val.pt")
        #     all_spectra, all_label = torch.load(torch_outfile)
        # except:
        i_store = 0
        print(i_store)
        for i, stack in enumerate(samples):
            # if i < i_store:
            #     continue
            if i%1000 == 0:
                print(i)
                save_JSONPICKLE_NEW(c.PATH_RUN, i, 'i')
            if i%1e5 == 0 and i != 0:
                spectra_tensor = [arr.clone().detach().to(torch.float16) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float16) for arr in all_spectra]
                all_labels = [lbl.clone().detach().to(torch.long) if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long) for lbl in all_labels]
                # spectra_tensor = [torch.tensor(arr, dtype=torch.float16) for arr in all_spectra]
                # all_labels = [lbl.clone().detach().to(torch.long) for lbl in all_labels]
                torch_outfile = os.path.join(c.PATH_RUN, f"data_train_{datafile}.pt")
                # torch_outfile = os.path.join(c.PATH_RUN, f"samples_test.pt")
                torch.save((spectra_tensor, all_labels), torch_outfile)
                print(f"Torch dataset saved to: {torch_outfile}")
            
            # 1) Call RayFlare
            RAT_array = Call_RayFlare_with_dict(c, stack, EOS_TOKEN)
            
            all_spectra.append(RAT_array)
            all_labels.append([token_to_idx[d] for d in stack])  # or some numeric ID for stack
            # if i%999==0 and i != 0:
            #     break
        
        # Convert each array in all_spectra to a single 2D tensor
        # shape = (# of samples, # of wavelength points)
        # spectra_tensor = [arr.clone().detach().to(torch.float16) for arr in all_spectra]
        spectra_tensor = [arr.clone().detach().to(torch.float16) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float16) for arr in all_spectra]
        all_labels = [lbl.clone().detach().to(torch.long) if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long) for lbl in all_labels]
        torch_outfile = os.path.join(c.PATH_RUN, f"data_train_{datafile}.pt")
        # torch_outfile = os.path.join(c.PATH_RUN, f"samples_test.pt")
        # torch_outfile = os.path.join(c.PATH_RUN, "data_val.pt")
        torch.save((spectra_tensor, all_labels), torch_outfile)
        print(f"Torch dataset saved to: {torch_outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()

# spectra_tensor, all_labels = torch.load(torch_outfile)
