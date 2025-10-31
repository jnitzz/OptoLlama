# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:57:28 2025

@author: a3536
"""

import os
from rayflare.options import default_options
import solcore
from solcore import material
from solcore import si
from solcore.structure import Layer
from rayflare.transfer_matrix_method import tmm_structure
from solcore.absorption_calculator import download_db
from solcore.material_system import create_new_material
from scipy.interpolate import CubicSpline, interp1d
from functools import lru_cache
import numpy as np
# import config_RAYFLARE as c

def load_materialdata_file(filename):
    """
    Reads a material data file where the line immediately above the first numeric data row
    indicates whether wavelength is in nm or um.

    Steps:
    1. Collect all lines except comments, empty lines, lines with references or keys (DATA:, etc.).
    2. Find the first line that parses as three floats -> data start.
       The line just above it is the 'header' line for nm/um detection.
    3. Parse numeric data from that line onward.
    4. Convert wavelength to meters based on whether header says 'nm' or 'um'.
    5. Return: (wavelengths_in_m, n_vals, k_vals, short_name)
    """
    short_name = os.path.splitext(os.path.basename(filename))[0]  # e.g. "Ag" or "ITO"
    
    ignore_keywords = ["REFERENCES:", "COMMENTS:", "DATA:"]
    valid_lines = []

    with open(filename, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                # Skip empty or comment lines
                continue
            if any(keyword in line.upper() for keyword in ignore_keywords):
                # Skip lines with references, comments, data block markers
                continue
            valid_lines.append(line)

    if not valid_lines:
        print(f"Warning: {filename} had no valid lines after filtering.")
        # Return empty arrays
        return np.array([]), np.array([]), np.array([]), short_name

    first_data_idx = None
    for i, line in enumerate(valid_lines):
        parts = line.split()
        if len(parts) == 3:
            try:
                w, n, k = map(float, parts)
                first_data_idx = i
                break
            except ValueError:
                pass

    if first_data_idx is None:
        print(f"Warning: {filename} had no parseable numeric data.")
        return np.array([]), np.array([]), np.array([]), short_name

    # Check line above data (header) for nm/um
    wavelength_unit = None
    if first_data_idx > 0:
        header_line = valid_lines[first_data_idx - 1].lower()
        if "um" in header_line or "μm" in header_line:
            wavelength_unit = "um"
        elif "nm" in header_line:
            wavelength_unit = "nm"

    if wavelength_unit is None:
        # Default to um or nm as you prefer:
        wavelength_unit = "um"
        print(f"Warning: {filename} - no explicit nm/um in header above data. Defaulting to {wavelength_unit}.")

    wavelengths = []
    n_vals = []
    k_vals = []

    # Parse numeric data
    for line in valid_lines[first_data_idx:]:
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            w, n, k = map(float, parts)
            wavelengths.append(w)
            n_vals.append(n)
            k_vals.append(k)
        except ValueError:
            pass

    wavelengths = np.array(wavelengths)
    n_vals = np.array(n_vals)
    k_vals = np.array(k_vals)

    # Convert to meters
    if wavelength_unit == "nm":
        wavelengths_si = wavelengths * 1e-9
    else:  # 'um'
        wavelengths_si = wavelengths * 1e-6

    return wavelengths_si, n_vals, k_vals, short_name

def DBinitNewMats(c):
    if c.DB_PLOTNK == False:
        return
    if c.DB_DOWNLOAD == True:
        download_db(confirm=True)
    else:
        pass
    
    # Example: gather files from a folder (excluding any that start with 'XX').
    material_files = [item for item in sorted(os.listdir(c.PATH_MATERIALS)) 
                      if not item.startswith('XX')]
    
    # We'll store each file's data in a list of tuples: (short_name, wavelengths_si, n, k).
    data_list = []
    for mat_file in material_files:
        full_path = os.path.join(c.PATH_MATERIALS, mat_file)
        if not os.path.isfile(full_path):
            print(f"File not found: {full_path}")
            continue

        wavelengths_si, n_vals, k_vals, short_name = load_materialdata_file(full_path)
        xnew = np.linspace(c.WAVELENGTH_MIN* 1e-9, c.WAVELENGTH_MAX* 1e-9, 171)
        # clamped_n = CubicSpline(wavelengths_si, n_vals, bc_type='not-a-knot')
        # clamped_k = CubicSpline(wavelengths_si, k_vals, bc_type='not-a-knot')
        clamped_n = interp1d(wavelengths_si, n_vals, axis=0, bounds_error=False, kind='linear', fill_value=(n_vals[0], n_vals[-1]))
        clamped_k = interp1d(wavelengths_si, k_vals, axis=0, bounds_error=False, kind='linear', fill_value=(k_vals[0], k_vals[-1]))
        n_vals_new = clamped_n(xnew)
        k_vals_new = clamped_k(xnew)
        wavelengths_si, n_vals, k_vals = xnew, n_vals_new, k_vals_new
        np.savetxt("temp_n.txt",np.array([wavelengths_si,n_vals]).T)            #TODO tempfile
        np.savetxt("temp_k.txt",np.array([wavelengths_si,k_vals]).T)
        create_new_material(mat_name = f'{short_name}', n_source='temp_n.txt', k_source='temp_k.txt', overwrite=True)
        os.remove('temp_n.txt')
        os.remove('temp_k.txt')
        
        # Create database entries
        data_list.append((short_name, wavelengths_si, n_vals, k_vals))
    if bool(c.DB_PLOTNK):
        import matplotlib.pyplot as plt
        # Now define the wavelength limits in nm, and convert to meters.
        plot_min_m = c.WAVELENGTH_MIN * 1e-9
        plot_max_m = c.WAVELENGTH_MAX * 1e-9
        colormap = plt.cm.tab10(np.linspace(0,1,10))#np.linspace(0,1,20))
        # First plot: all n vs. wavelength
        plt.figure(figsize=(8,6), dpi=400)
        i=0
        for (short_name, w_si, n_array, k_array) in data_list:
            # Optionally filter to the x-range
            in_range = (w_si >= plot_min_m) & (w_si <= plot_max_m)
            if not np.any(in_range):
                continue
            wl_plot = w_si[in_range]
            n_plot = n_array[in_range]
            if i < 10:
                plt.plot(wl_plot, n_plot, '-', label=short_name, color=colormap[i])
            elif i >= 10 and i < 20:
                plt.plot(wl_plot, n_plot, '--', label=short_name, color=colormap[i-10])
            elif i >= 20:
                plt.plot(wl_plot, n_plot, '-.', label=short_name, color=colormap[i-20])
            i +=1
    
        plt.xlim(plot_min_m, plot_max_m)
        plt.title("Refractive index (n) vs. Wavelength")
        plt.xlabel("Wavelength (m)")
        plt.ylabel("n")
        plt.legend()
        plt.savefig(f"{c.PATH_RUN}/refractive_index_data_n.pdf", dpi=600, bbox_inches='tight')
    
        # Second plot: all k vs. wavelength
        plt.figure(figsize=(8,6), dpi=400)
        i=0
        for (short_name, w_si, n_array, k_array) in data_list:
            # Optionally filter to the x-range
            in_range = (w_si >= plot_min_m) & (w_si <= plot_max_m)
            if not np.any(in_range):
                continue
            
            wl_plot = w_si[in_range]
            k_plot = k_array[in_range]
            if i < 10:
                plt.plot(wl_plot, k_plot, '-', label=short_name, color=colormap[i])
            elif i >= 10 and i < 20:
                plt.plot(wl_plot, k_plot, '--', label=short_name, color=colormap[i-10])
            elif i >= 20:
                plt.plot(wl_plot, k_plot, '-.', label=short_name, color=colormap[i-20])
            i +=1
    
        plt.xlim(plot_min_m, plot_max_m)
        plt.title("Extinction coefficient (k) vs. Wavelength")
        plt.xlabel("Wavelength (m)")
        plt.ylabel("k")
        plt.legend()
        plt.savefig(f"{c.PATH_RUN}/refractive_index_data_k.pdf", dpi=600, bbox_inches='tight')

def Call_RayFlare_with_dict(c, stack_str_list):
    """
    Takes a list of strings like ["Ag__100", "AlN__200", ...],
    builds the TMM structure, and returns the combined R/A/T
    as a np.array of shape [3*len(wavelengths)] in the order [R, A, T].
    """
    
    c.THICKNESSES = np.arange(c.THICKNESS_MIN,c.THICKNESS_MAX+1,c.THICKNESS_STEPS)
    c.WAVELENGTHS = np.arange(c.WAVELENGTH_MIN,c.WAVELENGTH_MAX+1,c.WAVELENGTH_STEPS)
    
    options = default_options()
    options.wavelength = c.WAVELENGTHS * 1e-9  # convert nm to meters
    options.parallel = True
    # options.bulk_profile = True
    # incidence and transmission media
    incidence = material('air')()
    transmission = material('EVA')()

    # Build the structure
    layer_list = []
    for item in stack_str_list:
        if "__" not in item:
            # skip or handle as needed
            continue
        mat_str, thick_str = item.split("__")
        if thick_str[0] == '-':
            thick_str = '0'
        # thickness in nm => convert to m with si()
        layer_list.append(Layer(si(thick_str + 'nm'), material(mat_str)()))

    struc = tmm_structure(layer_list, incidence=incidence, transmission=transmission,no_back_reflection=False)
    options.coherent = True
    options.theta_in = np.deg2rad(c.INCIDENCE_ANGLE)
    RAT = tmm_structure.calculate(struc, options, profile=True)
    # Flatten R, A, T
    R = RAT['R']
    A = RAT['A']
    T = RAT['T']
    out_arr = np.concatenate([R, A, T])
    return out_arr