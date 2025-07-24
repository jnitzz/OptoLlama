from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import config_EVAL as cfg
import os
from itertools import zip_longest
import seaborn as sns
import pandas as pd
from mcd_inference import parse_tokens

def heat_map(data, target, counter):
    file_name_heat = "heat_map_" + str(counter) + ".png"
    result_path_heat = os.path.join(cfg.PATH_RUN, file_name_heat)
    file_name_table = "table_" + str(counter) + ".png"
    result_path_table = os.path.join(cfg.PATH_RUN, file_name_table)
    max_len = max(len(sublist) for sublist in data)
    #data = [sublist + [("None", 0)] * (max_len - len(sublist)) for sublist in data]
    all_mats = set()
    for sublist in data:
        for item in sublist:
            all_mats.add(item[0])
    all_mats = sorted(all_mats)

    frequency = {s: [0] * max_len for s in all_mats}

    for sublist in data:
        for i, (s, _) in enumerate(sublist):
            frequency[s][i] += 1

    heatmap_data = np.array([frequency[s] for s in all_mats])

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, xticklabels=[f"Pos {i}" for i in range(max_len)],
                yticklabels=all_mats, cmap="YlGnBu", cbar_kws={'label': 'Count'})
    plt.title("Frequency of Materials at Each Layer")
    plt.xlabel("Layer")
    plt.ylabel("Material")
    plt.tight_layout()
    plt.savefig(result_path_heat)
    plt.clf()
    plt.close()

    ordered_predictions = []
    transposed = list(map(list, zip_longest(*data, fillvalue="")))

    for i in range(len(transposed)):
        freq_counter = dict()
        for item in transposed[i]:
            if item == "":
                continue
            if item[0] in freq_counter:
                freq_counter[item[0]] += 1
            else:
                freq_counter[item[0]] = 1
        sorted_frequency = sorted(freq_counter, key=lambda k: (-freq_counter[k], k))
        ordered_predictions.append(sorted_frequency)
    
    transposed_predictions = list(map(list, zip_longest(*ordered_predictions, fillvalue="")))

    for i in range(len(transposed_predictions)):
        while len(transposed_predictions[i]) < max_len:
            transposed_predictions[i].append("")
        transposed_predictions[i].insert(0, str("Prediction (Top " + str(i+1) + ")"))

    target.insert(0, "Target")
    transposed_predictions.insert(0, target)

    max_cols = max(len(row) for row in transposed_predictions)
    for row in transposed_predictions:
        while len(row) < max_cols:
            row.append("")

    # Create the plot
    fig, ax = plt.subplots()
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=transposed_predictions,
        cellLoc='center',
        loc='center',
        colWidths=[0.2]*max_cols
    )

    #print(transposed_predictions)
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        value = transposed_predictions[row_idx][col_idx]
        if row_idx == 0:
            # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif value == "":
            # Empty cells
            cell.set_facecolor('#ffe6e6')  # light red
        else:
            # Regular data cells
            cell.set_facecolor('#f1f1f2')  # light gray

    plt.savefig(result_path_table, bbox_inches='tight')
    plt.close()


def scatter_plot(predictions, counter):
    file_name_scatter_plot = "scatter_plot_" + str(counter) + ".png"
    result_path_scatter_plot = os.path.join(cfg.PATH_RUN, file_name_scatter_plot)

    layer_material_thickness = defaultdict(list)
    for pred in predictions:
        for layer_idx, (mat, thick) in enumerate(pred):
            layer_material_thickness[(layer_idx, mat)].append(thick)

    data = []
    for (layer, mat), thicknesses in layer_material_thickness.items():
        for t in thicknesses:
            data.append({'Layer': f'Layer {layer+1}', 'Material': mat, 'Thickness': t})
    df = pd.DataFrame(data)

    # Plot
    palette = sns.color_palette('tab20')
    plt.figure(figsize=(8, 6))
    ax = sns.stripplot(data=df, x='Layer', y='Thickness', hue='Material',
                    dodge=True, jitter=True, size=6, alpha=0.8, palette=palette)

    # Move legend outside
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title='Material', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title('Thickness Predictions per Layer and Material')

    num_layers = df['Layer'].nunique()
    for x in range(num_layers - 1):
        plt.axvline(x + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(result_path_scatter_plot)
    plt.close()
    

def plots(predictions, target):
    transposed = list(map(list, zip(*predictions))) # transpose list so that the batch dimension is on the outer side
    for i in range(len(transposed)): # for each item in batch
        parsed_samples = []
        parsed_target = []
        for sample in transposed[i]: # for each sample drawn
            parsed_stack = [] # create a new prediction
            for layer in sample: # for each predicted layer
                parsed_stack.append(parse_tokens(layer)) # parse the tokens of the layer and add it to the new prediction
            parsed_samples.append(parsed_stack)
        for tgt_layer in target[i]:
            if tgt_layer == '<EOS>':
                continue
            token = parse_tokens(tgt_layer)
            parsed_target.append(token[0])
        heat_map(parsed_samples, parsed_target, i)
        scatter_plot(parsed_samples, i)
