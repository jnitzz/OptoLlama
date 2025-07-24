from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config_EVAL as cfg
import os
from tabulate import tabulate
from itertools import zip_longest
import seaborn as sns
import pandas as pd

def parse_tokens(token):
    a, b = token.split("__")
    parsed = (a, float(b))
    return parsed
def reconstruct_tokens(data):
    reconstructed = [
        [f"{a}__{b}" for a, b in sublist]
        for sublist in data
    ]
    return reconstructed
    

def aggregate_predictions(predictions):
    max_length = max(len(sublist) for sublist in predictions)
    result = []

    for i in range(max_length):
        # Collect strings and floats at position i
        freq = Counter()
        value_map = defaultdict(list)

        for sublist in predictions:
            if i < len(sublist):
                s, f = sublist[i]
                freq[s] += 1
                value_map[s].append(f)

        # Resolve most frequent string with alphabetical tie-breaker
        max_count = max(freq.values())
        candidates = [s for s, count in freq.items() if count == max_count]
        selected_string = min(candidates)

        # Average the floats for the selected string
        avg_float = sum(value_map[selected_string]) / len(value_map[selected_string])
        result.append((selected_string, avg_float))
    
    # Use the closest prediction to the aggregated prediction for forecasting
    '''
    closest = None
    min_dist = -1
    for prediction in predictions:
        dist = pairwise_distance(prediction,result)
        if closest == None:
            closest = prediction
            min_dist = dist
        elif dist < min_dist:
            min_dist = dist
            closest = prediction
    '''

    return result

def pairwise_distance(s1, s2):
    min_length = min(len(s1), len(s2))
    max_length = max(len(s1), len(s2))

    dist = max_length - min_length

    for layer in range(min_length):
        mat1, thick1 = s1[layer]
        mat2, thick2 = s2[layer]
        if mat1 != mat2:
            dist += 1
        else:
            dist += abs(thick1 - thick2) / max(thick1,thick2)
    
    return dist

def variance(predictions):
    agg_preds = aggregate_predictions(predictions)
    sum = 0
    for pred in predictions:
        dist = pairwise_distance(pred, agg_preds)
        squared_dist = dist * dist
        sum += squared_dist
    variance = sum/len(predictions)
    return variance


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


def box_plot(predictions, counter):
    file_name_box_plot = "box_plot_" + str(counter) + ".png"
    result_path_box_plot = os.path.join(cfg.PATH_RUN, file_name_box_plot)

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
    plt.savefig(result_path_box_plot)
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
        box_plot(parsed_samples, i)


def aggregation(predictions):
    transposed = list(map(list, zip(*predictions))) # transpose list so that the batch dimension is on the outer side
    aggregated = [] # new prediction list
    for item in transposed: # for each item in batch
        parsed_samples = []
        for sample in item: # for each sample drawn
            parsed_stack = [] # create a new prediction
            for layer in sample: # for each predicted layer
                parsed_stack.append(parse_tokens(layer)) # parse the tokens of the layer and add it to the new prediction
            parsed_samples.append(parsed_stack)
        aggregated_samples = aggregate_predictions(parsed_samples)
        aggregated.append(aggregated_samples)
    final = reconstruct_tokens(aggregated)
    return final

    