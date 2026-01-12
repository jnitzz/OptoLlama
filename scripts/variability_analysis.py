from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import zip_longest
import seaborn as sns
import pandas as pd
import json
import argparse

def load_stack_all_maes(path):
    with open(path, "r") as f:
        data = json.load(f)

    # If the file contains a single object
    if isinstance(data, dict):
        return data.get("stack_all_maes", [])

    # If the file contains a list of objects
    maes = []
    counter = -1
    for obj in data:
        counter += 1
        if counter in [1691, 1879, 2067, 2255, 2443, 2631, 2819, 3007]:
            continue
        if "stack_all_maes" in obj:
            maes.append(obj["stack_all_maes"])
    return maes


def parse_tokens(data):
    '''
    Creates a tuple of material (string) and thickness (float) out of a token.
    '''
    new_data = []
    for stack in data:
        new_stack = []
        for token in stack:
            a, b = token.split("_")
            parsed = (a, float(b))
            new_stack.append(parsed)
        new_data.append(new_stack)
    return new_data



def heat_map(data, target, counter, result_path):
    file_name_heat = "heat_map_" + str(counter) + ".png"
    result_path_heat = os.path.join(result_path, file_name_heat)
    file_name_table = "table_" + str(counter) + ".png"
    result_path_table = os.path.join(result_path, file_name_table)
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


def scatter_plot(predictions, counter, result_path):
    file_name_scatter_plot = "scatter_plot_" + str(counter) + ".png"
    result_path_scatter_plot = os.path.join(result_path, file_name_scatter_plot)

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
    
def best_mae(mae, index, result_path):
    file_name_mae_plot = "MAE_plot_" + str(index) + ".png"
    result_path_mae_plot = os.path.join(result_path, file_name_mae_plot)

    losses = np.asarray(mae)
    best_losses = np.minimum.accumulate(losses)  # running minimum

    plt.figure(figsize=(8, 5))
    plt.plot(best_losses, label="Best MAE so far", linewidth=2)
    plt.xlabel("Drawn Sample")
    plt.ylabel("Best (lowest) MAE")
    plt.title("Evolution of Best MAE Over Samples")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(result_path_mae_plot)
    plt.close()

def collective_best_mae(maes, result_path):
    #print(maes)
    result_path_mae_plot = os.path.join(result_path, "MAE_plot_full.png")
    result_path_relative_mae_plot = os.path.join(result_path, "relative_mae.png")
    result_path_box_plot = os.path.join(result_path, "box_plot.png")
    result_path_violin_plot = os.path.join(result_path, "violin_plot.png")

    losses = np.asarray(maes)  # shape: (runs, test_samples)

    # Aggregate across test set
    best_per_sample = np.minimum.accumulate(losses, axis=1)

    mean = np.mean(best_per_sample, axis=0)
    median_curve = np.median(best_per_sample, axis=0)
    q10 = np.percentile(best_per_sample, 10, axis=0)
    q25 = np.percentile(best_per_sample, 25, axis=0)
    q75 = np.percentile(best_per_sample, 75, axis=0)
    q90 = np.percentile(best_per_sample, 90, axis=0)
    worst_best_mae = np.max(best_per_sample, axis=0)
    best_best_mae = np.min(best_per_sample, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(median_curve, label="Median best MAE")
    plt.plot(mean, label="Mean best MAE")
    plt.fill_between(range(len(median_curve)), q25, q75, alpha=0.5, label="25–75% band")
    plt.fill_between(range(len(median_curve)), q10, q90, alpha=0.2, label="10–90% band")
    #plt.plot(worst_best_mae, label="Worst best MAE", color="red")
    #plt.plot(best_best_mae, label="Best best MAE", color="green")
    plt.xlabel("Drawn Sample")
    plt.ylabel("Best MAE")
    plt.title("Evolution of Best MAE Over Stochastic Draws")
    plt.yscale("log") 
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(result_path_mae_plot, bbox_inches="tight")
    plt.close()

    prev = best_per_sample[:, :-1]
    curr = best_per_sample[:, 1:]
    relative = (prev - curr) / prev  # step-wise relative improvement
    relative = np.clip(relative, 0, None)  # avoid tiny negatives due to numerical errors

    runs = np.arange(1, best_per_sample.shape[1])  # run indices for x-axis

    median_curve = np.median(relative, axis=0)
    mean_curve = np.mean(relative, axis=0)
    q10 = np.percentile(relative, 10, axis=0)
    q25 = np.percentile(relative, 25, axis=0)
    q75 = np.percentile(relative, 75, axis=0)
    q90 = np.percentile(relative, 90, axis=0)

    print(q90)
    print(q10)

    plt.figure(figsize=(10, 5))
    plt.plot(runs, median_curve, label="Median Relative Improvement")
    plt.plot(runs, mean_curve, label="Mean Relative Improvement")
    plt.fill_between(runs, q25, q75, alpha=0.5, label="25–75% band")
    plt.fill_between(runs, q10, q90, alpha=0.2, label="10–90% band")

    plt.xlabel("Run number")
    plt.ylabel("Step-wise relative improvement")
    plt.title("Relative Improvement of Best MAE per Step")
    #plt.yscale("log") 
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.savefig(result_path_relative_mae_plot, bbox_inches="tight")
    plt.close()



    run_indices = [0, 4, 9, 19, 29, 39]   # runs 1, 5, 10, 20
    labels = ["Run 1", "Run 5", "Run 10", "Run 20", "Run 30", "Run 40"]
    data = [best_per_sample[:, k] for k in run_indices]
    df = pd.DataFrame({label: best_per_sample[:, k] for label, k in zip(labels, run_indices)})
    df.to_csv("boxplot_data.csv", index=False)

    plt.figure(figsize=(8, 5))

    plt.boxplot(data, tick_labels=labels, showfliers=False)

    plt.xlabel("Run number")
    plt.ylabel("Best-so-far loss")
    plt.title("Distribution of Best Loss Across Runs (Box Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Optional if losses vary a lot:
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(result_path_box_plot)
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.violinplot(data, showmeans=True, showmedians=False)

    plt.xticks(range(1, len(labels)+1), labels)

    plt.xlabel("Run number")
    plt.ylabel("Best-so-far loss")
    plt.title("Distribution of Best Loss Across Runs (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.yscale("log")   # optional
    plt.tight_layout()
    plt.savefig(result_path_violin_plot)
    plt.close()



def plots(predictions, target, mae, index):
    result_path = "./plots/"

    predictions = parse_tokens(predictions)
    heat_map(predictions, target, index, result_path)
    scatter_plot(predictions, index, result_path)
    best_mae(mae, index, result_path)

if __name__ == "__main__":
    path = "/p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/runs/OptoLlama/results_OptoLlama_valid.json"

    maes = load_stack_all_maes(path)
    collective_best_mae(maes,"/p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/plots")
