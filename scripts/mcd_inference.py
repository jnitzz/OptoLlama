from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config_EVAL as cfg
import os

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


def heat_map(data, counter):
    file_name = "heat_map_" + str(counter) + ".png"
    result_path = os.path.join(cfg.PATH_RUN, file_name)
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
    plt.savefig(result_path)



def aggregation(predictions, plot):
    counter = 0
    transposed = list(map(list, zip(*predictions))) # transpose list so that the batch dimension is on the outer side
    aggregated = [] # new prediction list
    for item in transposed: # for each item in batch
        parsed_samples = []
        for sample in item: # for each sample drawn
            parsed_stack = [] # create a new prediction
            for layer in sample: # for each predicted layer
                parsed_stack.append(parse_tokens(layer)) # parse the tokens of the layer and add it to the new prediction
            parsed_samples.append(parsed_stack)
        if plot:
            heat_map(parsed_samples, counter)
            counter += 1
        aggregated_samples = aggregate_predictions(parsed_samples)
        aggregated.append(aggregated_samples)
    final = reconstruct_tokens(aggregated)
    return final

    