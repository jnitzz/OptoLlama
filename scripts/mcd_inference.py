from collections import defaultdict, Counter
import numpy as np
import os

def parse_tokens(token):
    '''
    Creates a tuple of material (string) and thickness (float) out of a token.
    '''
    a, b = token.split("__")
    parsed = (a, float(b))
    return parsed

def reconstruct_tokens(data):
    '''
    Given a list of tuples, recretaes the tokens from which the tuples originated.
    '''
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
    
    '''
    # Use the closest prediction to the aggregated prediction for forecasting
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
            if max(thick1,thick2) == 0:
                continue
            dist += abs(thick1 - thick2) / max(thick1,thick2)
    
    return dist

def variance(predictions):
    agg_preds = aggregate_predictions(predictions)
    total = 0
    for pred in predictions:
        dist = pairwise_distance(pred, agg_preds)
        squared_dist = dist * dist
        total += squared_dist
    variance = total/len(predictions)
    return variance


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
