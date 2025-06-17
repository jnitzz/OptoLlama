from collections import defaultdict, Counter
def parse_tokens(token):
    parsed = [tuple([a, float(b)]) for a, b in (s.split("__") for s in token)]
    return parsed
    

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


def aggregation(predictions):
    transposed = list(map(list, zip(*predictions))) # transpose list so that the batch dimension is on the outer side
    aggregated = [] # new prediction list
    for item in transposed: # for each item in batch
        parsed_samples = []
        for sample in item: # for each sample drawn
            parsed_stack = [] # create a new prediction
            for layer in sample: # for each predicted layer
                parsed_stack.append(parse_tokens(layer)) # parse the tokens of the layer and add it to the new prediction
            parsed_samples.append(parced_stack)
        aggregated_samples = aggregate_predictions(parsed_samples)
        aggregated.append(parsed_stack)
    