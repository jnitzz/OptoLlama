import os
from safetensors import safe_open
from optollama.data.token import init_tokens
import json
from optollama.data.bragg_stack_finder import parse_tokens, detect_Bragg, detect_Bragg_continued, detect_Bragg_only_length_3


def find_Bragg_stacks_in_test_data(total_Bragg_count, includes_Bragg, target, option, count_dict):

    data_path = "./data/TF_safetensors/tokens.json"
    file_path = os.path.join("./runs/OptoLlama/", f"{target}/results_OptoLlama_target_ids.json")
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = init_tokens(data_path)

    with open(file_path, "r") as f:
        data = json.load(f)

    films = data[0]

    for item in films:
        new_item = parse_tokens(item, idx_to_token, eos_idx, pad_idx, msk_idx)
        #print(new_item)
        if option == "regular": 
            bragg_count = detect_Bragg(new_item)
        elif option == "continued":
            bragg_count, count_dict = detect_Bragg_continued(new_item, count_dict)
        elif option == "just_len_3":
            bragg_count = detect_Bragg_only_length_3(new_item)
        total_Bragg_count = total_Bragg_count + bragg_count
        if bragg_count != 0:
            includes_Bragg += 1
            #print(f"Bragg Stack found. Current Bragg count: {total_Bragg_count}")
        #print(bragg_count)
    
    return total_Bragg_count, includes_Bragg, count_dict

if __name__ == "__main__":

    total_Bragg_count = 0
    includes_Bragg = 0

    for option in ["regular", "continued", "just_len_3"]:
        for i in ["morphocolor","perfectabsorber","thermalreflector","bandnotch"]:
            count_dict = {}
            total_Bragg_count = 0
            includes_Bragg = 0
            total_Bragg_count, includes_Bragg, count_dict = find_Bragg_stacks_in_test_data(total_Bragg_count, includes_Bragg, i, option, count_dict)
            print(option)
            print(i)
            print(f"Total Bragg Count: {total_Bragg_count}")
            print(f"Total data items including a Bragg Stack: {includes_Bragg}")
            print(count_dict)


