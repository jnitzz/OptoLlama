import os
from safetensors import safe_open
from optollama.utils.utils import init_tokenmaps
from optollama.utils.bragg_stack_finder import parse_tokens, detect_Bragg, detect_Bragg_continued, detect_Bragg_only_length_3


def find_Bragg_stacks_in_train_data(total_Bragg_count, includes_Bragg, index, option, count_dict):

    data_path = "./data/TF_safetensors/"
    file_path = os.path.join(data_path, f"train-{index}.safetensors")
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = init_tokenmaps(data_path)

    with safe_open(file_path, framework="pt") as f:
        print("Keys:", f.keys())
        
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"\nKey: {key}")
            print("Shape:", tensor.shape)
            print("Dtype:", tensor.dtype)

        films = f.get_tensor('thin_films')

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
    options = ["regular", "continued", "just_len_3"]
    count_dict = {}

    for option in options:
        print(option)
        for i in range(10):
            total_Bragg_count, includes_Bragg, count_dict = find_Bragg_stacks_in_train_data(total_Bragg_count, includes_Bragg, i, option, count_dict)
            print(f"Total Bragg Count: {total_Bragg_count}")
            print(f"Total data items including a Bragg Stack: {includes_Bragg}")

    print(count_dict)