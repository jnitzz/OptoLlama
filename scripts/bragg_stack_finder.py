import os
from safetensors import safe_open
from optollama.utils.utils import init_tokenmaps

def parse_tokens(data, idx_to_token, eos_idx, pad_idx, msk_idx):
    '''
    Creates a tuple of material (string) and thickness (float) out of a token.
    '''
    tokens = [idx_to_token[int(t)] for t in data[: 21] if int(t) not in (eos_idx, pad_idx, msk_idx)]
    
    new_stack = []
    for token in tokens:
        a, b = token.split("_")
        parsed = (a, int(b))
        new_stack.append(parsed)
 
    return new_stack

def is_same_mat(mat1, mat2):
    if mat1[0] == mat2[0]: # and abs(mat1[1] - mat2[1]) <= 20:
        return True
    else:
        return False

def is_dif_ok(list_mats):
    max_thickness = list_mats[0][1]
    min_thickness = list_mats[0][1]
    for mat in list_mats:
        if mat[1] > max_thickness:
            max_thickness = mat[1]
        if mat[1] < min_thickness:
            min_thickness = mat[1]
    if max_thickness - min_thickness <= 20:
        return True
    else:
        return False

def is_dif_ok_separate(list_mats):
    even = is_dif_ok(list_mats[::2])
    odd = is_dif_ok(list_mats[1::2])
    result = even and odd
    return result

    

def detect_Bragg(stack):
    # Countin all
    # if ABABA counting ABABA ABA BAB ABA -> 4
    count = 0 # How many Bragg stacks are detected
    start = 0 # Current starting index of stack
    prior_length = 0 # If a Bragg stack is continued the length of the existing longest stack
    while start < len(stack) - 2: # Starting point must be at least 3 away from the end (minimum stack lengh is 3)
        if prior_length==0: # we are not continueing an existing stack
            if is_same_mat(stack[start], stack[start+2]) and not is_same_mat(stack[start], stack[start+1]) and is_dif_ok_separate(stack[start:start+3]):
                # same material exists in the position after the next and the material in between is different and the thickness differens is ok
                count += 1 # add to count
                prior_length = 3 # we have a stack that can be continued of length 3
                #print(stack[start:start+3])
            else:
                start += 1 # no stack that starts in this index, continue
        else: # continueing existing stack
            if start + prior_length + 1 < len(stack) and is_same_mat(stack[start], stack[start + prior_length + 1]) and is_same_mat(stack[start + 1], stack[start + prior_length]) and is_dif_ok_separate(stack[start:start + prior_length + 2]):
                # the index will not run out of bounds and starting material is the same as ending material and the sandwiched materials are also the same and difference is ok
                count += 1 # We have an extended stack
                #print(stack[start: start+prior_length + 2])
                prior_length += 2 # prior length has grown
            else:
                start +=1 # Stack is finished moving on
                prior_length = 0
    return count
            
def detect_Bragg_only_length_3(stack):
    # only counting lengths of 3
    # if ABABA -> coungting ABA BAB ABA -> 3
    count = 0 # How many Bragg stacks are detected
    start = 0 # Current starting index of stack
    while start < len(stack) - 2: # Starting point must be at least 3 away from the end (minimum stack lengh is 3)
        if is_same_mat(stack[start], stack[start+2]) and not is_same_mat(stack[start], stack[start+1]) and is_dif_ok_separate(stack[start:start+3]):
            # same material exists in the position after the next and the material in between is different and the thickness differens is ok
            count += 1 # add to count
        start += 1 # no stack that starts in this index, continue
    return count

def detect_Bragg_continued(stack, count_dict):
    # countin continued stacks only once
    # if ABABA -> just ABABA -> 1
    # but if ABACA -> ABA and ACA -> 2
    count = 0 # How many Bragg stacks are detected
    start = 0 # Current starting index of stack
    prior_length = 0 # If a Bragg stack is continued the length of the existing longest stack
    while start < len(stack) - 2: # Starting point must be at least 3 away from the end (minimum stack lengh is 3)
        if prior_length==0: # we are not continueing an existing stack
            if is_same_mat(stack[start], stack[start+2]) and not is_same_mat(stack[start], stack[start+1]) and is_dif_ok_separate(stack[start:start+3]):
                # same material exists in the position after the next and the material in between is different and the thickness differens is ok
                count += 1 # add to count
                prior_length = 3 # we have a stack that can be continued of length 3
                #print(stack[start:start+3])
            else:
                start += 1 # no stack that starts in this index, continue
        else: # continueing existing stack
            if start + prior_length + 1 < len(stack) and is_same_mat(stack[start], stack[start + prior_length + 1]) and is_same_mat(stack[start + 1], stack[start + prior_length]) and is_dif_ok_separate(stack[start:start + prior_length + 2]):
                # the index will not run out of bounds and starting material is the same as ending material and the sandwiched materials are also the same and difference is ok
                #count += 1 # We have an extended stack
                #print(stack[start: start+prior_length + 2])
                prior_length += 2 # prior length has grown
            else:
                count_dict[prior_length] = count_dict.get(prior_length, 0) + 1
                start = start + prior_length - 1 # Stack is finished moving on
                prior_length = 0
    return count, count_dict


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
    option = "continued" # "regular" "continued" "just_len_3"
    count_dict = {}

    print("Only counting stacks of length 3")
    for i in range(10):
        total_Bragg_count, includes_Bragg, count_dict = find_Bragg_stacks_in_train_data(total_Bragg_count, includes_Bragg, i, option, count_dict)
        print(f"Total Bragg Count: {total_Bragg_count}")
        print(f"Total data items including a Bragg Stack: {includes_Bragg}")

    print(count_dict)


