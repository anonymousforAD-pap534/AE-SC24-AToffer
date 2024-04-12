#            _____________     ____  ______      ______
#  if  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14
#      0  1  0  0  0  0  0  1  0  0   1   1   0   1   1   
#      k0 k1     k2        k3   k4     k5    k6    k7  ---- 8Group 
#      G0        G1             G2     G3    G4    G5
#      1     0 - 1000001         2      2     1    2
#      2     1 - 0111110   
#
#            ______*______  *  ____   _____       _____
# true 0  1  0  0  0  0  0  1  0  0   1   1   0   1   1   sparse  init value  41115
#       k0        k1        k2  k3      k4    k5    k6   
#      01/11    000000      1      
#
#
import os
import sys
import csv
import timeit
import math
import random
import numpy as np


def translate_value(value):
    binary_str = bin(value)[2:].zfill(16)  
    result = []
    segment_count = 0
    segment_start = None
    prev_bit = None
    
    for i in range(1, 16):
        if i == 1:
            prev_bit = binary_str[i]
            segment_start = i
        elif binary_str[i] == prev_bit:
            continue
        else:
            if i - segment_start > 1:
                result.extend([segment_start, i - 1])
                segment_count += 1
            prev_bit = binary_str[i]
            segment_start = i

    if 16 - segment_start > 1:  
        result.extend([segment_start, 15])
        segment_count += 1
        
    if segment_count == 1 and result[0] == 0 and result[-1] == 15:
        segment_count = 0
        result = []

    # return f"{binary_str} {binary_str[0]} {segment_count} {' '.join(map(str, result)) if segment_count > 0 else ''}"
    return f"{binary_str[0]} {segment_count} {' '.join(map(str, result)) if segment_count > 0 else ''}"


def binary_to_decimal(binary_str):
    return int(binary_str, 2)

def Keyto16bit(key):
    binary_str = bin(key)[2:].zfill(6)
    result = ""  
    
    if int(binary_str[0]) == 0:
        result += "00"
    else:
        result += "10"
    # print("6bit bitstr: ", binary_str)
    if int(binary_str[1])== 0:
        result += "1000001"
    else:
        result += "0111110"
    if int(binary_str[2]) == 0:
        result += "00"
    else:
        result += "11"
        
    if int(binary_str[3]) == 0:
        result += "00"
    else:
        result += "11"
        
    if int(binary_str[4]) == 0:
        result += "0"
    else:
        result += "1"
    
    if int(binary_str[5]) == 0:
        result += "00"
    else:
        result += "11"
        
    return result


def main():
    batch_size       = 16
    layer_num        = 12
    seq_len          = int(sys.argv[1])
    head_num         = 12
    head_size        = 64
    mask_id          = int(sys.argv[2])
    if_fused         = 1
    iter_time        = 128
    round_time       = 5
    value_init       = 41115
    change_prob_iter = 4
    results = []
    min_times = {}
    global_best_config = 0
    global_min_time = min_times.get(1)
    
    search_high_bound = 49151
    search_low_bound = 16384
    random_num = 1000
    group_len = [1, 1, 2, 2, 1, 2]  # len = 6 group
        
    t0_start = timeit.default_timer() 
    
    for ii in range(round_time):
        iteration_times = []
        min_times.clear()
        score_group = [0] * len(group_len) 
        min_time_this_prob = None
        
        pow_group_len = [int(math.pow(2, length)) for length in group_len]
        prefix_pow_group = []
        prefix_sum = 0
        for length in pow_group_len:
            prefix_sum += length
            prefix_pow_group.append(prefix_sum)
        print("init prefix:", prefix_pow_group)
        prob_coefficient = random_num/prefix_pow_group[len(group_len) - 1] 
        
        appeared_values = set()
        value = value_init 
        int_group = [random.randint(0, high-1) for high in pow_group_len]
        binstr_group = [bin(value)[2:].zfill(length) for value, length in zip(int_group, group_len)]
        if binstr_group[1] == "0":
            binstr_group[1] = "1000001"
        else:
            binstr_group[0] = "0111110"
        first_bit = "1" 
        binary_string = first_bit + ''.join(binstr_group)
        prefix_prob = [int(prob_coefficient * length) for length in prefix_pow_group]
        
                
        index = 1
        while index <= iter_time:
            if index % change_prob_iter == 1 and index != 1:
                first_bit = str(random.randint(0, 1))
                min_time_this_prob = min(min_times.values())
                score_group = [0] * len(group_len)   
                
            iter_prob = random.randint(0, random_num)
            
            group_this_id = None
            for id, prefix in enumerate(prefix_prob):
                if iter_prob <= prefix:
                    group_this_id = id
                    break
           
            int_group[group_this_id] = random.randint(0, pow_group_len[group_this_id] - 1)                
            binstr_group = [bin(value)[2:].zfill(length) for value, length in zip(int_group, group_len)]
            if binstr_group[1] == "0":
                binstr_group[1] = "1000001"
            else:
                binstr_group[0] = "0111110"
                
            binary_string = first_bit + ''.join(binstr_group)
            value = binary_to_decimal(binary_string)
            if index == 1:
                value = value_init
    
            if value not in appeared_values and search_low_bound <= value < search_high_bound:
                appeared_values.add(value)
          
                # ---------------------------------------------
                translation = translate_value(value)
                parts = translation.split()
                segment_num = int(parts[1])
                segments = []
                for i in range(2, len(parts), 2):
                    start = int(parts[i]) - 1 
                    end = int(parts[i+1]) - 1
                    segments.extend([start, end])
            
                command = "python generate_genidx.py" + " {} {} {} {} {} {} {} {}".format(batch_size, layer_num, seq_len, head_num, head_size, mask_id, if_fused, value)
                output = os.popen(command).read()
                time_value = float(output.split("time costs:")[1].split("ms")[0].strip())
                
                #To print some INFO 
                if index % 16 == 1:
                    print("round:",ii+1," iter:", index, " time_value:", time_value ,"\tvalue:", value, segment_num, segments)
                
                min_time = min_times.get(index)
                if min_time is None or time_value < min_time:
                    min_times[index] = time_value
                    min_time = min(min_times.values())
                    if global_min_time is None or min_time < global_min_time:
                        global_min_time = min_time
                        global_best_config = value
                iteration_times.append(min_time)
                # ---------------------------------------------
                
                
                if min_time_this_prob != None and time_value < min_time_this_prob:
                    # print("min_time_this_prob: ", min_time_this_prob, "\t this Time_value: ", time_value)
                    score_group[group_this_id] = score_group[group_this_id] + 1
                
                if index % change_prob_iter == 0:
                    max_index = max(range(len(score_group)), key=lambda i: score_group[i])
                    max_value = score_group[max_index]
                    same_max_values = [i for i, v in enumerate(score_group) if v == max_value]
               

                    if max_value != 0 and len(same_max_values) != len(group_len):
                       
                    
                        for max_idx in same_max_values:
                            pow_group_len[max_idx] = pow_group_len[max_idx] + int(math.pow(2, max_value))                    
                        
                        prefix_pow_group = []
                        prefix_sum = 0
                        for length in pow_group_len:
                            prefix_sum += length
                            prefix_pow_group.append(prefix_sum)
                        print("! Changed prefix:", prefix_pow_group)
                        prob_coefficient = random_num/prefix_pow_group[len(group_len) - 1] 
                        
                        
                    
                index += 1     
            
        results.append(iteration_times)
    
    
    
    csv_filename = f"./search_result/{mask_id}_ours_seq{seq_len}_Time.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration"] + [f"round_{ii+1}" for ii in range(round_time)]) 
        for index in range(iter_time):
            row = [index+1] + [results[ii][index] for ii in range(round_time)]
            writer.writerow(row)
              
    print(f"Result have been stored in {csv_filename} !")
    
    t0_end = timeit.default_timer()
    print("\nOur probability search time costs:  \t{:.1f} s".format((t0_end - t0_start))) 
    print("Mask_id:", mask_id, "\tSeqlen:", seq_len)
    print("Global Min Time:", global_min_time, "\tGlobal best Config:", global_best_config)


    with open("best_config_ours.txt", "a") as file:
        file.write("Mask_id:{} \tSeqlen:{} \tGlobal Min Time:{}\tGlobal best Config: {}\n".format(mask_id, seq_len, global_min_time, global_best_config))

    
if __name__ == "__main__":
    main()