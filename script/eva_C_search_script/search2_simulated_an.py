import os
import sys
import csv
import timeit
import math
import random
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
    results = []
    min_times = {}
    global_best_config = 0
    global_min_time = min_times.get(1)

    t0_start = timeit.default_timer() 
    
    for ii in range(5):
        iteration_times = []
        min_times.clear()
        current_value = random.randint(0, 65535)
        new_value = current_value
        
        initial_temperature = 100
        cooling_rate = 0.99
        temperature = initial_temperature
        
        index = 1
        while index <= iter_time:
            if index != 1:
                new_value = current_value ^ (1 << random.randint(0, 15)) 
            
            translation = translate_value(new_value)
        
            parts = translation.split()
            segment_num = int(parts[1])
            segments = []
            for i in range(2, len(parts), 2):
                start = int(parts[i]) - 1 
                end = int(parts[i+1]) - 1
                segments.extend([start, end])
        
            command = "python generate_genidx.py" + " {} {} {} {} {} {} {} {}".format(batch_size, layer_num, seq_len, head_num, head_size, mask_id, if_fused, new_value)
            output = os.popen(command).read()
            time_value = float(output.split("time costs:")[1].split("ms")[0].strip())        
            
            #To print some INFO 
            if index % 16 == 1:
                print("round:",ii+1," iter:", index, " time_value:", time_value ,"\tvalue:", new_value, segment_num, segments)
            
            if index == 1:
                best_time_value = time_value
                current_time_value = time_value
            

            else:
                if time_value < current_time_value or random.random() < math.exp((current_time_value - time_value) / temperature):
                    current_value = new_value
                    current_time_value = time_value
                    if time_value < best_time_value:
                        best_time_value = time_value
                

                temperature *= cooling_rate
            
            min_time = min_times.get(index)
            if min_time is None or time_value < min_time:
                min_times[index] = time_value
                min_time = min(min_times.values())
                if global_min_time is None or min_time < global_min_time:
                    global_min_time = min_time
                    global_best_config = new_value
            iteration_times.append(min_time)
            
            index += 1 
            
        results.append(iteration_times)
            
             
    csv_filename = f"./search_result/{mask_id}_sa_seq{seq_len}_Time.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration"] + [f"round_{ii+1}" for ii in range(round_time)]) 
        for index in range(iter_time):
            row = [index+1] + [results[ii][index] for ii in range(round_time)]
            writer.writerow(row)
            
    print(f"Result have been stored in {csv_filename} !")
    
    t0_end = timeit.default_timer()
    print("\nSA search time costs:  \t{:.1f} s".format((t0_end - t0_start))) 
    print("Global Min Time:", global_min_time, "\tGlobal best Config:", global_best_config)
    
if __name__ == "__main__":
    main()
