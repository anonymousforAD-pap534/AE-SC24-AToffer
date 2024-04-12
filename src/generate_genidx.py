     
import sys
import os
from utils.generate_utils import get_prefix, func_define_insert, func_return_insert, func_use_insert

def generate_head_code(import_end, filename):
    with open("forward.py", "r") as f_forward:
        lines_forward = f_forward.readlines()
    with open(filename + ".py", "w", encoding='UTF-8') as f_fuse:
        for line in lines_forward[:import_end]:
            f_fuse.write(line)
            
            
def generate_body_code(code_start, import_end, filename):
    with open("forward.py", "r") as f_forward:
        lines_forward = f_forward.readlines()
    with open(filename + ".py", "a") as f_fuse:
        for line in lines_forward[import_end : code_start]:
            f_fuse.write(line)
            
            
def generate_tail_code(code_end, filename):
    with open("forward.py", "r") as f_forward:
        lines_forward = f_forward.readlines()
    with open(filename + ".py", "a") as f_fuse:
        for line in lines_forward[code_end : ]:
            f_fuse.write(line)
        
            
def generate_nofused_code(if_fused, filename):
    with open("forward.py", "r") as f_forward:
        lines_forward = f_forward.readlines()
    if if_fused == 0:
        with open(filename + ".py", "w") as f_fuse:
            for line in lines_forward[0:]:
                f_fuse.write(line)
        return
    

def generate_callfunc_code(if_handwritten, mask_id, start, end, filename):
    name_table = ['gemm0', 'bias0', 'QK', 'mask', 'sfmx', 'pV', 'trans', 'gemm1', 'bias1', 'lynorm1', 'gemm2'
                , 'bias2act', 'gemm3', 'bias3', 'lynorm3']
    
    if if_handwritten == 1 and start == 2 and end == 6:
        with open("{}.py".format(filename), "a") as f_fuse:
            # To insert the func
            f_fuse.write("                # Replaced with my fused function to run -------------------\n")
            my_func = " "*16
            if(mask_id == 0):
                my_func += "hidden_states = syncfree_attention.run_syncfree_triangle_attention(q, k, v)"
            elif(mask_id == 1): 
                my_func += "hidden_states = syncfree_attention.run_syncfree_strided_attention(q, k, v)"
            else:
                my_func += "hidden_states = syncfree_attention.run_syncfree_fixed_attention(q, k, v)"
            
            f_fuse.write(my_func + "\n\n")
        
    else:
        fused_funcname = "fused_with_jit"
        for name_idx in range(end - start + 1):
            fused_funcname += ("_"+name_table[start + name_idx])        
        with open(filename + ".py", "a") as f_fuse:
            f_fuse.write(" "*16 + func_use_insert(fused_funcname, start, end) + "\n\n")
        


def generate_fused_code2(code_start, start, end, filename):
    line_table = [2, 5, 1, 1, 1, 1, 3, 1, 2, 2, 1, 2, 1, 2, 1]
    prefix_table = get_prefix(line_table)
    
    with open("forward.py", "r") as f_forward:
        lines_forward = f_forward.readlines()
    
    if start == 0:
        read_start = 0
    else:
        read_start = prefix_table[start - 1]
    read_end = prefix_table[end]
    with open(filename + ".py", "a") as f_fuse:
        for line in lines_forward[code_start + read_start : code_start + read_end]:
            f_fuse.write(line)



def generate_fused_code1(code_start, start, end, if_handwritten, mask_id, filename):
    line_table = [2, 5, 1, 1, 1, 1, 3, 1, 2, 2, 1, 2, 1, 2, 1]
    name_table = ['gemm0', 'bias0', 'QK', 'mask', 'sfmx', 'pV', 'trans', 'gemm1', 'bias1', 'lynorm1', 'gemm2'
                , 'bias2act', 'gemm3', 'bias3', 'lynorm3']
    prefix_table = get_prefix(line_table)
    with open("forward.py", "r") as f_forward:
        lines_forward = f_forward.readlines()
    
    
    if if_handwritten == 1 and start == 2 and end == 6:
        return 
        
    else:
        fused_funcname = "fused_with_jit"
        for name_idx in range(end - start + 1):
            fused_funcname += ("_"+name_table[start + name_idx])
        
        with open("{}.py".format(filename), "a") as f_fuse:
            # Prepare the lines from forward.py for new fusion function
            f_fuse.write("\n@torch.jit.script\n")
            f_fuse.write("def " + func_define_insert(fused_funcname, start, end) + ":\n") 
            
            if start == 0: 
                read_start = 0
            else:
                read_start = prefix_table[start - 1]
            for line in lines_forward[code_start + read_start : code_start + prefix_table[end]]:
                f_fuse.write("\t" + line.strip() + "\n")
            
            f_fuse.write("\treturn " + func_return_insert(start, end) + "\n\n") 



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
    batch_size       = int(sys.argv[1])
    layer_num        = int(sys.argv[2])
    seq_len          = int(sys.argv[3])
    head_num         = int(sys.argv[4])
    head_size        = int(sys.argv[5])
    mask_id          = int(sys.argv[6])
    if_fused         = int(sys.argv[7])
    if(mask_id == 0):
        mask_name = 'Lower_triangle_mask_'
    elif(mask_id == 1): 
        mask_name = 'Strided_mask_'
    else: 
        mask_name = 'Fixed_mask_'
        
    # just no_fusion
    if len(sys.argv) ==8 or if_fused == 0: 
        filename = mask_name + "nofused_forward"
        generate_nofused_code(if_fused, filename)
        # print("[INFO]: Multiseg-code nofusion generated successfully!")
        os.system("python "+filename + ".py" + 
            " {} {} {} {} {} {}".format(batch_size, layer_num, seq_len, head_num, head_size, mask_id))
        
        return
    
    # if fused, check value
    try:
        value            = int(sys.argv[8])
    except ValueError:
        print("[ERROR]: Invalid input value")
        sys.exit(1)
        
    if value < 0 or value > 65535:
        print("[ERROR]: Input value must be between 0 and 65535")
        sys.exit(1)
    

    translation = translate_value(value)
    

    parts = translation.split()
    if_handwritten = int(parts[0])
    segment_num = int(parts[1])
    segments = []
    for i in range(2, len(parts), 2):
        start = int(parts[i])
        end = int(parts[i+1])
        segments.extend([start - 1, end - 1])
        
    # check num of segment
    if len(segments) != segment_num*2 :    
        print("[ERROR]: Number of segments does not match the number of provided start/end pairs")
        return
    
    print("[INFO]: input parameter check success !")
    print("  batch_size:", batch_size, "  layer_num:", layer_num, "  seq_len:", seq_len, "  head_num:", head_num, "  head_size", head_size)
    print("  mask_name:", mask_name, "  if_fused:", if_fused, "  if_handwritten:", if_handwritten, "  segment_num:", segment_num, segments)
    filename = mask_name + "fused_forward"
    
    
    code_start = 102
    code_end = code_start + 41
    import_end = 17  
    
    if segment_num == 0:
        filename = mask_name + "nofused_forward"
        if_fused = 0
        generate_nofused_code(if_fused, filename)
        # print("[INFO]: Multiseg-code nofusion generated successfully!")
        os.system("python "+filename + ".py" + 
            " {} {} {} {} {} {}".format(batch_size, layer_num, seq_len, head_num, head_size, mask_id))
        return
    
    for ii in range(segment_num):
        if ii == 0:
            generate_head_code(import_end, filename)
        start = segments[2*ii]
        end = segments[2*ii + 1]
        generate_fused_code1(code_start, start, end, if_handwritten, mask_id, filename)
        
    generate_body_code(code_start, import_end, filename)
    
    
    for ii in range(segment_num):
        start = segments[2*ii]
        end = segments[2*ii + 1]
        if ii == 0 and start != 0:
            generate_fused_code2(code_start, 0, start - 1, filename)
        generate_callfunc_code(if_handwritten, mask_id, start, end, filename) 
        if ii == segment_num - 1 and end != 14:
            generate_fused_code2(code_start, end + 1, 14, filename)
        
        if ii + 1 < segment_num:
            generate_fused_code2(code_start, end + 1,segments[2*ii + 2] - 1, filename)
        
            
    generate_tail_code(code_end, filename)
    
    # To execute    
    os.system("python "+filename + ".py" + 
            " {} {} {} {} {} {}".format(batch_size, layer_num, seq_len, head_num, head_size, mask_id))
    
if __name__ == "__main__":
    main()