
def get_prefix(line_table):
    prefix_table = []
    prefix_sum = 0
    for num in line_table:
        prefix_sum += (num + 1)
        prefix_table.append(prefix_sum)
    return prefix_table

def func_return_insert(start, end):
    return_code = ""
    if start == 0 and end <=7: 
        return_code += "input_tensor, "
    if start <= 9 and end >=9 and end <=12:
        return_code += "residual, "
    if start <= 1 and end <=4: 
        return_code += "v, "
    
    if end == 0:
        return_code += "qkv"
    elif end == 1:
        return_code +=  "q, k, v"
    elif end == 2 or end == 3 :
        return_code +=  "scores"
    elif end == 4:
        return_code +=  "probs"
    elif end == 5:
        return_code +=  "h"
    else:  # 6 7 8 9 10 11 12 13 14
        return_code += "hidden_states"
    
    return return_code
    
def func_define_insert(fused_funcname, start, end):
    define_code = fused_funcname + "("
    if(start == 0):
        define_code
    
    if end >= 0:
        if start == 0:
            define_code +=  "qkv_kernel_this, "
    if end >= 1:
        if start <= 1:
            define_code +=  "qkv_bias_this, "
    if end >= 7:
        if start <= 7:
            define_code +=  "attr_output_kernel_this, "
    if end >= 8:
        if start <= 8:
            define_code +=  "attr_output_bias_this, "
    if end >= 9:
        if start <= 9:
            define_code +=  "attr_output_layernorm_gamma_this, attr_output_layernorm_beta_this, "
    if end >= 10:
        if start <= 10:
            define_code +=  "inter_kernel_this, "
    if end >= 11:
        if start <= 11:
            define_code +=  "inter_bias_this, "
    if end >= 12:
        if start <= 12:
            define_code +=  "output_kernel_this, "
    if end >= 13:
        if start <= 13:
            define_code +=  "output_bias_this, "
    if end == 14:
        define_code +=  "output_layernorm_gamma_this, output_layernorm_beta_this, "
               

    
    if 1 <= start <= 8 and end >= 8:
        define_code += "input_tensor, " 
    if start == 1 and end >= 1:
        define_code += "qkv, "
    if start == 2:
        define_code +=  "q, k, "
    if 2 <= start <= 5 and end >= 5:
        define_code +=  "v, "
    if start <=3 and end >= 3:
        define_code +=  "mask, "
    if start == 3 or start == 4:
        define_code +=  "scores, "
    if start == 5 and end >= 5:
        define_code +=  "probs, "
    if start == 6 and end >= 6:
        define_code += "h, "
    if 10 <= start <= 13 and end >= 13:
        define_code += "residual, "
    
    
    if start == 0 or 6 <= start <= 14:
        define_code += "hidden_states, "
        
    if define_code.endswith(", "):
        define_code = define_code[:-2]
    
    define_code += ")"
    return define_code
    
    
def func_use_insert(fused_funcname, start, end):
    use_code = func_return_insert(start, end) +" = "+func_define_insert(fused_funcname, start, end)
    return use_code  