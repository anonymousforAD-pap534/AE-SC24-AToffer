import argparse
import timeit
import torch
import numpy as np
import random
import torch.nn.functional as F
from utils.utils import *
import os
from torch.utils.cpp_extension import load

path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
syncfree_attention = load(
    name = "syncfree_attention",
    sources = [os.path.join(path, "src/syncfree_attention.cu"), os.path.join(path, "src/syncfree_attention.cpp")],
    verbose = False,
)


@torch.jit.script
def fused_with_jit_gemm1_bias1_lynorm1_gemm2(attr_output_kernel_this, attr_output_bias_this, attr_output_layernorm_gamma_this, attr_output_layernorm_beta_this, inter_kernel_this, input_tensor, hidden_states):
	hidden_states = torch.matmul(hidden_states, attr_output_kernel_this)      # 7 GEMM1
	
	hidden_states = hidden_states + attr_output_bias_this                     # 8 add bias
	hidden_states = hidden_states + input_tensor
	
	hidden_states = F.layer_norm(hidden_states, (768, ), weight=attr_output_layernorm_gamma_this, bias=attr_output_layernorm_beta_this)
	residual = hidden_states                                                  # 9 layernorm
	
	hidden_states = torch.matmul(hidden_states, inter_kernel_this)            # 10 GEMM2
	
	return residual, hidden_states


@torch.jit.script
def fused_with_jit_bias2act_gemm3_bias3_lynorm3(inter_bias_this, output_kernel_this, output_bias_this, output_layernorm_gamma_this, output_layernorm_beta_this, residual, hidden_states):
	hidden_states = hidden_states + inter_bias_this                           # 11 add bias & act
	hidden_states = F.gelu(hidden_states)
	
	hidden_states = torch.matmul(hidden_states, output_kernel_this)            # 12 GEMM3
	
	hidden_states = hidden_states + output_bias_this                           # 13 add bias
	hidden_states = hidden_states + residual
	
	hidden_states = F.layer_norm(hidden_states, (768, ), weight=output_layernorm_gamma_this, bias=output_layernorm_beta_this)                                           # 14 layernorm
	
	return hidden_states


def bert_forward(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # torch_cuda_deactive()
    batch_size = args['batch_size']  # 1
    layer_num = args['layer_num']  # 12
    seq_len = args['seq_len']  # 64
    head_num = args['head_num']   # 12
    head_size = args['head_size']   # 64
    mask_id = args['mask_id']   # type of mask: 0-lower triangle, 1-strided, 2-fixed
    avg_seq_len = -1
    hidden_dim = head_num * head_size  # always 12 * 64 = 768
    dtype = "fp32"
    
    # for key in args:
    #     print("{:13} : {:5} ".format(key, args[key]))
    # print("-"*21, "Argument", "-"*21)
    
    if avg_seq_len <= 0:
        avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1) if 2 * avg_seq_len > seq_len else (0, 2 * avg_seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask                   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), dtype)
    
    lower_triangle_mask = generate_triangle_mask(attr_mask).cuda()
    strided_mask = generate_strided_mask(attr_mask).cuda()
    fixed_mask = generate_fixed_mask(attr_mask).cuda()
    
    if(mask_id == 0):
        mask_name = 'Lower_triangle_mask'
        mask = lower_triangle_mask
    elif(mask_id == 1): 
        mask_name = 'Strided_mask'
        mask = strided_mask
    else:
        mask_name = 'Fixed_mask'
        mask = fixed_mask
    
    input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
    qkv_kernel                  = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_kernel          = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_bias            = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_gamma = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_beta  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_kernel                = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_bias                  = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_kernel               = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_bias                 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_gamma      = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_beta       = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    transformer_output          = [None for _ in range(layer_num)]
    
    hidden_states = input_from_tensor
    layer = 0 
    qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]

    warmup_iters = 10
    iters = 100
    for i in range(warmup_iters + iters):
        if i == warmup_iters:    
            torch.cuda.synchronize()
            t0_start = timeit.default_timer() 

        with torch.no_grad():
            hidden_states = input_from_tensor
            for layer in range(layer_num):
                qkv_kernel_this = qkv_kernel[layer]
                qkv_bias_this = qkv_bias[layer]   
                attr_output_kernel_this = attr_output_kernel[layer]
                attr_output_bias_this = attr_output_bias[layer] 
                attr_output_layernorm_gamma_this = attr_output_layernorm_gamma[layer]
                attr_output_layernorm_beta_this = attr_output_layernorm_beta[layer]
                inter_kernel_this = inter_kernel[layer]
                inter_bias_this = inter_bias[layer]
                output_kernel_this = output_kernel[layer]
                output_bias_this = output_bias[layer]
                output_layernorm_gamma_this = output_layernorm_gamma[layer]
                output_layernorm_beta_this = output_layernorm_beta[layer]
            
                
                input_tensor = hidden_states                                                # 0 qkv GEMM0
                qkv = torch.matmul(hidden_states, qkv_kernel_this)
                
                qkv = qkv + qkv_bias_this                                                   # 1 add bias
                q, k, v = qkv.chunk(3, dim=-1)                                              
                q = transpose_for_scores1(q)                            
                k = transpose_for_scores1(k)                            
                v = transpose_for_scores1(v)                            
                
                # Replaced with my fused function to run -------------------
                hidden_states = syncfree_attention.run_syncfree_strided_attention(q, k, v)

                residual, hidden_states = fused_with_jit_gemm1_bias1_lynorm1_gemm2(attr_output_kernel_this, attr_output_bias_this, attr_output_layernorm_gamma_this, attr_output_layernorm_beta_this, inter_kernel_this, input_tensor, hidden_states)

                hidden_states = fused_with_jit_bias2act_gemm3_bias3_lynorm3(inter_bias_this, output_kernel_this, output_bias_this, output_layernorm_gamma_this, output_layernorm_beta_this, residual, hidden_states)

                transformer_output[layer] = hidden_states                                 
            
    torch.cuda.synchronize()
    t0_end = timeit.default_timer()
    print("({}) time costs:  \t{:.3f} ms / iter".format(mask_name, (t0_end - t0_start) * 1000 / iters)) 
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, default=1, help='batch size')
    parser.add_argument('layer_num', type=int, default=12, help='number of layers')
    parser.add_argument('seq_len', type=int, default=64, help='sequence length')
    parser.add_argument('head_num', type=int, default=12, help='head number')
    parser.add_argument('head_size', type=int, default=64, help='size per head')
    parser.add_argument('mask_id', type=int, default=0, help='type of mask: 0-lower triangle, 1-strided, 2-fixed')
    args = parser.parse_args()
    bert_forward(vars(args))