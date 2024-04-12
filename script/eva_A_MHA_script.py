# 2024.3.31 Mar. Sun. 
#  实验 A的数据脚本
#  对比的对象只包含/pyotrch jit/pytorch  baseline --- 
#  适宜于批量测试1/8/16 不同 seq_len、 mask（Fixed / strided）
#
#  python A_script_fusedMHA.py 
import csv
import sys
import timeit
import torch
import numpy as np
import random
import torch.nn.functional as F
import os
from torch.utils.cpp_extension import load
from utils.utils import *

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

syncfree_attention = load(
    name = "syncfree_attention",
    sources = [os.path.join(path, "../src/syncfree_attention.cu"), os.path.join(path, "../src/syncfree_attention.cpp")],
    verbose = False,
)

@torch.jit.script
def Attention_Jit(q, k, v, mask):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    scores -= 10000.0 * (1.0 - mask.unsqueeze(1))   
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous() 
    new_context_layer_shape = h.size()[:-2] + (q.shape[1]*q.shape[3], )
    hidden_states0 = h.view(new_context_layer_shape) 
    return h.view(new_context_layer_shape)    


def bert_example():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    platform = int(sys.argv[1])  #输入参数
    head_num = 12
    head_size = 64
    avg_seq_len = -1
    hidden_dim = head_num * head_size
    layer_num = 12
    batch_size = 16
    warmup_iters = 10
    iters = 100
    dtype = "fp32"
    

    for mask_id in [1, 2]:
        if(mask_id == 1):
            mask_name = 'Strided'
        else:
            mask_name = 'Fixed'
            
        #一个平台应该保存两张这个表出来
        with open('{}_{}_A_script_result.csv'.format(platform, mask_name), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['batch_size', '64', '128', '256', '384', '512', '768', '1024'])
        
            for batch_size in [1, 8, 16]:
                base_times = []
                ours_times = []
                jit_times = []
                scaled_time_base = []
                scaled_time_jit = []
                one_times = [1]*7
                
                for seq_len in [64, 128, 256, 384, 512, 768, 1024]:
                # for seq_len in [64, 128]:
                    if avg_seq_len <= 0:
                        avg_seq_len = seq_len
                    low, high = (2 * avg_seq_len - seq_len, seq_len + 1) if 2 * avg_seq_len > seq_len else (0, 2 * avg_seq_len + 1)
                    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
                    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
                    attr_mask                   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), dtype)
                    
                    input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
                    qkv_kernel                  = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
                    qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
                    
                    strided_mask = generate_strided_mask(attr_mask).cuda()
                    fixed_mask = generate_fixed_mask(attr_mask).cuda()
                    if(mask_id == 1):
                        mask = strided_mask
                    else:
                        mask = fixed_mask
                    
                    
                    hidden_states = input_from_tensor
                    layer = 0 
                    qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
                    q, k, v = qkv.chunk(3, dim=-1)
                    q = transpose_for_scores(q, head_num, head_size)
                    k = transpose_for_scores(k, head_num, head_size)
                    v = transpose_for_scores(v, head_num, head_size)
                
                
                    for i in range(warmup_iters + iters):
                        if i == warmup_iters:    
                            torch.cuda.synchronize()
                            t0_start = timeit.default_timer()                   
                        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
                        scores -= 10000.0 * (1.0 - mask.unsqueeze(1))   
                        probs = F.softmax(scores, dim=-1)
                        h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous() 
                        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                        hidden_states1 = h.view(new_context_layer_shape)    
                    torch.cuda.synchronize()
                    t0_end = timeit.default_timer()
                    
                    base_time = (t0_end - t0_start) * 1000 / iters
                    base_times.append(base_time)
                    print("{} bs:{} Seqlen:{} Base time costs:  \t{:.3f} ms / iter".format(mask_name, batch_size, seq_len, base_time))  
                    
                    
                    for i in range(warmup_iters + iters):
                        if i == warmup_iters:    
                            torch.cuda.synchronize()
                            t2_start = timeit.default_timer()   
                        hidden_states3 = Attention_Jit(q, k, v, mask) 
                    torch.cuda.synchronize()
                    t2_end = timeit.default_timer()
                    jit_time = (t2_end - t2_start) * 1000 / iters
                    jit_times.append(jit_time)
                    print("{} bs:{} Seqlen:{} Jit time costs:  \t{:.3f} ms / iter".format(mask_name, batch_size, seq_len, jit_time))  
                    
                    
                    for i in range(warmup_iters + iters):
                        if i == warmup_iters:    
                            torch.cuda.synchronize()
                            t1_start = timeit.default_timer()
                        if(mask_id == 1):
                            hidden_states2 = syncfree_attention.run_syncfree_strided_attention(q, k, v)
                        else:
                            hidden_states2 = syncfree_attention.run_syncfree_fixed_attention(q, k, v)  
                    torch.cuda.synchronize()
                    t1_end = timeit.default_timer()
                    
                    ours_time = (t1_end - t1_start) * 1000 / iters
                    ours_times.append(ours_time)
                    print("{} bs:{} Seqlen:{} Ours time costs:  \t{:.3f} ms / iter\n".format(mask_name, batch_size, seq_len, ours_time))  
            
                scaled_time_base = [base / ours for base, ours in zip(base_times, ours_times)]
                scaled_time_jit = [base / ours for base, ours in zip(jit_times, ours_times)]
                writer.writerow([batch_size] + one_times)
                writer.writerow([batch_size] + scaled_time_jit)
                writer.writerow([batch_size] + scaled_time_base)
                
                            
if __name__ == '__main__':
        
    bert_example()
