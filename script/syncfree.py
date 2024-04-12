#  python syncfree1.py 8 1 256 12 64 1
import argparse
import timeit
import torch
import numpy as np
import random
import torch.nn.functional as F
from utils.utils import *
import os
from torch.utils.cpp_extension import load

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

syncfree_attention = load(
    name = "syncfree_attention",
    sources = [os.path.join(path, "../src/syncfree_attention.cu"), os.path.join(path, "../src/syncfree_attention.cpp")],
    verbose = True,
)

@torch.jit.script
def Attention_Jit(q, k, v, mask):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    scores -= 10000.0 * (1.0 - mask.unsqueeze(1))   
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous() 
    new_context_layer_shape = h.size()[:-2] + (q.shape[1]*q.shape[3], )
    return h.view(new_context_layer_shape)    


def bert_example(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    batch_size = args['batch_size']  # 1
    layer_num = args['layer_num']  # 12
    seq_len = args['seq_len']  # 64
    head_num = args['head_num']   # 12
    head_size = args['head_size']   # 64
    mask_id = args['mask_id']   # type of mask: 1-strided, 2-fixed
    avg_seq_len = args['avg_seq_len']  #  -1
    hidden_dim = head_num * head_size
    
    warmup_iters = 10
    iters = 100
    dtype = "fp32"
    
    for key in args:
        print("{:13} : {:5} ".format(key, args[key]))
    print("-"*21, "Argument", "-"*21)
    
    if avg_seq_len <= 0:
        avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1) if 2 * avg_seq_len > seq_len else (0, 2 * avg_seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask                   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), dtype)
    
    lower_triangle_mask = generate_triangle_mask(attr_mask).cuda()
    strided_mask = generate_strided_mask(attr_mask).cuda()
    fixed_mask = generate_fixed_mask(attr_mask).cuda()
    
    if(mask_id == 1): 
        mask_name = 'Strided_mask'
        mask = strided_mask
    else:
        mask_name = 'Fixed_mask'
        mask = fixed_mask
    
    input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
    qkv_kernel                  = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
  
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
        hidden_states0 = h.view(new_context_layer_shape)    
    torch.cuda.synchronize()
    t0_end = timeit.default_timer()
    

    for i in range(warmup_iters + iters):
        if i == warmup_iters:    
            torch.cuda.synchronize()
            t1_start = timeit.default_timer()   
        hidden_states2 = Attention_Jit(q, k, v, mask) 
    torch.cuda.synchronize()
    t1_end = timeit.default_timer()
    
    
    for i in range(warmup_iters + iters):
        if i == warmup_iters:    
            torch.cuda.synchronize()
            t2_start = timeit.default_timer()
        if(mask_id == 1): 
            hidden_states1 = syncfree_attention.run_syncfree_strided_attention(q, k, v)
        else:
            hidden_states1 = syncfree_attention.run_syncfree_fixed_attention(q, k, v)
    torch.cuda.synchronize()
    t2_end = timeit.default_timer()
    
    
    diff = torch.abs(hidden_states0 - hidden_states1)
    print('Mean diff ours: {:.8f}'.format(torch.mean(diff).item()))
    diff2 = torch.abs(hidden_states0 - hidden_states2)
    print('Mean diff jit : {:.8f}'.format(torch.mean(diff2).item()))
    print("Base({}) time costs:  \t{:.3f} ms / iter".format(mask_name, (t0_end - t0_start) * 1000 / iters)) 
    print("Jit ({}) time costs:  \t{:.3f} ms / iter".format(mask_name, (t1_end - t1_start) * 1000 / iters)) 
    print("Fuse({}) time costs:  \t{:.3f} ms / iter".format(mask_name, (t2_end - t2_start) * 1000 / iters)) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, default=1, help='batch size')
    parser.add_argument('layer_num', type=int, default=12, help='number of layers')
    parser.add_argument('seq_len', type=int, default=64, help='sequence length')
    parser.add_argument('head_num', type=int, default=12, help='head number')
    parser.add_argument('head_size', type=int, default=64, help='size per head')
    parser.add_argument('mask_id', type=int, default=0, help='type of mask: 1-strided, 2-fixed')
    parser.add_argument('--avg_seq_len', type=int, default=-1, metavar='NUMBER', help='average sequence length (default: -1)')
    args = parser.parse_args()
    bert_example(vars(args))
