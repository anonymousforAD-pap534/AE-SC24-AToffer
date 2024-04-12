import argparse
import math
import torch
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.font_manager as font_manager

def set_dtype(ts: torch.Tensor, dtype: str):
    if dtype == "fp32":
        return ts.float()
    elif dtype == "fp16":
        return ts.half()
    raise RuntimeError(f"Unsupported dtype {dtype}")

def sequence_mask(lengths, max_len=None, is_2d=True):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if is_2d:
        return mask
    else:
        mask = mask.view(-1, 1, 1, max_len)
        m2 = mask.transpose(2, 3)
        return mask * m2

def transpose_for_scores(x, n_heads, head_size):
    
    new_x_shape = x.size()[:-1] + (n_heads, head_size)
    x = x.view(new_x_shape)

    return x.permute(0, 2, 1, 3)


def generate_strided_mask(attr_mask):
    # gernerate stride mask
    stride_step = int(math.sqrt(attr_mask.shape[1]))
    seq_len = attr_mask.shape[1]
    strided_mask = torch.zeros_like(attr_mask)
    
    for batch in range(strided_mask.shape[0]):
        for i in range(seq_len):
            for j in range(i+1):
                if((i - j) % stride_step == 0):
                    strided_mask[batch, i, j] = 1.0  
                if(j > i - stride_step):
                    strided_mask[batch, i, j] = 1.0  
    return strided_mask

def generate_fixed_mask(attr_mask):
    # gernerate stride mask
    fixed_step = int(math.sqrt(attr_mask.shape[1]))
    seq_len = attr_mask.shape[1]
    fixed_mask = torch.zeros_like(attr_mask)
    
    for batch in range(fixed_mask.shape[0]):
        for i in range(seq_len):
            for j in range(i+1):
                if(j % fixed_step == fixed_step-1):
                    fixed_mask[batch, i, j] = 1.0  
                if(j > i + (j % fixed_step) - fixed_step):
                    fixed_mask[batch, i, j] = 1.0  
    return fixed_mask

def seqlen_to_mask(lengths, max_len):
    batch_size = lengths.numel()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask

def analyze_tensor(maskname, tensor, tensor_name="Tensor", threshold=None, threshold_ratio=0.1):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    dimensions = tensor.size()
    min_value = tensor.min().item()
    mean_value = tensor.mean().item()
    max_value = tensor.max().item()
    if threshold == None:
        threshold = min_value + (max_value - min_value)*threshold_ratio 
    
    stats = {
        "Dimensions": dimensions,
        "Number of Elements": tensor.numel(),
        "Number of Zero Elements": (tensor == 0).sum().item(),
        "Min, Mean, Max": (min_value, mean_value, max_value),
        "Non-zero Element Ratio": (tensor != 0).sum().item() / tensor.numel(),
        "Elements Below Threshold Ratio": (tensor.abs() < threshold).sum().item() / tensor.numel(),
    }

    # Print the tensor name
    print(f"Tensor Name: {tensor_name}")

    i = 0
    # Print the statistics in a formatted way
    for key, value in stats.items():
        if key == "Dimensions":
            print(f"{key}: {value} ({' x '.join(map(str, value))})")
        elif key == "Min, Mean, Max":
            print("{:30} : ({:.2f}, {:.2f}, {:.2f})".format(key, min_value, mean_value, max_value))
        else:
            if i > 2:
                print("{:30} : {:8.2f}%".format(key, value * 100))
            else:
                print("{:30} : {:8}".format(key, value))
        i += 1
    print("-" * 40)
    
    font = font_manager.FontProperties()
    font.set_size(28)
    
    
    # Plot and save the histogram
    total_elements = len(tensor.flatten().cpu().numpy())
    bins=[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]
    
    hist, bins = np.histogram(tensor.flatten().cpu().numpy(), bins=bins, range=(0, 1.0))
    ratios = hist / total_elements
    plt.figure(figsize=(7, 6))
    plt.bar(bins[:-1], ratios, width=np.diff(bins), align='edge')

    plt.xlabel("Element Value", fontproperties=font)
    plt.ylabel("Ratio", fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.xticks(fontproperties=font)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    

    plt.tight_layout(pad=1.0)
    plt.savefig("{}_Tensor_Histogram.pdf".format(maskname))
    


def plot_heatmap(maskname, tensor_data, title):
    tensor_values = tensor_data.cpu().numpy()
    heatmap_data = tensor_values[0] 
    
    font = font_manager.FontProperties()
    font.set_size(20)

    plt.imshow(heatmap_data, cmap='Blues', aspect='auto', vmin=heatmap_data.min(), vmax=heatmap_data.max())
    cbar = plt.colorbar() 
    cbar.ax.yaxis.set_tick_params(labelsize=20)  

    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font)  
        
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.xlabel('Sequence Column', fontproperties=font)
    plt.ylabel('Sequence Row', fontproperties=font)

    
    plt.gca().set_aspect('equal', adjustable='box')  

    plt.tight_layout(pad=2.0)
    plt.savefig('{}_Heatmap_{}.pdf'.format(maskname, title))
    

def plot_heatmap_new(maskname, tensor_data, title):
    tensor_values = tensor_data.cpu().numpy()
    tensor_values = tensor_values[0] 
    

    block_size = 32
    heatmap_data = np.zeros((512 // block_size, 512 // block_size)) 
    for row_block in range(heatmap_data.shape[0]):
        for col_block in range(heatmap_data.shape[1]):
            block = tensor_values[row_block * block_size:(row_block + 1) * block_size, col_block * block_size:(col_block + 1) * block_size]
            non_zero_count = np.count_nonzero(block)
            total_elements = block_size ** 2
            heatmap_data[row_block, col_block] = non_zero_count / total_elements * 100  

    font = font_manager.FontProperties()
    font.set_size(14)


    plt.imshow(heatmap_data, cmap='Blues', aspect='auto', vmin=0, vmax=100)  
    
    cbar = plt.colorbar(ticks=np.arange(0, 101, 20))  
    cbar.ax.set_yticklabels([f'{i}%' for i in np.arange(0, 101, 20)], fontproperties=font)  

    plt.xticks(fontproperties=font, ticks=np.arange(0, 16, 2))
    plt.yticks(fontproperties=font, ticks=np.arange(0, 16, 2))

    plt.xlabel('Block Columns', fontproperties=font)
    plt.ylabel('Block Rows', fontproperties=font)

    plt.gca().set_aspect('equal', adjustable='box')  

    plt.tight_layout(pad=2.0)
    plt.savefig(f'{maskname}_Heatmap_{title}.pdf')




def bert_example(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    batch_size = 1
    layer_num = 1
    seq_len = args['seq_len']  
    head_num = 12
    head_size = 64
    avg_seq_len =   -1
    threshold_ratio = 0.1  
    mask_id = args['mask_id']  
    hidden_dim = head_num * head_size
    dtype = "fp32"
    
    for key in args:
        print("{:13} : {:5} ".format(key, args[key]))
    

    if avg_seq_len <= 0:
        avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1) if 2 * avg_seq_len > seq_len else (0, 2 * avg_seq_len + 1)
    
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)    
    attr_mask                   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), dtype)

    strided_mask = generate_strided_mask(attr_mask).cuda()
    fixed_mask = generate_fixed_mask(attr_mask).cuda()
    maskname = ''
    

    if mask_id == 1:
        maskname = 'Strided'
        mask = strided_mask
    else:
        maskname = 'Fixed'
        mask = fixed_mask
    
    

    input_from_tensor           = torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda().float()
    qkv_kernel                  = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        

    warmup_iters = 0
    iters = 1
    for i in range(warmup_iters + iters):

        hidden_states = input_from_tensor
        for layer in range(layer_num):
            input_tensor = hidden_states
             
            qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]

        
            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)



            # ------------------------------------------------------------- Attention start
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
            scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
            
            probs = F.softmax(scores, dim=-1) 
            # analyze_tensor(maskname, probs, "Sample2.1 ------ probs: Scores after softmax", threshold_ratio)
            # plot_heatmap_new(maskname, probs[0], 'probs')
            plot_heatmap(maskname, probs[0], 'probs')
            
            h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
            
                    
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.view(new_context_layer_shape)
            # ------------------------------------------------------------ Attention End                

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_len', type=int, default=64, help='sequence length')
    parser.add_argument('mask_id', type=int, default=1, help='maskid')
    args = parser.parse_args()
    
    print(matplotlib.get_cachedir())
    
    bert_example(vars(args))