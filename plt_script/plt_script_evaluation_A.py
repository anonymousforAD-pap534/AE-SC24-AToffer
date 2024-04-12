import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.transforms import blended_transform_factory


fontprop = fm.FontProperties(size=16)  

for platform in [3090, 4090]:

    for mask in ['Strided', 'Fixed']: 
        
        df = pd.read_csv('./{}_{}_A_script_result_8.csv'.format(platform, mask))
        bar_width = 0.35
        fact_bar_width = 0.26
        batch_sizes = df['batch_size'].unique()
        sequence_lengths = df.columns[1:].astype(int)
        
        fig, ax = plt.subplots(figsize=(9, 3))
        batch_spacing = len(sequence_lengths) * bar_width * 3+1
        
        color_ours = (255/255, 178/255, 98/255)  
        color_jit = '#91CBFC'  
        color_base = (113/255, 157/255, 199/255)  

        for i, batch_size in enumerate(batch_sizes):
            batch_data = df[df['batch_size'] == batch_size]
            ours_data = batch_data.iloc[0, 1:].tolist()
            jit_data = batch_data.iloc[1, 1:].tolist()
            base_data = batch_data.iloc[2, 1:].tolist()

            # 计算当前 batch_size 的 x 坐标，并减去offset
            x_coords = np.arange(len(sequence_lengths)) * bar_width * 3 + batch_spacing * i * 0.93

            # 绘制 ours 和 base 的条形图
            ax.bar(x_coords - fact_bar_width*1.15, ours_data, fact_bar_width, color=color_ours, label=f'AToffer', zorder = 2)
            ax.bar(x_coords, jit_data, fact_bar_width, color=color_jit, label=f'PyTorch JIT', zorder = 2)
            ax.bar(x_coords + fact_bar_width*1.15, base_data, fact_bar_width, color=color_base, label=f'Native PyTorch', zorder = 2)

        
        xticks = []
        xticklabels = []
        for i in range(len(batch_sizes)):
            start_pos = i * batch_spacing * 0.93
            xticks += list(start_pos + np.arange(len(sequence_lengths)) * bar_width * 3)
            xticklabels += [str(sl) for sl in sequence_lengths]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, fontproperties=fontprop)
        
    
        x_max = max(xticks) + bar_width * 2 
        label_padding = 8.5  
        x_min_adj = min(x_coords) - label_padding
        ax.set_xlim(left=x_min_adj, right=x_max)
        
        ax.set_ylabel('Normalized Time', fontproperties=fontprop)
        plt.yticks(fontproperties=fontprop)
        ax.set_ylim(bottom=0.5, top=3.0)
        ax.grid(axis='y', which = 'major', linestyle='-', linewidth=0.5, color='lightgray', zorder=0)

        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15  
        for i in range(1, len(batch_sizes)):
            line_pos = i * batch_spacing - bar_width * 1.5
            transform = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(line_pos - batch_spacing / 2 - 0.5, ax.get_ylim()[0] - y_offset * 1.7, f'batch_size={batch_sizes[i-1]}', ha='center', fontproperties=fontprop)

        last_batch_pos = (len(batch_sizes) - 1) * batch_spacing
        last_batch_text_x = last_batch_pos + batch_spacing * 0.43 
        ax.text(last_batch_text_x - 0.95, ax.get_ylim()[0] - y_offset * 1.7, f'batch_size={batch_sizes[-1]}', ha='center', fontproperties=fontprop)
        
        
        separator_positions = []
        for i in range(1, len(batch_sizes)):
            separator_positions.append((i - 1) * batch_spacing + (len(sequence_lengths) - 1) * bar_width * 3)

        for sep in separator_positions:
            ax.axvline(x=sep + 0.75 , color='black', linestyle='dashed', linewidth=1.0, alpha=0.5)

        transform = blended_transform_factory(ax.transData, ax.transAxes)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fontprop.set_size(16)
        ax.legend(by_label.values(), by_label.keys(), prop=fontprop, loc='upper right', framealpha=0.6, ncol = 3)


        plt.tight_layout(pad = 1.0)
        filename = '{}_{}_evaA_result_3group.pdf'.format(platform, mask)
        plt.savefig(filename)
