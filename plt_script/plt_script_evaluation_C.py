import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置数据文件路径
data_paths = {
    'rd': './search_result/{}_rd_seq{}_Time.csv',
    'sa': './search_result/{}_sa_seq{}_Time.csv',
    'ours': './search_result/{}_ours_seq{}_Time.csv'
}

custom_ticks = [1, 16, 32, 48, 64, 80, 96, 112, 128]
sample_points = [1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]

for mask in [1, 2]:
    for seqlen in [64, 128, 256, 512]:
        df_rd = pd.read_csv(data_paths['rd'].format(mask, seqlen))
        df_sa = pd.read_csv(data_paths['sa'].format(mask, seqlen))
        df_ours = pd.read_csv(data_paths['ours'].format(mask, seqlen))

        rd_filtered = df_rd.set_index('iteration').reindex(sample_points).reset_index()
        sa_filtered = df_sa.set_index('iteration').reindex(sample_points).reset_index()
        ours_filtered = df_ours.set_index('iteration').reindex(sample_points).reset_index()

        rd_filtered.dropna(inplace=True)
        sa_filtered.dropna(inplace=True)
        ours_filtered.dropna(inplace=True)

        rd_data = rd_filtered[['round_1', 'round_2', 'round_3', 'round_4', 'round_5']].values
        sa_data = sa_filtered[['round_1', 'round_2', 'round_3', 'round_4', 'round_5']].values
        ours_data = ours_filtered[['round_1', 'round_2', 'round_3', 'round_4', 'round_5']].values

        rd_means = rd_data.mean(axis=1)
        sa_means = sa_data.mean(axis=1)
        ours_means = ours_data.mean(axis=1)

        color_rand = (113/255, 157/255, 199/255) 
        ours_color = (255/255, 178/255, 98/255)
        sa_color = '#C5E0B4'  
        
        fig, ax = plt.subplots(figsize=(6, 4))
        

        line_rand = ax.plot(sample_points, rd_means, marker='o', linestyle='-', label='Random Search', color = color_rand)
        line_group = ax.plot(sample_points, sa_means, marker='o', linestyle='-', label='SA Search', color = sa_color)
        line_group = ax.plot(sample_points, ours_means, marker='o', linestyle='-', label='AToffer Search', color = ours_color)


        prop = fm.FontProperties(size=18)

        plt.xticks(custom_ticks, fontproperties=prop)
        plt.yticks(fontproperties=prop)
        ax.set_xlabel('Iteration', fontproperties=prop)
        ax.set_ylabel('Min. Execution Time (ms)', fontproperties=prop)


        ax.text(0.68, 0.43, 'seq_len = {}'.format(seqlen), transform=ax.transAxes, fontproperties=prop)


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        prop.set_size(17)
        legend = ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.0, 1.0),
                        prop=prop, framealpha=0.5)  



        filename = '{}_seq{}_sampled.pdf'.format(mask, seqlen)
        plt.tight_layout(pad = 1.0)
        plt.savefig(filename)

