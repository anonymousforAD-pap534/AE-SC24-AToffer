import matplotlib.font_manager as font_manager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font = font_manager.FontProperties()

data = pd.DataFrame({
    'seqlen': [64, 128, 256, 384, 512, 768, 1024],
    'strided': [2063.6, 2415.8, 3320.1, 4682.9, 6105.9, 9841.8, 14128.8],
    'fixed': [2063.6, 2413.0, 3419.5, 4691.4, 6187.5, 9694.0, 13624.0],
})

partA_time_s = 285.5 / 1000  # ms -> s

data['partA_strided_percent'] = partA_time_s / data['strided']
data['partA_fixed_percent'] = partA_time_s / data['fixed']

data[['partA_strided_percent', 'partA_fixed_percent']] *= 100


fig, ax = plt.subplots(figsize=(9, 4))

color1_stride = '#C5E0B4'  
color2_fixed = (255/255, 178/255, 98/255)  

width = 0.35
x_values = data['seqlen'].values.tolist()
rects1 = ax.bar(np.arange(len(x_values)) - width * 0.6, data['partA_strided_percent'], width, label='Strided Mask', color=color1_stride, zorder = 2)
rects2 = ax.bar(np.arange(len(x_values)) + width * 0.6, data['partA_fixed_percent'], width, label='Fixed Mask', color=color2_fixed, zorder = 2)


yticks_custom = [0 + i * 0.004 for i in range(int(0.014 / 0.004) + 1)]
ax.yaxis.set_ticks(yticks_custom)
ax.grid(axis='y', which = 'major', linestyle='-', linewidth=0.5, color='lightgray', zorder=0)



font.set_size(16)
ax.set_xticks(np.arange(len(x_values)))
ax.set_xticklabels(x_values, fontproperties=font)
plt.yticks(fontproperties=font)


# font.set_size(16)
ax.set_xlabel('Sequence Length', fontproperties=font)
ax.set_ylabel('Time Proportion (%)', fontproperties=font)
ax.legend(prop=font)

plt.tight_layout(pad = 3.0)
plt.savefig('evalution_search-cost.pdf')