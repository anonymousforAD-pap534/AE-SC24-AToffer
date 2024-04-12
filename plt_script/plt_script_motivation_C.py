import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


labels = ['cudaStreamSynchronize', 'cudaMemcpyAsync', 'cudaMemcpy', 'cudaLaunchKernel', 'cudaDeviceSynchronize', 'Others']
sizes = [56.9, 24.0, 6.8, 6.5, 1.5, 4.3]
colors = plt.cm.tab20.colors  

custom_font = FontProperties(size=14)


fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, labels=None, startangle=90, colors=colors, 
                                  wedgeprops={'edgecolor': 'white'}, 
                                  autopct='',
                                  textprops=dict(color="black"),
                                  pctdistance = 0.6)

for patch in wedges:
    patch.set_alpha(0.7)  

legend = plt.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), prop=custom_font)
for text in legend.get_texts():
    text.set_fontproperties(custom_font)


plt.subplots_adjust(right=0.5)


plt.axis('equal')


plt.savefig('moti3_improved.pdf', bbox_inches='tight')

