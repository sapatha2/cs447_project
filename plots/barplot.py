import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
df = pd.read_csv('bleu.csv')

fig, ax = plt.subplots(1, 1, figsize = (3,3))
d = df.pivot(index = 'Language', columns='Model', values = 'BLEU4')
d = d.iloc[[2, 3, 4, 0, 1]]
d = d[['Base','Attention','Deep Attention']]
d['Deep Attention'] = d['Deep Attention'] - d['Attention']
d['Attention'] = d['Attention'] - d['Base']
d.columns = ['Base - Base', 'Base - Att', 'Deep - Att']
d.plot.bar(rot = 40, stacked=True, ax = ax, 
        color = {'Base - Base': colors[0], 'Base - Att': colors[1], 'Deep - Att': colors[3]}, 
        edgecolor='k', width=0.75)
plt.xlabel('')
plt.ylabel('BLEU4')
legend = plt.legend()
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_alpha(1.0)
plt.savefig('barplot.pdf',bbox_inches='tight')
