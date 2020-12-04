import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
df = pd.read_csv('bleu.csv')

fig, ax = plt.subplots(nrows = 4, ncols = 2, sharex = True, figsize = (6, 9))
for i, b in enumerate(['BLEU1','BLEU2','BLEU3','BLEU4']):
    plotdf = df.pivot(index='Language', columns='Model', values = b)[['Deep Attention','Attention','Base']]
    plotdf = plotdf.loc[['Italian','German','French','Spanish','Portugese']]
    plotdf['Deep Attention'] = plotdf['Deep Attention'].values - plotdf['Attention'].values
    plotdf['Attention'] = plotdf['Attention'].values - plotdf['Base'].values

    plotdf[['Attention']].plot(kind = 'bar', ax = ax[i,0], rot = 40, width = 0.75, edgecolor='k', color=colors[0])
    ax[i,0].set_xlabel('')
    ax[i,0].set_ylabel(b)
    ax[i,0].get_legend().remove()
    ax[i,0].set_ylim((5, 15))
    ax[i,0].set_yticks([5, 10, 15])

    plotdf[['Deep Attention']].plot(kind = 'bar', ax = ax[i,1], rot = 40, width = 0.75, edgecolor='k', color=colors[2])
    ax[i,1].set_xlabel('')
    ax[i,1].get_legend().remove()
    ax[i,1].set_ylim((0, 2.2))
    ax[i,1].set_yticks([0, 1, 2])

ax[0,0].set_title('Attention improvements')
ax[0,1].set_title('Deep improvements')
plt.savefig('barplot.pdf',bbox_inches='tight')
plt.close()
