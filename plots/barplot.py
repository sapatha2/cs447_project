import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('bleu.csv')

fig, ax = plt.subplots(nrows = 4, ncols = 1, sharex = True, sharey = True, figsize = (3, 9))
for i, b in enumerate(['BLEU1','BLEU2','BLEU3','BLEU4']):
    plotdf = df.pivot(index='Language', columns='Model', values = b)[['Attention','Deep Attention']]
    plotdf = plotdf.loc[['Italian','German','French','Spanish','Portugese']]
    plotdf['Deep Attention'] = plotdf['Deep Attention'].values - plotdf['Attention'].values
    plotdf = plotdf[['Deep Attention']]
    
    plotdf.plot(kind = 'bar', ax = ax[i], rot = 40, width = 0.75, edgecolor='k')
    ax[i].set_xlabel('')
    ax[i].set_ylabel(b)
    ax[i].get_legend().remove()

ax[0].set_ylim((0, 2.2))
ax[0].set_yticks([0, 1, 2])
ax[0].set_title('BLEU improvements')
plt.savefig('barplot.pdf',bbox_inches='tight')
