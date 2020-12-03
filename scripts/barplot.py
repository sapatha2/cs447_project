import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
        'Language':  ['Spanish'] * 3 + ['French'] * 3 + ['Portugese'] * 3 + ['Italian'] * 3 + ['German'] * 3,
        'Model': ['Base', 'Attention', 'Deep Attention'] * 5,
        'BLEU1': [65.336, 71.209, 72.299,
                  58.261, 67.889, 69.058,
                  64.503, 73.920, 74.541, 
                  62.758, 72.786, 74.061,     
                  61.709, 68.164, 69.502,
        ],
        'BLEU2': [48.556, 57.435, 58.727,
                  40.780, 53.907, 55.411,
                  48.221, 61.357, 62.273, 
                  46.850, 60.997, 62.757, 
                  42.844, 51.383, 53.033,
        ],
        'BLEU3': [35.846, 45.270, 46.777, 
                  29.350, 42.483, 44.059,
                  36.353, 50.156, 51.303, 
                  36.379, 51.073, 53.096, 
                  30.862, 39.427, 41.214,
        ],
        'BLEU4': [28.290, 36.956, 38.362,
                  23.052, 34.856, 36.338,
                  29.310, 42.165, 43.417,
                  29.542, 43.221, 45.296,
                  24.540, 32.279, 34.009,
        ],
        'parms': [478672, 544593, 643665,
                  560455, 626376, 725448, 
                  539781, 605702, 704774, 
                  765210, 831131, 930203, 
                  635019, 700940, 800012],
        'corpus': [114550] * 3 + [162842] * 3 + [158723] * 3 + [334008] * 3 + [204544] * 3,
})

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
