import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('bleu.csv')
df['Params'] = df['parms']
df = df[['Language','Model','Params','BLEU1','BLEU2','BLEU3','BLEU4']]
for b in ['BLEU1','BLEU2','BLEU3','BLEU4']:
    df[b] = np.around(df[b].values, 1)
groups = df.groupby(by='Language')

s = ''
for g in ['Italian','German','French','Spanish','Portugese']:
    d = df[df['Language']==g]
    s += '\\textbf{' + g + '} \\\\ \n'
    s += d.drop(columns=['Language']).to_latex(index = False) + ' \\\\ \\\\ \n'

f = open('table.tex','w')
f.write(s)
f.close()
