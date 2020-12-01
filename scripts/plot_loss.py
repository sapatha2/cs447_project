import numpy as np 
import matplotlib.pyplot as plt

# Shallow en-de, tau = 1e-4
train = [1.261236, 0.120570, 0.035911, 0.240032, 0.063245, 0.04445, 0.036736, 0.031693, 0.158925, 0.109270]
test = [5.012477, 4.689450, 4.602453, 4.671889, 4.609075, 4.593455, 4.597940, 4.591094, 4.738880, 4.635400]

fig, ax = plt.subplots(ncols = 1, nrows = 2, sharex = True, sharey=False, figsize=(3, 7))
ax[0].plot(train, 'o-')
ax[0].axhline(np.mean(train[-5:]), color='k', ls='--')
ax[0].set_ylabel('Train Loss')

ax[1].plot(test, 'o-')
ax[1].axhline(np.mean(test[-5:]), color='k', ls='--')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Test Loss')

ax[0].set_title(r'EN-DE, $\tau$ = 1e-4')
plt.savefig('plot_loss_en-de_1e-4.pdf',bbox_inches='tight')
plt.close()
