import numpy as np 
import matplotlib.pyplot as plt

# Shallow en-de, tau = 2e-5
train = [3.097516,1.660124,1.128313,0.690136,0.378768,0.187663,0.087333,0.046830,0.033066,0.028416,0.025627,0.023141,0.021115,0.020595,0.018238,0.017062,0.016710,0.017312,0.015831,0.016700,0.015840,0.016172,0.022082,0.023248,0.022405]
test =  [4.237552,4.135298,4.196503,4.113661,4.065952,4.009482,3.998476,4.131479,4.147719,4.051108,3.958168,3.911585,3.991293,3.923655,3.822002,3.838171,3.809381,3.771868,3.803760,3.769855,3.762888,3.841823,3.783433,3.775214,3.849793]

fig, ax = plt.subplots(ncols = 1, nrows = 2, sharex = True, sharey=False, figsize=(3, 7))
ax[0].plot(train, 'o-')
ax[0].axhline(np.mean(train[-10:]), color='k', ls='--')
ax[0].set_ylabel('Train Loss')

ax[1].plot(test, 'o-')
ax[1].axhline(np.mean(test[-10:]), color='k', ls='--')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Test Loss')

ax[0].set_title(r'EN-DE, $\tau$ = 2e-5')
plt.savefig('plot_loss_en-de_2e-5.pdf',bbox_inches='tight')
plt.close()
