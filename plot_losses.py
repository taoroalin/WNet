import matplotlib.pyplot as plt
import numpy as np

n_cut_losses = np.load('n_cut_losses_10_epochs.npy')
rec_losses = np.load('rec_losses_10_epochs.npy')
plt.plot(n_cut_losses)
plt.plot(rec_losses)
plt.show()