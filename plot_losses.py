import matplotlib.pyplot as plt
import numpy as np

n_cut_losses = np.load('n_cut_losses.npy')
rec_losses = np.load('rec_losses.npy')
print(n_cut_losses)
print(rec_losses)
plt.plot(n_cut_losses)
plt.plot(rec_losses)
plt.show()