
import numpy as np
import matplotlib.pyplot as plt

batches = 10000 
epochs = 30
lr_slope = 1e-2 / (batches * epochs)

batch = np.array(range(epochs)) * batches
lr = np.clip(lr_slope * batch, 1e-3, 1e-2)

print (lr_slope * 432000)

plt.plot(batch, lr)
plt.show()

