import numpy as np
a = np.array([1,2,3])
b = np.array([1,2,3])
c = np.concatenate((a, b), axis=0)
c = np.expand_dims(c, -1)
print(c.shape)