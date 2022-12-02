import numpy as np

a = np.ones(shape = (10, 20))
b = np.zeros(shape = (10,1))

print(np.concatenate((a, b), axis = 1).shape)