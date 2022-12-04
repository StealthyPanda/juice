from juice import actsigmoid
import numpy as np

zero = np.zeros(shape = (2, 2))

print(actsigmoid(zero) * np.array([[1, 0],[0, 1]]))
print(np.matmul(actsigmoid(zero) , np.array([[1, 0],[0, 1]])))