from juice import *



quad = lambda x: x[0]**2

x0 = np.array([9], dtype= np.float32)

print(quad(x0))

x = gradientdescent(
    quad, x0, alpha=0.5, maxiterations=pow(10, 6)
)
print(x)