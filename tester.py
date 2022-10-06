from juice import *
from math import sin, cos
from visualize import plot

r = [1, 0.5, 0.7, 0.2, 1, 0.1, 1.5]

def arm(r, theta):
    x, y = 0, 0

    for each in range(len(r)):
        x += (r[each] * cos(theta[each]))
        y += (r[each] * sin(theta[each]))
    
    return (x, y)


point = (1, 2)
def j(w):
    x, y = arm(r, w)
    return pow( pow((x - point[0]), 2) + pow((y - point[1]), 2) , 0.5)

w0 = [0 for i in range(7)]

w0 = np.array(w0, dtype = np.float32)

w = gradientdescent(j, w0, delta = pow(10, -6), alpha = 0.01, maxiterations=pow(10, 5), sequential=True)

for i in range(pow(10,5)):
    print(i)
    plot(r,next(w))
print(w, w0)
print(arm(r, w))