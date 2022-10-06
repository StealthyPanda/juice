from juice import *
from math import sin, cos

r = [1, 2, 3, 0.5]

def arm(r, theta):
    x, y = 0, 0

    for each in range(len(r)):
        x += (r[each] * cos(theta[each]))
        y += (r[each] * sin(theta[each]))
    
    return (x, y)


point = (0, 0)
def j(w):
    x, y = arm(r, w)
    return (pow( pow((x - point[0]), 2) + pow((y - point[1]), 2) , 0.5) * np.linalg.norm(w))

w0 = [-3 for i in range(len(r))]

w0 = np.array(w0, dtype = np.float32)

w, record = gradientdescent(j, w0, delta = pow(10, -6), alpha = 0.001, maxiterations=pow(10, 4), sequential=True, record = True)
print(w, w0)
print(arm(r, w))

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 10))
ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
ax.plot([point[0]], [point[1]], 'ro')
line, = ax.plot([], [], 'bo-', lw = 3,) 

def init(): 
    line.set_data([], [])
    return line,

def animfunc(i):
    x, y = [0], [0]

    for each in range(len(r)):
        bx, by = 0, 0
        for k in range((each) + 1):
            bx += r[k] * cos(record[i][k])
            by += r[k] * sin(record[i][k])
        x.append(bx)
        y.append(by)
    line.set_data(x, y)
    # line.set('bo-')
    return line,

print(len(record))

anim = FuncAnimation(fig, animfunc, init_func = init,
                     frames = len(record), interval = 1, blit = True)

# anim.save('continuousSineWave.mp4', 
#           writer = 'ffmpeg', fps = 60)

plt.show()