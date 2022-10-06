import matplotlib.pyplot as plt
from IPython import display
from math import sin,cos,pi
plt.ion()

def get_points(r,w):
    ip=[0,0]
    x=[0]
    y=[0]
    for i in range(len(r)):
        x.append(r[i]*cos(w[0]*pi/180)+x[-1])
        y.append(r[i]*sin(w[0]*pi/180)+y[-1])
    return x,y

def plot(r,w):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Moving...')
    plt.xlim([0,10])
    plt.ylim([0,4])
    pt_x,pt_y=get_points(r,w)
    plt.plot(pt_x,pt_y)
    plt.show(block=False)
    plt.pause(1e-10)
    
    


