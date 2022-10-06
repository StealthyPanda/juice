import matplotlib.animation as animation 
import matplotlib.pyplot as plt 
import numpy as np 


anim = animation.FuncAnimation(fig, animate, init_func = init, 
                               frames = 500, interval = 20, blit = True) 