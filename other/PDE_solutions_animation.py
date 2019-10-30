import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from sys import argv

 

# Plot time-dependent animations of solution to advection test problems
# Run me like << python PDE_solutions_animation.py 
    
if len(argv) > 1:
    dim = int(argv[1])
else:
    dim = 1
fs = 18


### ----------------------------------- ###
### ------ One spatial dimension ------ ###
### ----------------------------------- ###
if dim == 1:
    nx = 2 ** 7
    nt = nx
    x  = np.linspace(-1, 1, nx+1)

    T  = 2 # Final time
    nt = 100
    t  = np.linspace(0, T, nt) # nt points in  [0,T]

    # The exact solutions for the test problems
    def uexact(t):
        return np.cos(np.pi*(x - t)) * np.exp(np.cos(2*np.pi*t) - 1)

    fig = plt.figure()
    ax  = plt.axes(xlim=(-1, 1), ylim=(-1, 1))

    plot = plt.plot(x, uexact(0), color = "b")    # first image on screen

    # animation function
    def animate(i):
        global plot
        z = uexact(t[i])
        plt.clf()
        plt.axes(xlim=(-1, 1), ylim=(-1, 1))
        plot = plt.plot(x, z, color = "b")
        plt.title("$u(x,t={:.2f})$".format(t[i]), fontsize = fs)
        plt.xlabel("$x$", fontsize = fs)
        return plot

    anim = animation.FuncAnimation(fig, animate, frames=nt, repeat=False, interval = 70)                         
    plt.show()



### ------------------------------------ ###
### ------ Two spatial dimensions ------ ###
### ------------------------------------ ###
elif dim == 2:
    cmap = plt.cm.get_cmap("coolwarm")
    nx = 2 ** 6
    ny = nx
    nt = nx
    x  = np.linspace(-1, 1, nx+1)
    y  = np.linspace(-1, 1, ny+1)
    [X, Y] = np.meshgrid(x,y)

    T  = 2 # Final time
    nt = 100
    t  = np.linspace(0, T, nt) # nt points in  [0,T]

    def uexact(t):
        return np.cos(np.pi*(X - t)) * np.cos(np.pi*(Y - t)) * np.exp(np.cos(4*np.pi*t) - 1)

    fig = plt.figure()
    ax  = plt.axes(xlim=(-1, 1), ylim=(-1, 1))

    cvals = np.linspace(-1,1,14)      # set contour values 
    cont = plt.contourf(X, Y, uexact(0), cvals, cmap = cmap)    # first image on screen
    plt.colorbar()

    # animation function
    def animate(i):
        global cont
        z = uexact(t[i])
        for c in cont.collections:
            c.remove()  # removes only the contours, leaves the rest intact
        cont = plt.contourf(X, Y, z, cvals, cmap = cmap)
        plt.title("$u(x,y,t={:.2f})$".format(t[i]), fontsize = fs)
        plt.xlabel("$x$", fontsize = fs)
        plt.ylabel("$y$", fontsize = fs)
        return cont

    anim = animation.FuncAnimation(fig, animate, frames=nt, repeat=False, interval = 70)                         
    plt.show()