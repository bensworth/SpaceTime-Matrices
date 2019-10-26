import numpy as np

import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sys import argv

import pdb
from numpy.linalg import norm
from scipy.sparse import load_npz

# Plot solution to advection test problems
# Run me like << python PDE_solutions_plot.py 
    
if len(argv) > 1:
    dim = int(argv[1])
else:
    dim = 1
    
### ----------------------------------- ###
### ------ One spatial dimension ------ ###
### ----------------------------------- ###
if dim == 1:
    T  = 2 # Final time
    nx = 2 ** 7
    nt = nx
    x  = np.linspace(-1, 1, nx+1)
    x  = x[:-1] # nx points in [-1, 1)
    t  = np.linspace(0, T, nt) # nt points in  [0,T]
    [X, T] = np.meshgrid(x,t)

    # The exact solutions for the test problems
    def uexact(x,t):
        return np.cos(np.pi*(x - t)) * np.exp(np.cos(2*np.pi*t) - 1)

    # The wave speed for the test problems
    def wavespeed(x,t):
        return np.cos(np.pi*(x - t)) * np.exp(-(np.sin(2*np.pi*t)**2)) 

    U     = np.zeros((nx, nt))
    alpha = np.zeros((nx,nt))
    for n in range(0,nt):
        U[n,:]     = uexact(x,t[n])
        alpha[n,:] = wavespeed(x,t[n])


    # Figure settings
    fs = 20
    cmap = plt.cm.get_cmap("coolwarm")
    # Sit there here to enable latex-style font in plots... But very slow...
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['text.usetex'] = True


    ###  --- Solution --- ###
    fig = plt.figure(1)
    levels = np.linspace(np.amin(U, axis = (0,1)), np.amax(U, axis = (0,1)), 200)
    plt.contourf(X, T, U, levels=levels,cmap=cmap)
    plt.colorbar(ticks=np.linspace(np.amin(U), np.amax(U), 7), format='%0.1f')	
    plt.title("$u(x,t)$", fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.ylabel("$t$", fontsize = fs)
    #plt.savefig("solution1D.pdf", bbox_inches='tight')


    ###  --- Wavespeed --- ###
    fig = plt.figure(2)
    levels = np.linspace(np.amin(alpha, axis = (0,1)), np.amax(alpha, axis = (0,1)), 200)
    plt.contourf(X, T, alpha, levels=levels,cmap=cmap)
    plt.colorbar(ticks=np.linspace(np.amin(alpha), np.amax(alpha), 7), format='%0.1f')	
    plt.title("$\\alpha(x,t)$", fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.ylabel("$t$", fontsize = fs)
    #plt.savefig("wavespeed1D.pdf", bbox_inches='tight')

    plt.show()    

### ------------------------------------ ###
### ------ Two spatial dimensions ------ ###
### ------------------------------------ ###
elif dim == 2:
    T  = 0.5 # Final time
    nx = 2 ** 7
    ny = nx
    nt = nx
    x  = np.linspace(-1, 1, nx+1)
    y  = np.linspace(-1, 1, ny+1)
    x  = x[:-1] # nx points in [-1, 1)
    y  = y[:-1] # ny points in [-1, 1)
    [X, Y] = np.meshgrid(x,y)

    # The exact solutions for the test problems
    def uexact(x,y,t):
        return np.cos(np.pi*(x - t)) * np.cos(np.pi*(y - t)) * np.exp(np.cos(4*np.pi*t) - 1)

    # The wave speed for the test problems
    def wavespeed(x,y,t,component):
        if component == 1:
            return np.cos(np.pi*(x - t)) * np.cos(np.pi*y) * np.exp(-(np.sin(2*np.pi*t)**2)) 
        elif component == 2:
            return np.sin(np.pi*x) * np.cos(np.pi*(y - t)) * np.exp(-(np.sin(2*np.pi*t)**2)) 
            

    U      = np.zeros((nx,ny))
    alpha1 = np.zeros((nx,ny))
    alpha2 = np.zeros((nx,ny))
    for j in range(0,ny):
        U[j,:]      = uexact(x,y[j],T)
        alpha1[j,:] = wavespeed(x,y[j],T,1)
        alpha2[j,:] = wavespeed(x,y[j],T,2)


    # Figure settings
    fs = 20
    cmap = plt.cm.get_cmap("coolwarm")
    # Sit there here to enable latex-style font in plots... But very slow...
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['text.usetex'] = True


    ###  --- Solution --- ###
    fig = plt.figure(1)
    levels = np.linspace(np.amin(U, axis = (0,1)), np.amax(U, axis = (0,1)), 200)
    plt.contourf(X, Y, U, levels=levels,cmap=cmap)
    plt.colorbar(ticks=np.linspace(np.amin(U), np.amax(U), 7), format='%0.1f')	
    plt.title("$u(x,y,{})$".format(T), fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.ylabel("$y$", fontsize = fs)
    #plt.savefig("solution2D.pdf", bbox_inches='tight')


    ###  --- Wavespeed: 1st component --- ###
    fig = plt.figure(2)
    levels = np.linspace(np.amin(alpha1, axis = (0,1)), np.amax(alpha1, axis = (0,1)), 200)
    plt.contourf(X, Y, alpha1, levels=levels,cmap=cmap)
    plt.colorbar(ticks=np.linspace(np.amin(alpha1), np.amax(alpha1), 7), format='%0.1f')	
    plt.title("$\\alpha_{{1}}(x,y,{})$".format(T), fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.ylabel("$y$", fontsize = fs)
    #plt.savefig("wavespeed2D_1.pdf", bbox_inches='tight')
    
    ###  --- Wavespeed: 2nd component --- ###
    fig = plt.figure(3)
    levels = np.linspace(np.amin(alpha2, axis = (0,1)), np.amax(alpha2, axis = (0,1)), 200)
    plt.contourf(X, Y, alpha2, levels=levels,cmap=cmap)
    plt.colorbar(ticks=np.linspace(np.amin(alpha2), np.amax(alpha2), 7), format='%0.1f')	
    plt.title("$\\alpha_{{2}}(x,y,{})$".format(T), fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.ylabel("$y$", fontsize = fs)
    #plt.savefig("wavespeed2D_2.pdf", bbox_inches='tight')

    plt.show()   