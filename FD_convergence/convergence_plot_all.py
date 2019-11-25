
# Plot up errors at final time by measuring the differece between the numerical 
# solution and the exact solution.
#
#NOTES: 
# -Pass through the command line the names of the data files to plot on the 
#   same set of axes. E.g., all 1D solves using implicit time-stepping would be
# >> python convergence_plot_all.py data/U_RK211_U1_d1_FD2_pit0.npy  
#                                   data/U_RK222_U2_d1_FD2_pit0.npy 
#                                   data/U_RK233_U3_d1_FD2_pit0.npy  
#                                   data/U_RK254_U4_d1_FD2_pit0.npy 

import numpy as np

import matplotlib
import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sys import argv
from numpy.linalg import norm
import os.path

xlims = [None, None]
ylims = [None, None]
fs = {"fontsize": 18}


# Can uncomment these things if saving a plot...
#ylims = [1e-9, 1]
#ylims = [1e-7, 1]
#xlims = [0.825*2**3, 1.175*2**8]
#fs = {"fontsize": 18, "usetex": True}


#enorm = "1"
#enorm = "2"
enorm = "inf"

if len(argv) > 1:
    filenames = []
    for filename in argv[1:]:
        filenames.append(filename)
    #print("Reading data from: " + filename)
else:
    raise ValueError("Must pass the name of a numpy file...")

colours = ["g", "r", "b", "c"]
for count, filename in enumerate(filenames):
    # If output filename exists, open it, otherwise create it
    if os.path.isfile(filename):
        globalList = list(np.load(filename, allow_pickle = True))
    else:
        print("What are you doing?! I don't know what this ('{}') file is".format(filename))

    e1    = []
    e2    = []
    einf  = []
    dx    = []
    nx    = []
    order = int(globalList[0]["space_order"])

    for solve in range(0, len(globalList)):
        e1.append(globalList[solve]["e1"])
        e2.append(globalList[solve]["e2"])
        einf.append(globalList[solve]["einf"])
        dx.append(globalList[solve]["dx"])
        nx.append(globalList[solve]["nx"])
        
    e1   = np.array(e1)
    e2   = np.array(e2)
    einf = np.array(einf)
    dx   = np.array(dx)
    nx   = np.array(nx)
        
    
    if enorm == "inf":
        e = einf
    elif enorm == "2":
        e = e2
    elif enorm == "1":
        e = e1
    
    # Plot errors for given solves and line indicating expected asymptotic convergence rate
    plt.loglog(nx, 0.5*e[-1]*(nx/float(nx[-1]))**(-order), linestyle = '--', color = 'k')
    plt.loglog(nx, e, label = "T{}+U{}".format(globalList[0]["timeDisc"], order), marker = "o", color = colours[count], basex = 2)


axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

# Set string for error norm
if enorm == "inf":
    enormstr = "$\\Vert \\mathbf{{e}}(x,T) \\Vert_{{\\infty}}$"
elif enorm == "2":
    enormstr = "$\\Vert \\mathbf{{e}}(x,T) \\Vert_{{2}}$"
elif enorm == "1":
    enormstr = "$\\Vert \\mathbf{{e}}(x,T) \\Vert_{{1}}$"


plt.legend(fontsize = fs["fontsize"]-3)
plt.xlabel("$n_x$", **fs)
plt.ylabel(enormstr, **fs)
T = globalList[0]["dt"]  * (globalList[0]["nt"] - 1) # Final time
if globalList[0]["pit"]:
    title = "Space-time:\t"
else:
    title = "Time-stepping:\t"

title += str(globalList[0]["space_dim"]) + "D, $T_{{\\rm f}}\\approx{:.2f}$".format(T)
plt.title(title, **fs)

# Generate name to save figure with...
filenameOUT = "plots/" + globalList[0]["time"] + "/"
if globalList[0]["pit"]:
    filenameOUT += "spaceTime_"
else:
    filenameOUT += "timeStepping_"

if int(globalList[0]["implicit"]):
    filenameOUT += "implicit_"
else:
    filenameOUT += "explicit_"    

filenameOUT += "d" + str(globalList[0]["space_dim"]) + "_FD" + str(globalList[0]["problemID"])
#plt.savefig('{}.pdf'.format(filenameOUT), bbox_inches='tight')
plt.show()  

  


