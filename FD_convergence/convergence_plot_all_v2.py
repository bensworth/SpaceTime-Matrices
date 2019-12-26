
# Plot up errors at final time by measuring the differece between the numerical 
# solution and the exact solution.
#
#NOTES:
# Assumes text files are stored in the form:
# <DIR>/U_<TIMEDISC>_U<SPACEDISC>_d<D>_l<SPATIALREFINEMENT>_FD<FD>_pit<PIT>

## python convergence_plot_all_v2.py -dir data/BDF/ -t 31 32 33 -o 1 2 3 -d 1 -lmin 3 3 3 -lmax 7 7 7 -FD 101 -pit 0 



import numpy as np

import matplotlib
import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sys
from sys import argv
from numpy.linalg import norm
import os.path

import argparse



parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-dir','--dir', help = 'Directory of data', required = True)
parser.add_argument('-t','--timeDisc', nargs = "+", help = 'Temporal disc ID', required = True)
parser.add_argument('-o','--spaceDisc', nargs = "+", help = 'Space disc ID', required = True)
parser.add_argument('-lmin','--lmin', nargs = "+", help = 'Min spatial refinement', required = True)
parser.add_argument('-lmax','--lmax', nargs = "+", help = 'Max spatial refinement', required = True)
parser.add_argument('-d','--d', help = 'Spatial dimension', required = True)
parser.add_argument('-FD','--FD', help = 'Finite difference problem ID', required = True)
parser.add_argument('-pit','--pit', help = 'Parallel-in-time v.s. time-stepping flag', required = True)
parser.add_argument('-s','--save', help = 'Save figure', required = False, default = False)
args = vars(parser.parse_args())

print(args)

xlims = [None, None]
ylims = [None, None]
fs = {"fontsize": 18}

colours = ["g", "r", "b", "c"]


# Can uncomment these things if saving a plot...
#ylims = [1e-9, 1]
#ylims = [1e-7, 1]
#xlims = [0.825*2**3, 1.175*2**8]
if int(args["save"]):
    ylims = [1e-7, 1]
    fs = {"fontsize": 18, "usetex": True}


# Template for filenames
def filenametemp(scheme, t, spaceRefine):
    return args["dir"] + "U_" + t + "_U" + args["spaceDisc"][scheme] + "_d" + args["d"] + "_l" + str(spaceRefine) + "_FD" + args["FD"] + "_pit" + args["pit"]


for scheme, t in enumerate(args["timeDisc"]):
    e  = []
    dx = []
    nx = []
    
    for spaceRefine in range(int(args["lmin"][scheme]), int(args["lmax"][scheme])+1):
        
        filename = filenametemp(scheme, t, spaceRefine)
        print(filename)
    
    
        # If output filename exists, open it, otherwise create it
        if os.path.isfile(filename):
            params = {}
            with open(filename) as f:
                for line in f:
                   (key, val) = line.split()
                   params[key] = val
                   
        else:
            sys.exit("What are you doing? I cannot fine the file: ('{}')".format(filename))
    
    
        # Type cast the parameters from strings into their original types
        params["nx"] = int(params["nx"])
        params["dx"] = float(params["dx"])
        params["discerror"] = float(params["discerror"])
    
        e.append(params["discerror"])
        dx.append(params["dx"])
        nx.append(params["nx"])

    # Cast from lists to numpy arrays for plotting, etc.
    e  = np.array(e)
    dx = np.array(dx)
    nx = np.array(nx)
    
    order = int(args["spaceDisc"][scheme])
    e *= np.sqrt(dx)

    # Plot errors for given solves and line indicating expected asymptotic convergence rate
    plt.loglog(nx, 0.5*e[-1]*(nx/float(nx[-1]))**(-order), linestyle = '--', color = 'k')
    plt.loglog(nx, e, label = "T{}+U{}".format(t, order), marker = "o", color = colours[scheme], basex = 2)


axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)
 
# Set string for error norm
T = float(params["dt"]) * int(params["nt"]) # Final time
enormstr = "$\\Vert \\mathbf{{e}}(x,T\\approx {:.1f}) \\Vert_{{L2}}$".format(T) 


plt.legend(fontsize = fs["fontsize"]-3)
plt.xlabel("$n_x$", **fs)
plt.ylabel(enormstr, **fs)

if int(args["pit"]):
    title = "Space-time: "
else:
    title = "Time-stepping: "

title += args["d"] + "D"
title += ", " + args["FD"]
print(title)
plt.title(title, **fs)

if int(args["save"]):    
    # Generate name to save figure with...
    filenameOUT = "plots/" + params["time"] + "/"
    if int(args["pit"]):
        filenameOUT += "spaceTime_"
    else:
        filenameOUT += "timeStepping_"
    
    if int(params["implicit"]):
        filenameOUT += "implicit_"
    else:
        filenameOUT += "explicit_"    
    
    
    filenameOUT += "d" + args["d"] + "_FD" + args["FD"]
    plt.savefig('{}.pdf'.format(filenameOUT), bbox_inches='tight')
plt.show()  




