# Plot up errors at final time by measuring the differece between the solution found with AIR and the exact solution.
import numpy as np

import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sys import argv

import pdb
from numpy.linalg import norm
from scipy.sparse import load_npz
import os.path


if len(argv) > 1:
    filename = argv[1]
    print("Reading data from: " + filename)
else:
    raise ValueError("Must pass the name of a numpy file...")


# If output filename exists, open it, otherwise create it
if os.path.isfile(filename):
    globalList = list(np.load(filename))
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
    
colours = ["k", "r", "b", "c"]        
plt.loglog(nx, 0.5*einf[-1]*(nx/float(nx[-1]))**(-order), label = "$\\mathcal{{O}}(\\Delta x^{})$".format(order), linestyle = '--', color = colours[0])
plt.loglog(nx, e1, label = "$\\Vert e \\Vert_1$", marker = "o", color = colours[1])
plt.loglog(nx, e2, label = "$\\Vert e \\Vert_2$", marker = "o", color = colours[2])
plt.loglog(nx, einf, label = "$\\Vert e \\Vert_{{\\infty}}$", marker = "o", color = colours[3])

fs = 18
plt.legend(fontsize = fs)
plt.title("RK{}+U{}".format(globalList[0]["timeDisc"], order), fontsize = fs)
plt.xlabel("$n_x$", fontsize = fs)
#plt.savefig('RK{}_U{}.pdf'.format(globalList[0]["timeDisc"], order), bbox_inches='tight')
plt.savefig('{}.pdf'.format(filename), bbox_inches='tight')
plt.show()    


