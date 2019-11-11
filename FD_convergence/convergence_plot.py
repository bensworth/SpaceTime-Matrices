# Plot up errors at final time by measuring the differece between the numerical 
# solution and the exact solution.
#
#NOTES: 
# -Pass the name of a single data set through the command line

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

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['text.usetex'] = True
fontSpec = {"fontsize": 18, 'fontweight': 'medium'}
#rc('font.family=serif')
#rc('text', usetex=True)

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

print("einf = ")
print(einf)
#print(nx)
    
colours = ["k", "r", "b", "c"]        
plt.loglog(nx, 0.5*einf[-1]*(nx/float(nx[-1]))**(-order), label = "$\\mathcal{{O}}(\\Delta x^{})$".format(order), linestyle = '--', color = colours[0])
plt.loglog(nx, e1, label = "$\\Vert \\mathbf{{e}} \\Vert_1$", marker = "o", color = colours[1])
plt.loglog(nx, e2, label = "$\\Vert \\mathbf{{e}} \\Vert_2$", marker = "o", color = colours[2])
plt.loglog(nx, einf, label = "$\\Vert \\mathbf{{e}} \\Vert_{{\\infty}}$", marker = "o", color = colours[3], basex = 2)

fs = 18

plt.legend(fontsize = fs)
plt.title("RK{}+U{}".format(globalList[0]["timeDisc"], order), **fontSpec)
plt.xlabel("$n_x$", **fontSpec)
#plt.savefig('RK{}_U{}.pdf'.format(globalList[0]["timeDisc"], order), bbox_inches='tight')
#plt.savefig('{}.pdf'.format(filename), bbox_inches='tight')


plt.show()  


# einf_matlab = np.array([5.7442e-01,1.9988e-01,4.9921e-02,3.6126e-03,2.2710e-04,1.4210e-05,8.8943e-07,5.5632e-08,3.4777e-09,2.1752e-10])
# 
# print(nx)
# 
# colours = ["k", "r", "b", "c"]        
# plt.loglog(nx, 0.5*einf_matlab[-1]*(nx/float(nx[-1]))**(-order), label = "$\\mathcal{{O}}(\\Delta x^{})$".format(order), linestyle = '--', color = colours[0])
# plt.loglog(nx, einf_matlab, label = "MATLAB: $\\Vert e \\Vert_{{\\infty}}$", marker = "o", color = colours[1])
# plt.loglog(nx, einf, label = "C++: $\\Vert e \\Vert_{{\\infty}}$", marker = "x", linestyle = '--', color = colours[2], basex=2)
# 
# fs = 18
# plt.legend(fontsize = fs)
# plt.title("Time-stepping, RK{}+U{}".format(globalList[0]["timeDisc"], order), fontsize = fs)
# plt.xlabel("$n_x$", fontsize = fs)
# #plt.savefig('RK{}_U{}.pdf'.format(globalList[0]["timeDisc"], order), bbox_inches='tight')
# plt.savefig('huh_v2.pdf'.format(filename), bbox_inches='tight')
# plt.show()    


