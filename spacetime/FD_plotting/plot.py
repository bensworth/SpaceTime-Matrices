import numpy as np

import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sys import argv

import pdb
from numpy.linalg import norm
from scipy.sparse import load_npz

# Sit there here to enable latex-style font in plots...
plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'cm'
plt.rcParams['text.usetex'] = True

# Run me like << python plot.py <<path to file with solution information>>
# File should contain a list of (at least) the following parameters:
# P
# nt
# dt
# s
# space_dim
# spaceParallel
# nx

# if len(argv) > 1:
#     filename = argv[1]
# else:
#     raise ValueError("A filename must be passed through command line!")

filename = "../ST_class/data/X_FD.txt"

# Read data in and store in dictionary
params = {}
with open(filename) as f:
    for line in f:
       (key, val) = line.split()
       params[key] = val
       
# Type cast the parameters from strings into their original types
params["P"]  = int(params["P"])
params["nt"] = int(params["nt"])
params["s"]  = int(params["s"])
params["dt"] = float(params["dt"])
params["nx"] = int(params["nx"])
params["space_dim"] = int(params["space_dim"])
params["spatialParallel"] = int(params["spatialParallel"])

# Work out which processor uT lives on and where, and get its filename
if not params["spatialParallel"]:
    DOFsPerProc = int((params["s"] * params["nt"]) / params["P"]) # Number of temporal variables per proc
    PuT         = int(np.floor( (params["s"] * (params["nt"]-1)) / DOFsPerProc )) # Index of proc that owns uT
    PuT_DOF0Ind = PuT * DOFsPerProc # Global index of first variable on this proc
    PuT_uTInd   = (params["s"] * (params["nt"]-1)) - PuT_DOF0Ind # Local index of uT on its proc
    print(PuT, PuT_uTInd)
    
    # Filename for data output by processor output processor. Assumes format is <<filename>>.<<index of proc using 5 digits>>
    Ufilename  = filename + "." + "0" * (5-len(str(PuT))) + str(PuT)
    U0filename = filename + "." + "0" * 5
else:
    raise ValueError("Plotting for space parallel not implemented...")
 
print(Ufilename)



# Read all data from the proc
with open(Ufilename) as f:
    dims = f.readline()
dims.split(" ")
dims = [int(x) for x in dims.split()] 
# Get data from lines > 0
uT_dense = np.loadtxt(Ufilename, skiprows = 1, usecols = 1) # Ignore the 1st column since it's just indices..., top row has junk in it we don't need

# Read all data from the proc
with open(U0filename) as f:
    dims = f.readline()
dims.split(" ")
dims = [int(x) for x in dims.split()] 
# Get data from lines > 0
u0_dense = np.loadtxt(U0filename, skiprows = 1, usecols = 1) # Ignore the 1st column since it's just indices..., top row has junk in it we don't need


# Total number of DOFS in space
if params["space_dim"] == 1:
    NX = params["nx"] 
elif params["space_dim"] == 2:
    NX = params["nx"] ** 2
    
# Extract uT from other junk    
uT = np.zeros(NX)
uT[:] = uT_dense[PuT_uTInd*NX:(PuT_uTInd+1)*NX]
u0 = np.zeros(NX)
u0[:] = u0_dense[:NX]

# def uexact(x,t):
#     return np.cos(np.pi*x)  * np.cos(2*np.pi*t)

if params["space_dim"] == 1:
    
    
    
    nx = NX
    x = np.linspace(-1, 1, nx+1)
    x = x[:-1] # nx points in [-1, 1)
    T = params["dt"]  * (params["nt"] - 1)
    
    # for i in range(0,nx):
    #     u0[i] = uexact(x[i],T)
    
    # Compare uT against the exact solution
    print("nx = {}, |u0 - uT| = {:.4e}".format(nx, np.linalg.norm(u0 - uT, np.inf)))
    
    plt.plot(x, u0, linestyle = "--", marker = "o", color = "r")
    plt.plot(x, uT, color = "b")
    
    
    fs = 18
    plt.title("(RK,U-order,$n_x$)=({},{},{})".format(params["timeDisc"], params["space_order"], nx), fontsize = fs)
    plt.ylabel("$u(x,{:.2f})$".format(T), fontsize = fs)
    plt.xlabel("$x$", fontsize = fs)
    plt.show()
    


