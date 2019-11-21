# ---------------------------------- #
# -------------- README ------------ #
# ---------------------------------- #
# This is more or less the same script as used to plot the solutions 
# of FD discretizations, except here we don't plot them, we just 
# compute their errors against the exact solution and save them to disc
#
# Pass in data for a specific solve and it'll save it with other data from solves
# of the same problem at different spatial resolutions...


import numpy as np

import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sys import argv

from numpy.linalg import norm
from scipy.sparse import load_npz
import os.path


# Pass in name of file that we're to read data from
if len(argv) > 1:
    filenameIN = argv[1]
    print("Reading data from: " + filenameIN)
else:
    raise ValueError("You need to pass me the name of a file to read data from...")

# Read data in and store in dictionary
params = {}
with open(filenameIN) as f:
    for line in f:
       (key, val) = line.split()
       params[key] = val
       
# Type cast the parameters from strings into their original types
params["pit"]             = int(params["pit"])
params["P"]               = int(params["P"])
params["nt"]              = int(params["nt"])
params["dt"]              = float(params["dt"])
params["problemID"]       = int(params["problemID"])
params["nx"]              = int(params["nx"])
params["space_dim"]       = int(params["space_dim"])
params["space_refine"]    = int(params["space_refine"])
params["spatialParallel"] = int(params["spatialParallel"])

# If not using RK scheme set s=1 so data can be read in by common method
if "s" in params:
    params["s"] = int(params["s"])
else:
    params["s"] = 1

# Total number of DOFS in space
if params["space_dim"] == 1:
    NX = params["nx"]
    params["dx"] = 2/params["nx"] 
elif params["space_dim"] == 2:
    NX = params["nx"] ** 2
    params["dx"] = 2/params["nx"] 
    params["dy"] = 2/params["nx"] 
    #NX = params["nx"] * (params["nx"] + 7)

# Create filenameIN for outout of data but remove reference to spatial refinement
#filenameOUT = filenameIN.replace("U_", "data_") # Remove "U_" from start
filenameOUT = filenameIN.replace("_l"+str(params["space_refine"]), "") + ".npy"


# If output filenameIN exists, open it, otherwise create it
if os.path.isfile(filenameOUT):
    globalList = list(np.load(filenameOUT))
    print("The file '" + filenameOUT + "' already exists, I'm appending to it...")
else:
    globalList = list([])
    print("The file '" + filenameOUT + "' does not exist, I'm creating it...")

# Pass me error vector/matrix and I'll compute its norm and then save it to disc...
def save_error(e):
    e = e.flatten()
    if params["space_dim"] == 1:
        e1   = np.linalg.norm(e, 1) * (2/nx)
        e2   = np.linalg.norm(e, 2) * np.sqrt(2/nx)
    elif params["space_dim"] == 2: # Assume dy == dx
        e1   = np.linalg.norm(e, 1) * (2/nx) * (2/nx)
        e2   = np.linalg.norm(e, 2) * np.sqrt(2/nx) * np.sqrt(2/nx)
    einf = np.linalg.norm(e, np.inf)
    
    # Add errors to params dict
    params["e1"] = e1
    params["e2"] = e2
    params["einf"] = einf
    print("nx={}, nt={}:\t e1={:.2e}, e2={:.2e}, einf={:.2e}\n".format(params["nx"], params["nt"], e1, e2, einf))
    # Append params to existing list
    globalList.append(params)
    # Save list
    np.save(filenameOUT, globalList)




#############################
#  --- Parallel in time --- #
#############################
if (params["pit"] == 1):
    ### ----------------------------------------------------------------------------------- ###
    ### --- NO SPATIAL PARALLELISM: Work out which processor uT lives on and extract it --- ###
    if not params["spatialParallel"]:
        DOFsPerProc = int((params["s"] * params["nt"]) / params["P"]) # Number of temporal variables per proc
        PuT         = int(np.floor( (params["s"] * (params["nt"]-1)) / DOFsPerProc )) # Index of proc that owns uT
        PuT_DOF0Ind = PuT * DOFsPerProc # Global index of first variable on this proc
        PuT_uTInd   = (params["s"] * (params["nt"]-1)) - PuT_DOF0Ind # Local index of uT on its proc
        
        # filenameIN for data output by processor output processor. Assumes format is <<filenameIN>>.<<index of proc using 5 digits>>
        Ufilename  = filenameIN + "." + "0" * (5-len(str(PuT))) + str(PuT)

        # Read all data from this proc
        with open(Ufilename) as f:
            dims = f.readline()
        dims.split(" ")
        dims = [int(x) for x in dims.split()] 
        # Get data from lines > 0
        uT_dense = np.loadtxt(Ufilename, skiprows = 1, usecols = 1) # Ignore the 1st column since it's just indices..., top row has junk in it we don't need
            
        # Extract uT from other junk    
        uT = np.zeros(NX)
        uT[:] = uT_dense[PuT_uTInd*NX:(PuT_uTInd+1)*NX]


    ### ---------------------------------------------------------------------------------------- ###
    ### --- SPATIAL PARALLELISM: Work out which processors uT lives on extract it from them ---  ###
    # Note that ordering is preserved...
    else:
        params["p_xTotal"] = int(params["p_xTotal"])
        
        # Index of proc holding first component of uT
        PuT0 = (params["nt"]-1) * params["s"] * params["p_xTotal"]     
        
        # Get names of all procs holding uT data
        PuT = []
        for P in range(PuT0, PuT0 + params["p_xTotal"]):
            PuT.append(filenameIN + "." + "0" * (5-len(str(P))) + str(P))
        
        uT = np.zeros(NX)
        ind = 0
        for count, Ufilename in enumerate(PuT):
            #print(count, NX)
            # Read all data from the proc
            with open(Ufilename) as f:
                dims = f.readline()
            dims.split(" ")
            dims = [int(x) for x in dims.split()] 
            # Get data from lines > 0
            temp = np.loadtxt(Ufilename, skiprows = 1, usecols = 1) # Ignore the 1st column since it's just indices..., top row has junk in it we don't need
            DOFsOnProc = temp.shape[0]
            uT[ind:ind+DOFsOnProc] = temp
            ind += DOFsOnProc


###############################
#  --- Sequential in time --- #
###############################
else:
    # Get names of all procs holding data
    PuT = []
    for P in range(0, params["P"]):
        PuT.append(filenameIN + "." + "0" * (5-len(str(P))) + str(P))
    
    uT = np.zeros(NX)
    
    ind = 0
    for count, Ufilename in enumerate(PuT):
        # Read all data from the proc
        with open(Ufilename) as f:
            dims = f.readline()
        dims.split(" ")
        dims = [int(x) for x in dims.split()] 
        # Get data from lines > 0
        temp = np.loadtxt(Ufilename, skiprows = 1, usecols = 1) # Ignore the 1st column since it's just indices..., top row has junk in it we don't need
        DOFsOnProc = temp.shape[0]
        uT[ind:ind+DOFsOnProc] = temp
        ind += DOFsOnProc
            
    
    

### ---------------------------------------------------- ###
### --- COMPUTE ERRORS OF UT AGAINST EXACT SOLUTION ---  ###
### ---------------------------------------------------- ###

### ----------------------------------- ###
### --------------- 1D  --------------- ###
### ----------------------------------- ###
if params["space_dim"] == 1:
    nx = NX
    x = np.linspace(-1, 1, nx+1)
    x = x[:-1] # nx points in [-1, 1)
    T = params["dt"]  * (params["nt"] - 1)

    # The exact solutions for the test problems
    def uexact(x,t):
        if params["problemID"] == 1:
            temp = np.mod(x + 1  - t, 2) - 1
            return np.sin(np.pi*temp) ** 4
        elif (params["problemID"] == 2) or (params["problemID"] == 3):    
            return np.cos(np.pi*(x - t)) * np.exp(np.cos(2*np.pi*t) - 1)
            #return np.cos(np.pi*(x - t)) * np.exp(np.cos(t))/np.exp(1)

    uT_exact = np.zeros(nx)
    for i in range(0,nx):
        uT_exact[i] = uexact(x[i],T)

    # Compare uT against the exact solution
    #print("nx = {}, |uNum - uExact| = {:.4e}".format(nx, np.linalg.norm(uT_exact - uT, np.inf)))
    #print("(nt,nx) = ({},{}), |uNum - uExact| = {:.4e}".format(params["nt"], nx, np.sqrt(2/nx) * np.linalg.norm(uT_exact - uT, 2)))
    
    # Compute errors and save them
    e = uT_exact-uT
    save_error(e)
    
    
### ----------------------------------- ###
### --------------- 2D  --------------- ###
### ----------------------------------- ###
if params["space_dim"] == 2:
    nx = params["nx"]
    ny = params["nx"]
    x = np.linspace(-1, 1, nx+1)
    y = np.linspace(-1, 1, ny+1)
    x = x[:-1] # nx points in [-1, 1)
    y = y[:-1] # ny points in [-1, 1)
    [X, Y] = np.meshgrid(x, y)
    T = params["dt"]  * (params["nt"] - 1)

    # If used spatial parallelism, DOFs are not ordered in row-wise lexicographic, but instead
    # are blocked by proc, with procs in row-wise lexicographic order and DOFs on proc ordered
    # in row-wise lexicographic order
    if (params["spatialParallel"]):
        params["p_xTotal"] = int(params["p_xTotal"])
        perm = np.zeros(nx*ny, dtype = "int32")
        # Extract dimensions of processor grid if they were given
        if ("p_x0" in params):
            npInX = int(params["p_x0"])
            npInY = int(params["p_x1"])
        # Otherwise assume square processor grid
        else:
            npInX = int(np.sqrt(params["p_xTotal"])) 
            npInY = npInX 
        count = 0
        
        nxOnProcInt = int(nx / npInX)
        nyOnProcInt = int(ny / npInY)
        nxOnProcBnd = nx - (npInX-1)*int(nx / npInX) 
        nyOnProcBnd = ny - (npInY-1)*int(ny / npInY) 
        
        # Loop over DOFs in ascending order (i.e., ascending in their current order)
        for py in range(0, npInY):
            if (py < npInY-1):
                nyOnProc = nyOnProcInt
            else:
                nyOnProc = nyOnProcBnd
            for px in range(0, npInX):
                if (px < npInX - 1):
                    nxOnProc = nxOnProcInt 
                else:
                    nxOnProc = nxOnProcBnd
                for yIndOnProc in range(0, nyOnProc):
                    for xIndOnProc in range(0, nxOnProc):
                        xIndGlobal = px * nxOnProcInt + xIndOnProc # Global x-index for row we're in currently
                        globalInd  = py * (nyOnProcInt * nx) + yIndOnProc * nx + xIndGlobal # Global index of current DOF in ordering we want
                        perm[globalInd] = count
                        count += 1
        uT = uT[perm] # Permute solution array into correct ordering
        
        
    # Map 1D array into 2D for plotting.
    uT = uT.reshape(ny, nx)

    def u0(x,y):
        #return np.cos(np.pi*x) ** 4 * np.cos(np.pi*y) ** 4
        return np.cos(np.pi*x) ** 4 * np.sin(np.pi*y) ** 2
        # if ((x >= 0) and (y >= 0)):
        #     return 1.0
        # if ((x < 0) and (y >= 0)):
        #     return 2.0
        # if ((x < 0) and (y < 0)):
        #      return 3.0
        # if ((x >= 0) and (y < 0)):
        #      return 4.0

    # The exact solutions for the test problems
    def uexact(x, y ,t):
        if params["problemID"] == 1:
            tempx = np.mod(x + 1  - t, 2) - 1
            tempy = np.mod(y + 1  - t, 2) - 1
            return u0(tempx, tempy)
        elif  params["problemID"] == 2 or params["problemID"] == 3:
            return np.cos(np.pi*(x-t)) * np.cos(np.pi*(y-t)) * np.exp( np.cos(4*np.pi*t) - 1 )

    uT_exact = np.zeros((ny, nx))
    for j in range(0,ny):
        for i in range(0,nx):
            #uT_exact[i,j] = uexact(x[i],y[j],T)
            uT_exact[j,i] = uexact(x[i],y[j],T)


    # Compute errors and save them
    e = uT_exact-uT
    save_error(e)
