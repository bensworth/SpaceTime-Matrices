from scipy.io import mmread
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import matplotlib.pyplot as plt
import pdb

from scipy.sparse import load_npz

A_hypre_path = "/Users/oliverkrzysik/Software/mgrit_air/spacetime/FD_class/A_hypre.mm"
A_py_path    = "/Users/oliverkrzysik/Software/mgrit_air/spacetime/FD_class/Apy.npz"

# Get row/col numbers from first line of HYPRE out file.
with open(A_hypre_path) as f:
    dims = f.readline()
dims.split(" ")
dims = [int(x) for x in dims.split()] 
# Get mm data from lines > 0
dat = np.loadtxt(A_hypre_path, skiprows = 1)

A_hypre = csr_matrix((dat[:,2],(dat[:,0],dat[:,1])),shape=(dims[1]+1, dims[3]+1))
A_hypre.eliminate_zeros()
print(A_hypre.shape, A_hypre.nnz)

plt.figure(1)
plt.spy(A_hypre)
#plt.show()


A_py = load_npz(A_py_path)
print(A_py.shape, A_py.nnz)
plt.figure(2)
plt.spy(A_py)
plt.show()

 
# print(A_py.shape, A_py.nnz)
# print(A_hypre.shape, A_hypre.nnz)
# 
# 
# print(A_hypre.data - A_py.data)
# print(A_hypre.indices - A_py.indices)
# print(A_hypre.indptr - A_py.indptr)

# plt.spy(A_hypre - A_py)
# plt.show()