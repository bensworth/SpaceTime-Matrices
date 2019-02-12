from scipy.io import mmread
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import matplotlib.pyplot as plt
import pdb

from scipy.sparse import load_npz

A_hypre_path = "./A_hypre.mm"
A_py_path    = "./Apy.npz"

# Get row/col numbers from first line of HYPRE out file.
with open(A_hypre_path) as f:
    dims = f.readline()
dims.split(" ")
dims = [int(x) for x in dims.split()] 
# Get mm data from lines > 0
dat = np.loadtxt(A_hypre_path, skiprows = 1)

A_hypre = csr_matrix((dat[:,2],(dat[:,0],dat[:,1])),shape=(dims[1]+1, dims[3]+1))
A_hypre.data[np.abs(A_hypre.data)<1e-15] = 0.0
A_hypre.eliminate_zeros()
print(A_hypre.shape, A_hypre.nnz)

plt.figure(1)
#plt.spy(A_hypre)
A_hypre = np.abs(A_hypre)
plt.imshow(A_hypre.todense())
plt.title("HYPRE")
plt.colorbar()
#plt.show()


A_py = load_npz(A_py_path)
A_py.data[np.abs(A_py.data)<1e-15] = 0.0
A_py.eliminate_zeros()
A_py = np.abs(A_py)
print(A_py.shape, A_py.nnz)
plt.figure(2)
#plt.spy(A_py)
plt.imshow(A_py.todense())
plt.title("Python")
plt.colorbar()


# print(np.sort(np.abs(A_hypre.data)))
# print("\n\n\n")
# print(np.sort(np.abs(A_py.data.data)))
# 
# print(A_hypre.indptr)
# print(A_py.indptr)
# 
# print(A_hypre.indices)
# print(A_py.indices)


# Make permutation vector to map from variable followed by DOF 
# ordering to DOF followed by variable ordering
nu = int((dims[1]+1)/2)
colinds = np.zeros((2*nu,), dtype='int32')
for i in range(0, 2*nu-1, 2):
	colinds[i] = i/2
for i in range(1, 2*nu, 2):
	colinds[i] = (i-1)/2+nu
	
# Permute rows and cols of HYPRE matrix accordingly.
A_hypre = A_hypre[colinds, :][:, colinds]
plt.figure(3)
plt.imshow(A_hypre.todense())
plt.title("Permuted HYPRE")
plt.colorbar()

plt.show()


test = A_hypre - A_py
test.data[np.abs(test.data)<1e-14] = 0
test.eliminate_zeros()

pdb.set_trace()

 
# print(A_hypre.data - A_py.data)
# print(A_hypre.indices - A_py.indices)
# print(A_hypre.indptr - A_py.indptr)

# plt.spy(A_hypre - A_py)
# plt.show()