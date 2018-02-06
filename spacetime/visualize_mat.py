from scipy.io import mmread
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import matplotlib.pyplot as plt
import pdb

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

A0 = mmread("/Users/ben/Dropbox/Research/Math/Projects/ParallelTime/mgrit_air/spacetime/test_mat_0.mm")
ax = plot_coo_matrix(A0)
# plt.show()

# A1 = mmread("/Users/ben/Dropbox/Research/Math/Projects/ParallelTime/mgrit_air/spacetime/test_mat_1.mm")
# ax = plot_coo_matrix(A1)
# plt.show()

# A2 = mmread("/Users/ben/Dropbox/Research/Math/Projects/ParallelTime/mgrit_air/spacetime/test_mat_2.mm")
# ax = plot_coo_matrix(A2)
# plt.show()

# A3 = mmread("/Users/ben/Dropbox/Research/Math/Projects/ParallelTime/mgrit_air/spacetime/test_mat_3.mm")
# ax = plot_coo_matrix(A3)
# plt.show()






# Make sure AB2 is ordering time matrices correctly
A0 = A0.tocsr()
timeDependent = False
nt = 10
bsize = 51
for i in range(0,nt-2):
    at0 = A0[(i+2)*bsize:bsize*(i+3),:][:,bsize*(i+1):bsize*(i+2)]
    at1 = A0[(i+2)*bsize:bsize*(i+3),:][:,bsize*i:bsize*(i+1)]
    test = np.abs(at0) - np.abs(at1)
    # For non-time dependent, should have same off-diagonal and differ by identity
    # on diagonal (-I + dt*A) vs dt*A
    if not timeDependent:
        if np.max(np.abs(test.diagonal() - 1)) > 0:
            print "bad news! non-unit diagonal."
            pdb.set_trace()
        test[np.arange(0,bsize),np.arange(0,bsize)] = 0
        if len(test.data[test.data != 0]) > 0:
            print "bad news! non-zero off-diagonal."
            pdb.set_trace()
    # With time-depedence scaling by t, matrix data should be bigger for bigger t
    else:
        test[np.arange(0,bsize),np.arange(0,bsize)] = 0
        if len(test.data[test.data < 0]) > 0:
            print "bad news!"
            pdb.set_trace()

pdb.set_trace()



