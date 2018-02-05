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
plt.show()

A1 = mmread("/Users/ben/Dropbox/Research/Math/Projects/ParallelTime/mgrit_air/spacetime/test_mat_1.mm")
ax = plot_coo_matrix(A1)
plt.show()

# A2 = mmread("/Users/ben/Dropbox/Research/Math/Projects/ParallelTime/mgrit_air/spacetime/test_mat_2.mm")
# ax = plot_coo_matrix(A2)
# plt.show()

# A3 = mmread("/Users/ben/Dropbox/Research/Math/Projects/ParallelTime/mgrit_air/spacetime/test_mat_3.mm")
# ax = plot_coo_matrix(A3)
# plt.show()


pdb.set_trace()