import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scila
import scipy.sparse.linalg as spla
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.getcwd()))

from mhu_helper_functions.randomized_eig import (
    reigsh,
    reigsh_from_computed_actions,
    reigshg,
)

np.random.seed(0)
n = 500
k = 50
p = 5
A = np.random.randn(n, n)
u, _, _ = np.linalg.svd(A)
d = np.power(0.9, np.arange(n)) * 100
A = u @ np.diag(d) @ u.T
M = sparse.diags(np.arange(n) + 10)

# test reigsh
evals_true, evecs_true = spla.eigsh(A, k)
evals_true, evecs_true = evals_true[::-1], evecs_true[:, ::-1]
evals_single, evecs_single = reigsh(A, k, p, single_pass=True)
evals_double, evecs_double = reigsh(A, k, p, single_pass=False)
plt.figure()
plt.semilogy(evals_true, label="true eigenvalues")
plt.semilogy(evals_single, label="single pass eigenvales")
plt.semilogy(evals_double, label="double pass eigenvales")
plt.title("Eigenvalues")
plt.legend()
plt.savefig("test.png")


# test reigshg
evals_true, evecs_true = spla.eigsh(A, k, M)
evals_true, evecs_true = evals_true[::-1], evecs_true[:, ::-1]
evals_single, evecs_single = reigshg(A, k, M, p, single_pass=True)
evals_double, evecs_double = reigshg(A, k, M, p, single_pass=False)
plt.figure()
plt.semilogy(evals_true, label="true eigenvalues")
plt.semilogy(evals_single, label="single pass eigenvales")
plt.semilogy(evals_double, label="double pass eigenvales")
plt.title("Eigenvalues")
plt.legend()
plt.savefig("test.png")


# test reigsh_from_computed_actions
evals_true, evecs_true = spla.eigsh(A, k)
evals_true, evecs_true = evals_true[::-1], evecs_true[:, ::-1]
Omega = np.random.randn(n, k + p)
Y = A @ Omega
evals_single, evecs_single = reigsh_from_computed_actions(Omega, Y, k)
plt.figure()
plt.semilogy(evals_true, label="true eigenvalues")
plt.semilogy(evals_single, label="single pass eigenvales")
plt.title("Eigenvalues")
plt.legend()
plt.savefig("test.png")
