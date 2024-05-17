"""Randomized algorithms for eigenvalue problems. 

Reference: 

Nathan Halko, Per Gunnar Martinsson, and Joel A. Tropp,
Finding structure with randomness:
Probabilistic algorithms for constructing approximate matrix decompositions,
SIAM Review, 53 (2011), pp. 217-288.

Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis,
Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application
to computing Karhunen-Loeve expansion,
Numerical Linear Algebra with Applications, to appear.

hippylib: https://hippylib.readthedocs.io/en/latest/index.html

"""

import numpy as np
import scipy.sparse.linalg as spla


def reigsh(
    A: spla.LinearOperator | np.ndarray,
    k: int,
    p: int = 10,
    single_pass: bool = False,
    Omega: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The randomized algorithm for the Hermitian Eigenvalues Problems.

    Args:
        A (spla.LinearOperator): the Hermitian operator.
        k (int): the number of eigenpairs to extract.
        p (int, optional): the oversampling size. Defaults to 10.
        sinle_pass (bool, optional): whether use single pass algorithm. Default False (double pass).
        Omega (np.ndarray, optional): a random gassian matrix. Defaults to None.

    Returns:
        np.ndarray: the leading k eigenvalues.
        np.ndarray: the leading k eigenvectors.
    """
    A = spla.aslinearoperator(A)
    Omega = np.random.randn(A.shape[1], k + p)
    Y = A * Omega

    Q, _ = np.linalg.qr(Y, mode="reduced")

    if single_pass:
        Z = Omega.T @ Q
        W = Y.T @ Q
        T = np.linalg.solve(Z, W)
        T = 0.5 * T + 0.5 * T.T  # T \approx Q^T A Q
    else:
        T = Q.T @ (A * Q)

    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()[::-1]
    d = d[sort_perm[:k]]
    V = V[:, sort_perm[:k]]

    return d, Q @ V


def reigsh_from_computed_actions(
    Omega: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The single pass algorithm for the Hermitian Eigenvalues Problems, given computed actions.

    Args:
        Omega (np.ndarray): a random gassian matrix with shape (n,k+p).
        Y (np.ndarray): the result of A*Omega with shape (n,k+p)
        k (int): the number of eigenpairs to extract.

    Returns:
        np.ndarray: the leading k eigenvalues.
        np.ndarray: the leading k eigenvectors.
    """
    assert Omega.shape == Y.shape
    assert k < Omega.shape[1]

    Q, _ = np.linalg.qr(Y, mode="reduced")

    Z = Omega.T @ Q
    W = Y.T @ Q
    T = np.linalg.solve(Z, W)
    T = 0.5 * T + 0.5 * T.T

    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()[::-1]
    d = d[sort_perm[:k]]
    V = V[:, sort_perm[:k]]

    return d, Q @ V
