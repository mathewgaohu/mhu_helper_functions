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
import scipy.linalg as scila
import scipy.sparse.linalg as spla

def reigsh(
    A,
    k: int,
    p: int = 10,
    single_pass: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """The randomized algorithm for the Hermitian Eigenvalues Problems.

    Args:
        A (sparse matrix, dense matrix, LinearOperator): the Hermitian operator.
        k (int): the number of eigenpairs to extract.
        p (int, optional): the oversampling size. Defaults to 10.
        sinle_pass (bool, optional): whether use single pass algorithm. Default False (double pass).

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
        T = scila.solve(Z, W, overwrite_a=True, overwrite_b=True)
        T = 0.5 * T + 0.5 * T.T  # T \approx Q^T A Q
    else:
        T = Q.T @ (A * Q)

    d, V = scila.eigh(T, overwrite_a=True)
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
        Omega (np.ndarray): A random Gaussian matrix with shape (n,k+p).
        Y (np.ndarray): The result of A*Omega with shape (n,k+p)
        k (int): The number of eigenpairs to extract.

    Returns:
        np.ndarray: the leading k eigenvalues.
        np.ndarray: the leading k eigenvectors.
    """
    assert Omega.shape == Y.shape
    assert k <= Omega.shape[1]

    Q, _ = np.linalg.qr(Y, mode="reduced")

    Z = Omega.T @ Q
    W = Y.T @ Q
    T = scila.solve(Z, W, overwrite_a=True, overwrite_b=True)
    T = 0.5 * T + 0.5 * T.T

    d, V = scila.eigh(T, overwrite_a=True)
    sort_perm = d.argsort()[::-1]
    d = d[sort_perm[:k]]
    V = V[:, sort_perm[:k]]

    return d, Q @ V


def reigshg(
    A,
    k: int,
    M,
    p: int = 10,
    single_pass: bool = False,
    Minv=None,
) -> tuple[np.ndarray, np.ndarray]:
    """The randomized algorithm for the Hermitian Eigenvalues Problems.

    Args:
        A (sparse matrix, dense matrix, LinearOperator): the Hermitian operator.
        k (int): the number of eigenpairs to extract.
        M (sparse matrix, dense matrix, LinearOperator): The base matrix.
        p (int, optional): the oversampling size. Defaults to 10.
        sinle_pass (bool, optional): whether use single pass algorithm. Default False (double pass).
        Minv (sparse matrix, dense matrix, LinearOperator): The inversion of the base matrix.

    Returns:
        np.ndarray: the leading k eigenvalues.
        np.ndarray: the leading k eigenvectors.
    """
    A = spla.aslinearoperator(A)
    M = spla.aslinearoperator(M)
    if Minv is None:
        Minv = spla.LinearOperator(
            dtype=M.dtype,
            shape=M.shape[::-1],
            matvec=lambda x: spla.cg(M, x)[0],
            rmatvec=lambda x: spla.cg(M, x)[0],
        )

    Omega = np.random.randn(A.shape[1], k + p)
    Y = A * Omega
    Ybar = Minv * Y
    Q, MQ, _ = mgs_stable(M, Ybar)

    if single_pass:
        Z = Omega.T @ MQ
        W = Y.T @ Q
        T = scila.solve(Z, W, overwrite_a=True, overwrite_b=True)
        T = 0.5 * T + 0.5 * T.T  # T \approx Q^T A Q
    else:
        T = Q.T @ (A * Q)

    d, V = scila.eigh(T, overwrite_a=True)
    sort_perm = d.argsort()[::-1]
    d = d[sort_perm[:k]]
    V = V[:, sort_perm[:k]]

    return d, Q @ V


# https://github.com/arvindks/kle/blob/master/eigen/orth.py
def mgs_stable(A, Z, verbose=False):
    """
    Returns QR decomposition of Z. Q and R satisfy the following relations
    in exact arithmetic

    1. QR    	= Z
    2. Q^*AQ 	= I
    3. Q^*AZ	= R
    4. ZR^{-1}	= Q

    Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
    for computing the A-orthogonal QR factorization

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
            An array, sparse matrix, or LinearOperator representing
            the operation ``A * x``, where A is a real or complex square matrix.

    Z : ndarray

    verbose : bool, optional
              Displays information about the accuracy of the resulting QR
              Default: False

    Returns
    -------

    q : ndarray
            The A-orthogonal vectors

    Aq : ndarray
            The A^{-1}-orthogonal vectors

    r : ndarray
            The r of the QR decomposition


    See Also
    --------
    mgs : Modified Gram-Schmidt without re-orthogonalization
    precholqr  : Based on CholQR


    References
    ----------
    .. [1] A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized
            Hermitian Eigenvalue Problems with application to computing
            Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885

    .. [2] W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980

    Examples
    --------

    >>> import numpy as np
    >>> A = np.diag(np.arange(1,101))
    >>> Z = np.random.randn(100,10)
    >>> q, Aq, r = mgs_stable(A, Z, verbose = True)

    """

    # Get sizes
    m = np.size(Z, 0)
    n = np.size(Z, 1)

    # Convert into linear operator
    Aop = spla.aslinearoperator(A)

    # Initialize
    Aq = np.zeros_like(Z, dtype="d")
    q = np.zeros_like(Z, dtype="d")
    r = np.zeros((n, n), dtype="d")

    reorth = np.zeros((n,), dtype="d")
    eps = np.finfo(np.float64).eps

    q = np.copy(Z)

    for k in np.arange(n):
        Aq[:, k] = Aop.matvec(q[:, k])
        t = np.sqrt(np.dot(q[:, k].T, Aq[:, k]))

        nach = 1
        u = 0
        while nach:
            u += 1
            for i in np.arange(k):
                s = np.dot(Aq[:, i].T, q[:, k])
                r[i, k] += s
                q[:, k] -= s * q[:, i]

            Aq[:, k] = Aop.matvec(q[:, k])
            tt = np.sqrt(np.dot(q[:, k].T, Aq[:, k]))
            if tt > t * 10.0 * eps and tt < t / 10.0:
                nach = 1
                t = tt
            else:
                nach = 0
                if tt < 10.0 * eps * t:
                    tt = 0.0

        reorth[k] = u
        r[k, k] = tt
        tt = 1.0 / tt if np.abs(tt * eps) > 0.0 else 0.0
        q[:, k] *= tt
        Aq[:, k] *= tt

    if verbose:
        # Verify Q*R = Y
        print("||QR-Y|| is ", np.linalg.norm(np.dot(q, r) - Z, 2))

        # Verify Q'*A*Q = I
        T = np.dot(q.T, Aq)
        print("||Q^TAQ-I|| is ", np.linalg.norm(T - np.eye(n, dtype="d"), ord=2))

        # verify Q'AY = R
        print("||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T, Z) - r, 2))

        # Verify YR^{-1} = Q
        val = np.inf
        try:
            val = np.linalg.norm(np.linalg.solve(r.T, Z.T).T - q, 2)
        except scila.LinAlgError:
            print("Singular")
        print("||YR^{-1}-Q|| is ", val)

    return q, Aq, r
