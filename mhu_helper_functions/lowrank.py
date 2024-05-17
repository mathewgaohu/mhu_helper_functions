"""Build low-rank approximation to the operator, inversion, and inverse sqrt from computed generalized eigenpairs."""

import numpy as np
import scipy.sparse.linalg as spla
from scipy import sparse
from scipy.sparse.linalg._interface import IdentityOperator


def LowRankOperator(
    d: np.ndarray,
    U: np.ndarray,
    B: spla.LinearOperator | np.ndarray = None,
) -> spla.LinearOperator:
    """The low rank approximation based on generalized eigenpairs.

            (A-B) u_i = d_i B u_i,

    i.e. (A-B)U = BUD, and U^* BU=I

    For example, d, U = scipy.sparse.linalg.eigsh(A - B, B),

    Args:
        d: 1D array of the leading eigenvalues
        U: 2D array of the leading eigenvectors
        B: the base operator

    Returns:
        LinearOperator: the low rank approximation of the operator A

    """
    # Convert to linear operators
    n, r = U.shape
    D = spla.aslinearoperator(sparse.diags(d))
    U = spla.aslinearoperator(U)
    I = IdentityOperator((n, n), U.dtype)
    B = I if B is None else spla.aslinearoperator(B)

    # get required operators
    A = B * U * D * U.H * B + B
    return A


def LowRankInvOperator(
    d: np.ndarray,
    U: np.ndarray,
    Binv: spla.LinearOperator | np.ndarray = None,
) -> spla.LinearOperator:
    """The low rank approximation to inversion based on generalized eigenpairs.

            (A-B) u_i = d_i B u_i,

    i.e. (A-B)U = BUD, and U^* BU=I

    For example, d, U = scipy.sparse.linalg.eigsh(A - B, B),

    Args:
        d: 1D array of the leading eigenvalues
        U: 2D array of the leading eigenvectors
        Binv: the base operator's inversion

    Returns:
        LinearOperator: the low rank approximation of the operator A's inversion.

    """
    # Convert to linear operators
    n, r = U.shape
    Dsol = spla.aslinearoperator(sparse.diags(d / (d + 1.0)))
    U = spla.aslinearoperator(U)
    I = IdentityOperator((n, n), U.dtype)
    Binv = I if Binv is None else spla.aslinearoperator(Binv)

    # get required operators
    Ainv = Binv - U * Dsol * U.H
    return Ainv


def LowRankSqrtInvOperator(
    d: np.ndarray,
    U: np.ndarray,
    B: spla.LinearOperator | np.ndarray = None,
    sqrtBinv: spla.LinearOperator | np.ndarray = None,
) -> spla.LinearOperator:
    """The low rank approximation to inverse sqrt based on generalized eigenpairs.

            (A-B) u_i = d_i B u_i,

    i.e. (A-B)U = BUD, and U^* BU=I

    For example, d, U = scipy.sparse.linalg.eigsh(A - B, B),

    The inverse sqrt of an operator is defined as sqrtAinv * sqrtAinv.H = Ainv

    Args:
        d: 1D array of the leading eigenvalues
        U: 2D array of the leading eigenvectors
        B: the base operator
        sqrtBinv: the base operator's inverse sqrt

    Returns:
        LinearOperator: the low rank approximation of the operator A's inverse sqrt

    """
    # Convert to linear operators
    n, r = U.shape
    S = spla.aslinearoperator(sparse.diags(1.0 - (1.0 + d) ** (-0.5)))
    U = spla.aslinearoperator(U)
    I = IdentityOperator((n, n), U.dtype)
    B = I if B is None else spla.aslinearoperator(B)
    sqrtBinv = I if sqrtBinv is None else spla.aslinearoperator(sqrtBinv)

    # get required operators
    sqrtAinv = (I - U * S * U.H * B) * sqrtBinv
    return sqrtAinv


def test():
    n = 50
    r = 10
    R = np.random.rand(n, n)
    u, s, vt = np.linalg.svd(R)
    B = u @ np.diag(s) @ u.T
    Binv = u @ np.diag(s ** (-1)) @ u.T
    sqrtBinv = u @ np.diag(s ** (-0.5)) @ u.T
    R = np.random.randn(n, r)
    A = B + R @ R.T
    w, v = np.linalg.eigh(A)
    Ainv = v @ np.diag(w ** (-1)) @ v.T
    sqrtAinv = v @ np.diag(w ** (-0.5)) @ v.T

    d, U = spla.eigsh(A - B, r, B)
    print(d)

    C, Cinv, sqrtCinv = (
        LowRankOperator(d, U, B),
        LowRankInvOperator(d, U, Binv),
        LowRankSqrtInvOperator(d, U, B, sqrtBinv),
    )
    print(
        "A error = ",
        np.linalg.norm(A - C * np.eye(n), "fro") / np.linalg.norm(A, "fro"),
    )
    print(
        "Ainv error = ",
        np.linalg.norm(Ainv - Cinv * np.eye(n), "fro") / np.linalg.norm(Ainv, "fro"),
    )
    print(
        "sqrtAinv error = ",
        np.linalg.norm(sqrtCinv * sqrtCinv.H * np.eye(n) - Ainv, "fro")
        / np.linalg.norm(Ainv, "fro"),
    )
