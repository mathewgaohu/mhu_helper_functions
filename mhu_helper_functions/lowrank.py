import numpy as np
import scipy.sparse.linalg as spla
from scipy import sparse
from scipy.sparse.linalg._interface import IdentityOperator


def LowRankOperator(
    d: np.ndarray,
    U: np.ndarray,
    B,
) -> spla.LinearOperator:
    """d, U = eigsh(A - B, B) (generalized eigenvalues)"""
    # Convert to linear operators
    D = spla.aslinearoperator(sparse.diags(d))
    U = spla.aslinearoperator(U)
    B = spla.aslinearoperator(B)

    # get required operators
    A = B * U * D * U.H * B + B
    return A


def LowRankInvOperator(
    d: np.ndarray,
    U: np.ndarray,
    Binv,
) -> spla.LinearOperator:
    """d, U = eigsh(A - B, B) (generalized eigenvalues)"""
    # Convert to linear operators
    Dsol = spla.aslinearoperator(sparse.diags(d / (d + 1.0)))
    U = spla.aslinearoperator(U)
    Binv = spla.aslinearoperator(Binv)

    # get required operators
    Ainv = Binv - U * Dsol * U.H
    return Ainv


def LowRankSqrtInvOperator(
    d: np.ndarray,
    U: np.ndarray,
    B,
    sqrtBinv,
) -> spla.LinearOperator:
    """d, U = eigsh(A - B, B) (generalized eigenvalues)"""
    # Convert to linear operators
    S = spla.aslinearoperator(sparse.diags(1.0 - (1.0 + d) ** (-0.5)))
    U = spla.aslinearoperator(U)
    B = spla.aslinearoperator(B)
    sqrtBinv = spla.aslinearoperator(sqrtBinv)
    I = IdentityOperator(dtype=B.dtype, shape=B.shape)

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
