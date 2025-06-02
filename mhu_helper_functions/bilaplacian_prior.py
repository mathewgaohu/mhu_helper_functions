import math

import numpy as np
import scipy.sparse.linalg as spla
from scipy import sparse as sp


class BilaplacianPrior:  # TODO: Better implementation

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        gamma: float,
        delta: float,
        mean: np.ndarray = None,
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.gamma = gamma
        self.delta = delta
        self.mean = mean
        if self.mean is not None:
            assert self.mean.shape == (nx * ny,)
        else:
            self.mean = np.zeros((nx * ny,))

        Ix = sp.eye(nx)
        D2x = (
            sp.diags(
                [np.ones(nx - 1), -2.0 * np.ones(nx), np.ones(nx - 1)],
                [-1, 0, 1],
            )
            / dx**2
        ).tocsr()
        D2x[0, 0] = D2x[-1, -1] = -1.0 / dx**2  # Neumann boundary condition
        Iy = sp.eye(ny)
        D2y = (
            sp.diags(
                [np.ones(ny - 1), -2.0 * np.ones(ny), np.ones(ny - 1)],
                [-1, 0, 1],
            )
            / dy**2
        ).tocsr()
        D2y[0, 0] = D2y[-1, -1] = -1.0 / dy**2  # Neumann boundary condition
        I = sp.identity(nx * ny)
        D2 = sp.kron(D2x, Iy) + sp.kron(Ix, D2y)  # stiffness matrix
        self.Rh: sp.csr_matrix = delta * I - gamma * D2
        self.R: sp.csr_matrix = self.Rh * self.Rh

    def cost(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = self.Rh * (x - self.mean)
        return 0.5 * (y * y).sum()

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.Rh * (self.Rh * (x - self.mean))

    def hess(self, x: np.ndarray) -> spla.LinearOperator:
        return spla.aslinearoperator(self.R)


def BiLaplacianComputeCoefficients(
    sigma2: float, rho: float, ndim: int
) -> tuple[float, float]:
    """Ref: hippylib.BiLaplacianComputeCoefficients"""
    nu = 2.0 - 0.5 * ndim
    kappa = np.sqrt(8 * nu) / rho

    s = (
        np.sqrt(sigma2)
        * np.power(kappa, nu)
        * np.sqrt(np.power(4.0 * np.pi, 0.5 * ndim) / math.gamma(nu))
    )

    gamma = 1.0 / s
    delta = np.power(kappa, 2) / s

    return gamma, delta


class BilaplacianPriorWithoutBC:  # Old Implementation

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        gamma: float,
        delta: float,
        mean: np.ndarray = None,
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.gamma = gamma
        self.delta = delta
        self.mean = mean
        if self.mean is not None:
            assert self.mean.shape == (nx * ny,)
        else:
            self.mean = np.zeros((nx * ny,))

        Ix = sp.eye(nx)
        D2x = (
            sp.diags(
                [np.ones(nx - 1), -2.0 * np.ones(nx), np.ones(nx - 1)],
                [-1, 0, 1],
            )
            / dx**2
        ).tocsr()
        # D2x[0, 0] = D2x[-1, -1] = -1.0 / dx**2  # Neumann boundary condition
        Iy = sp.eye(ny)
        D2y = (
            sp.diags(
                [np.ones(ny - 1), -2.0 * np.ones(ny), np.ones(ny - 1)],
                [-1, 0, 1],
            )
            / dy**2
        ).tocsr()
        # D2y[0, 0] = D2y[-1, -1] = -1.0 / dy**2  # Neumann boundary condition
        I = sp.identity(nx * ny)
        D2 = sp.kron(D2x, Iy) + sp.kron(Ix, D2y)  # stiffness matrix
        self.Rh: sp.csr_matrix = delta * I - gamma * D2
        self.R: sp.csr_matrix = self.Rh * self.Rh

    def cost(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = self.Rh * (x - self.mean)
        return 0.5 * (y * y).sum()

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.Rh * (self.Rh * (x - self.mean))

    def hess(self, x: np.ndarray) -> spla.LinearOperator:
        return spla.aslinearoperator(self.R)
