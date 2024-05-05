from typing import Mapping

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla


class FunctionLinearOperator(spla.LinearOperator):

    def __init__(
        self,
        A: spla.LinearOperator,
        fun: Mapping,
        rank: int,
    ):
        assert A.shape[0] == A.shape[1]
        super().__init__(dtype=A.dtype, shape=A.shape)
        self.A = A
        self.fun = fun
        self.rank = rank

    def _matvec(self, x):
        N = self.shape[0]
        x = x.reshape(-1)
        qq = [np.zeros(x.shape), x / np.linalg.norm(x)]
        alphas = []
        betas = [0.0]

        for ii in range(self.rank):
            Q = np.asarray(qq[1:]).T.reshape((N, -1))
            q_prev = qq[-2]
            q_cur = qq[-1]
            v = self.A * q_cur
            alpha = q_cur.T @ v
            v = v - betas[-1] * q_prev - alpha * q_cur
            beta = np.linalg.norm(v)
            q_next = v / beta
            qq.append(q_next)
            alphas.append(alpha)
            betas.append(beta)

        evals_T, evecs_T = sla.eigh_tridiagonal(alphas, betas[1:-1])
        f_of_T = (evecs_T * self.fun(evals_T)) @ evecs_T.T
        return Q @ (f_of_T @ (Q.T @ x))

    def _rmatvec(self, x):
        return self._matvec(x)


def InvSqrtLinearOperator(A: spla.LinearOperator, rank: int):
    fun = lambda x: 1.0 / np.sqrt(x)
    return FunctionLinearOperator(A, fun, rank)
