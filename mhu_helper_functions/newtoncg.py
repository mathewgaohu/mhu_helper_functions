from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize._linesearch import line_search_armijo
from scipy.sparse.linalg import LinearOperator, cg


def inexact_newton_cg(
    cost: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], LinearOperator],
    x0: np.ndarray,
    maxiter: int = 100,
    maxcgiter: int = 200,
    rtol: float = 1e-6,
    stag_tol: float = 1e-12,
    gdx_tol: float = 1e-15,
    precond_func: Callable[[np.ndarray], LinearOperator] = None,
    start_precond: int = 0,
    callback: Callable = None,
    checkpoint_path: str = None,
) -> tuple[np.ndarray, dict]:
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)

    converged = False
    termination_reason = "Maximum number of Iteration reached"
    cost_history = []
    gradnorm_history = []
    cost_count_history = []
    grad_count_history = []
    hess_count_history = []

    cost_with_count = FunctionWithCount(cost)
    grad_with_count = FunctionWithCount(grad)

    iter: int = 0
    x = x0.copy()
    while iter < maxiter:

        cost_with_count.count = 0
        grad_with_count.count = 0

        if iter > 0:
            f_old = f

        f = cost_with_count(x)
        g = grad_with_count(x)
        H = LinearOperatorWithCount(hess(x))
        gradnorm = np.linalg.norm(g)
        # print(iter, f)

        if checkpoint_path:
            np.save(checkpoint_path / f"x{iter}.npy", x)
            np.save(checkpoint_path / f"f{iter}.npy", f)
            np.save(checkpoint_path / f"g{iter}.npy", g)

        if iter == 0:
            gradnorm0 = gradnorm

        if gradnorm <= rtol * gradnorm0:
            converged = True
            termination_reason = "RTOL_ACHIEVED"
            break

        if iter > 0 and (f_old - f) < stag_tol * f_old:
            converged = True
            # print(f"f={f}, f_old={f_old}")
            termination_reason = "DESCENT_STAGNATED"
            break

        if iter >= start_precond and precond_func is not None:
            P = precond_func(x)
        else:
            P = None

        # tolcg = min(0.5, math.sqrt(gradnorm / gradnorm0))
        tolcg = min(0.5, gradnorm / gradnorm0)
        p, info = cg(H, -g, rtol=tolcg, M=P, maxiter=maxcgiter)
        gdx = np.dot(g, p)

        if -gdx < gdx_tol:
            converged = True
            termination_reason = "(g, dx) less than tolerance"
            break

        if checkpoint_path:
            np.save(checkpoint_path / f"p{iter}.npy", p)

        alpha, f_count, f_val_at_alpha = line_search_armijo(cost_with_count, x, p, g, f)
        if alpha is None:
            termination_reason = "LINESEARCH_FAILED"
            break

        iter += 1
        x += alpha * p
        # print(f"a={alpha}, |p|={np.linalg.norm(p)}, |x|={np.linalg.norm(x)}")
        if callback:
            callback(x)

        cost_count_history.append(cost_with_count.count)
        grad_count_history.append(grad_with_count.count)
        hess_count_history.append(H.count)

        alpha = f"{alpha:.2e}" if alpha else "None"
        print(
            f"Iter {iter}, cost: {f:.4e}, |g|: {gradnorm:.4e}, cg_tol: {tolcg:.2e}, cg_it: {H.count}, step_size: {alpha}, "
        )

    print("Inexact Newton-CG done.")
    print("    Termination reason:", termination_reason)
    print("    Iterations:", iter)
    print("    Cost evaluations:", sum(cost_count_history))
    print("    Gradient evaluations:", sum(grad_count_history))
    print("    Hessian evaluations:", sum(hess_count_history))
    print("    Final cost:", f)
    print("    Final |g|:", gradnorm)

    info = {
        "converged": converged,
        "termination_reason": termination_reason,
        "cost": f,
        "gradnorm": gradnorm,
        "iter": iter,
        "cost_history": cost_history,
        "gradnorm_history": gradnorm_history,
        "num_cost_evals": sum(cost_count_history),
        "num_grad_evals": sum(grad_count_history),
        "num_hess_evals": sum(hess_count_history),
        "cost_count_history": cost_count_history,
        "grad_count_history": grad_count_history,
        "hess_conut_history": hess_count_history,
    }
    return x, info


class FunctionWithCount:
    def __init__(self, func: Callable):
        self.func = func
        self.last_x = None
        self.last_out = None
        self.count = 0

    def __call__(self, x: np.ndarray):
        if self.last_x is not None:
            if np.linalg.norm(x - self.last_x) <= 1e-15 * np.max(
                [np.linalg.norm(x), np.linalg.norm(self.last_x)]
            ):
                return self.last_out
        self.last_x = x.copy()
        self.last_out = self.func(x)
        self.count += 1
        return self.last_out


class LinearOperatorWithCount(LinearOperator):
    def __init__(self, op: LinearOperator):
        self.op = op
        self.shape = op.shape
        self.dtype = op.dtype
        self.count = 0

    def _matvec(self, x):
        self.count += 1
        return self.op.matvec(x)
