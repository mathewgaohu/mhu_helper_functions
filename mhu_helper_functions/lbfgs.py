from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Iterable

import numpy as np
import scipy.optimize._linesearch as ls
from scipy.optimize import line_search
from scipy.sparse.linalg import LinearOperator

from .wrappers import memoize_with_tol


def plbfgs(
    cost: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_vector_pairs_stored: int = 20,
    rtol: float = 1e-6,
    stag_tol: float = 1e-8,
    max_iter: int = 100,
    inv_hess0: LinearOperator | Callable = None,
    inv_hess0_update_freq: int = None,
    hess0: LinearOperator | Callable = None,
    iters_before_inv_hess0: int = 0,
    callback: Callable[[np.ndarray], Any] = None,
    checkpoint_dir: str = None,
    first_step_size: float = None,
    lbfgs_inv_hess_kwargs: dict[str, Any] = dict(),
    line_search_kwargs: dict[str, Any] = dict(),
) -> tuple[np.ndarray, dict, LbfgsInverseHessianApproximation]:
    """Computes argmin_x cost(x) via L-BFGS,
    with option for a user-supplied initial inverse Hessian approximation.
    """
    assert len(x0.shape) == 1, "Only implemented for 1D array x"

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)

    converged = False
    termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.MAXITER_REACHED
    cost_history: list[float] = []
    gradnorm_history: list[float] = []
    cost_count_history: list[int] = []
    grad_count_history: list[int] = []
    step_size_history: list[float] = []

    cost = memoize_with_tol(1e-15)(cost)
    grad = memoize_with_tol(1e-15)(grad)

    inv_hess = LbfgsInverseHessianApproximation(
        max_vector_pairs_stored,
        deque(),
        deque(),
        None,
        **lbfgs_inv_hess_kwargs,
    )

    x = x0.copy()

    for it in range(max_iter):
        print(f"\n---- Iter {it+1} ----")

        # Initialize iteration
        cost.ncalls = 0
        grad.ncalls = 0

        # Update preconditioner
        if it == iters_before_inv_hess0 and inv_hess0:
            print("Start using preconditioner")
            if isinstance(inv_hess0, LinearOperator):
                inv_hess.inv_hess0 = inv_hess0
                inv_hess.hess0 = hess0
            elif isinstance(inv_hess0, Callable):
                inv_hess.inv_hess0 = inv_hess0(x)
                inv_hess.hess0 = hess0(x)
            else:
                raise ValueError(type(inv_hess0))
        if (
            inv_hess0_update_freq
            and it > iters_before_inv_hess0
            and (it - iters_before_inv_hess0) % inv_hess0_update_freq == 0
        ):
            print("Update preconditioner")
            inv_hess.inv_hess0 = inv_hess0(x)
            inv_hess.hess0 = hess0(x)

        # Compute current state
        f: float = cost(x)
        g: np.ndarray = grad(x)
        gnorm: float = np.linalg.norm(g)
        if it == 0:
            gnorm0 = gnorm

        print(f"f = {f:.3e}, |g| = {gnorm:.3e}, |g|/|g0| = {gnorm/gnorm0:.3e}")

        # Record
        cost_history.append(f)
        gradnorm_history.append(gnorm)
        if checkpoint_dir:
            np.save(checkpoint_dir / f"x{it}.npy", x)
            np.save(checkpoint_dir / f"f{it}.npy", f)
            np.save(checkpoint_dir / f"g{it}.npy", g)

        # Whether converged
        if gnorm <= rtol * gnorm0:
            termination_reason: LbfgsTerminationReason = (
                LbfgsTerminationReason.RTOL_ACHIEVED
            )
            break

        if it > 0 and (f_old - f) < stag_tol * f_old:
            termination_reason: LbfgsTerminationReason = (
                LbfgsTerminationReason.DESCENT_STAGNATED
            )
            break

        # Update BFGS inv Hess vec pairs: s = x - x_old, y = g - g_old
        if it > 0:
            inv_hess.add_new_s_y_pair(x - x_old, g - g_old)

        # Compute update direction
        p: np.ndarray = inv_hess.matvec(-g)

        # Save checkpoints
        if checkpoint_dir:
            np.save(checkpoint_dir / f"p{it}.npy", p)

        # Line search
        if it == 0 and first_step_size is not None:
            p *= first_step_size
        step_size, new_f = _line_search(
            cost,
            grad,
            x,
            p,
            g,
            f,
            None,
            **line_search_kwargs,
        )
        print(f"step_size = {step_size:.3e}")
        print(f"f_count = {cost.ncalls}, g_count = {grad.ncalls}")

        # Update to next point
        if step_size:
            x_old = x
            f_old = f
            g_old = g
            x = x_old + step_size * p
            if callback:
                callback(x)
        else:
            termination_reason = LbfgsTerminationReason.LINESEARCH_FAILED
            break

        # Record
        cost_count_history.append(cost.ncalls)
        grad_count_history.append(grad.ncalls)
        step_size_history.append(step_size)

    print("\nLBFGS done.")
    print("    Termination reason:", termination_reason.name)
    print("    Iterations:", it)
    print("    Cost evaluations:", sum(cost_count_history))
    print("    Gradient evaluations:", sum(grad_count_history))
    print("    Final cost:", f)
    print("    Final |g|:", gnorm)

    info = {
        "converged": converged,
        "termination_reason": termination_reason.name,
        "cost": f,
        "gradnorm": gnorm,
        "iter": it,
        "cost_history": cost_history,
        "gradnorm_history": gradnorm_history,
        "num_cost_evals": sum(cost_count_history),
        "num_grad_evals": sum(grad_count_history),
        "cost_count_history": cost_count_history,
        "grad_count_history": grad_count_history,
    }
    return x, info, inv_hess


class LbfgsTerminationReason(Enum):
    MAXITER_REACHED = 0
    RTOL_ACHIEVED = 1
    DESCENT_STAGNATED = 2
    LINESEARCH_FAILED = 3


@dataclass
class LbfgsInverseHessianApproximation:
    """See Nocedal and Wright page 177-179."""

    m: int  # max vector pairs stored
    ss: Deque[
        np.ndarray
    ]  # GETS MODIFIED! ss=[s_(k-1), s_(k-2), ..., s_(k-m)], s_i = x_(i+1) - x_i. Eq 7.18, left, on page 177
    yy: Deque[
        np.ndarray
    ]  # GETS MODIFIED! yy=[y_(k-1), y_(k-2), ..., y_(k-m)], y_i = grad f_(i+1) - grad f_i. Eq 7.18, right, on page 177
    inv_hess0: Callable[[np.ndarray], np.ndarray] = (
        None  # Initial inverse Hessian approximation
    )
    hess0: Callable[[np.ndarray], np.ndarray] = None  # Initial Hessian approximation
    gamma_type: int = 0

    def __post_init__(self) -> None:
        assert self.m >= 0
        assert len(self.ss) == len(self.yy)
        while len(self.ss) > self.m:
            self.ss.pop()
        while len(self.yy) > self.m:
            self.yy.pop()

    def add_new_s_y_pair(self, s: np.ndarray, y: np.ndarray) -> None:
        self.ss.appendleft(s)
        if len(self.ss) > self.m:
            self.ss.pop()

        self.yy.appendleft(y)
        if len(self.yy) > self.m:
            self.yy.pop()

    def compute_gamma_k(self) -> float:
        if self.ss:
            s, y = self.ss[0], self.yy[0]
            if self.inv_hess0 is not None:
                if self.gamma_type == 0:
                    gamma_k = (s @ y) / (y @ (self.inv_hess0 @ y))
                elif self.gamma_type == 1:
                    assert self.hess0 is not None
                    gamma_k = (y @ (self.hess0 @ s)) / (y @ y)
                elif self.gamma_type == 2:
                    assert self.hess0 is not None
                    gamma_k = (s @ (self.hess0 @ s)) / (s @ y)
                else:
                    raise ValueError("gamma_type:", self.gamma_type)
            else:
                gamma_k = (s @ y) / (y @ y)
        else:
            gamma_k = 1.0
        print("gamma_k =", gamma_k)
        return gamma_k

    def apply_inv_hess0_k(self, x: np.ndarray) -> np.ndarray:
        gamma_k = self.compute_gamma_k()
        if self.inv_hess0 is not None:
            return gamma_k * (self.inv_hess0 @ x)  # H0_k = gamma_k*P
        else:
            return gamma_k * x  # H0_k = gamma_k*I

    def matvec(self, q: np.ndarray) -> np.ndarray:
        """Computes
            r = H_k grad f_k
        via L-BFGS two-loop recursion.
        Algorithm 7.4 on page 178 of Nocedal and Wright.
        """
        rhos = [
            _componentwise_inverse(_inner_product(y, s))
            for s, y in zip(self.ss, self.yy)
        ]  # 1.0 / inner(y, s). equation 7.17 (left) on page 177
        alphas = []
        for s, y, rho in zip(self.ss, self.yy, rhos):
            alpha = rho * _inner_product(s, q)
            q = _sub(q, _componentwise_scalar_mult(y, alpha))  # q = q - alpha*y
            alphas.append(alpha)
        r = self.apply_inv_hess0_k(q)
        for s, y, rho, alpha in zip(
            reversed(self.ss), reversed(self.yy), reversed(rhos), reversed(alphas)
        ):
            beta = rho * _inner_product(y, r)
            r = _add(
                r, _componentwise_scalar_mult(s, alpha - beta)
            )  # r = r + s * (alpha - beta)
        return r


def _is_container(x):
    return isinstance(x, Iterable) and (not isinstance(x, np.ndarray))


def _inner_product(x, y) -> float:
    if _is_container(x):
        return np.sum([_inner_product(xi, yi) for xi, yi in zip(x, y)])
    else:
        return (x * y).sum()


def _norm(x) -> float:  # ||x||
    return np.sqrt(_inner_product(x, x))


def _add(x, y):  # x + y
    if _is_container(x):
        T = type(x)
        return T([_add(xi, yi) for xi, yi in zip(x, y)])
    else:
        return x + y


def _neg(x):  # -x
    if _is_container(x):
        T = type(x)
        return T([_neg(xi) for xi in x])
    else:
        return -x


def _sub(x, y):  # x - y
    return _add(x, _neg(y))


def _componentwise_scalar_mult(x, s):  # x * s. x is a vector and s is a scalar
    if _is_container(x):
        T = type(x)
        return T([_componentwise_scalar_mult(xi, s) for xi in x])
    else:
        return x * s


def _componentwise_inverse(x):
    if _is_container(x):
        T = type(x)
        return T([_componentwise_inverse(xi) for xi in zip(x)])
    else:
        return 1.0 / x


def _line_search(cost, grad, x, p, g, f, old_old_fval, **kwargs):
    # First try wolf2 (more efficient)
    alpha, fc, gc, new_fval, old_fval, new_slopp = line_search(
        cost, grad, x, p, g, f, old_old_fval, maxiter=20, **kwargs
    )
    if alpha is None:  # Back to wolf1 (more robust)
        alpha, fc, gc, new_fval, old_fval, new_slopp = ls.line_search_wolfe1(
            cost, grad, x, p, g, f, old_old_fval, **kwargs
        )
    return alpha, new_fval
