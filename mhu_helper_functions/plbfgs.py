from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Iterable, NamedTuple, TypeVar

import numpy as np
import scipy.optimize._linesearch as ls
from scipy.optimize import line_search
from scipy.sparse.linalg import LinearOperator

VecType = TypeVar("VecType")


def plbfgs(
    cost: Callable[[VecType], float],
    grad: Callable[[VecType], VecType],
    x0: VecType,
    max_vector_pairs_stored: int = 20,
    rtol: float = 1e-6,
    stag_tol: float = 1e-8,
    max_iter: int = 100,
    print_level: int = 1,
    inv_hess0: LinearOperator | LbfgsInverseHessianApproximation | Callable = None,
    inv_hess0_update_freq: int = None,
    num_initial_iter: int = 0,  # number of initial iterations before inv_hess0 is used
    callback: Callable[[VecType], Any] = None,
    inv_hess_options: dict[str, Any] = dict(),
    line_search_options: dict[str, Any] = dict(),
    checkpoint_options: dict[str, Any] = dict(),
    first_step_size: float = None,
) -> LbfgsResult:
    """Computes argmin_x cost(x) via L-BFGS,
    with option for a user-supplied initial inverse Hessian approximation.

    checkpoint_options:
        - "path": str
    """
    # Settings
    if checkpoint_options:
        log_path = checkpoint_options["path"]

    # Initialization
    num_cost_evals: int = 0

    def __cost_with_counter(x):
        nonlocal num_cost_evals
        num_cost_evals += 1
        return cost(x)

    num_grad_evals: int = 0
    last_grad_x: VecType = None
    last_grad_g: VecType = None

    def __grad_with_counter(x):
        nonlocal num_grad_evals, last_grad_x, last_grad_g
        if last_grad_x is not None:
            if _norm(_sub(last_grad_x, x)) <= 1e-15 * np.max(
                [_norm(x), _norm(last_grad_x)]
            ):
                return last_grad_g
        last_grad_x = x
        last_grad_g = grad(x)
        num_grad_evals += 1
        return last_grad_g

    iter: int = 0
    if iter >= num_initial_iter and isinstance(
        inv_hess0, LbfgsInverseHessianApproximation
    ):
        inv_hess = inv_hess0
    else:
        if iter >= num_initial_iter:
            if isinstance(inv_hess0, LinearOperator):
                H0 = inv_hess0
            else:  # Callable
                H0 = inv_hess0(x0)
        else:
            H0 = None
        inv_hess = LbfgsInverseHessianApproximation(
            max_vector_pairs_stored,
            deque(),
            deque(),
            H0,
            print_level=print_level - 1,
            **inv_hess_options,
        )

    # Step 0: Compute inital f, g, p and line search step
    x: VecType = x0
    f: float = __cost_with_counter(x)
    g: VecType = __grad_with_counter(x)
    gradnorm: float = _norm(g)

    cost_history: list[float] = [f]
    gradnorm_history: list[float] = [gradnorm]

    p: VecType = inv_hess.matvec(_neg(g))
    if first_step_size is not None:
        p *= first_step_size
    if checkpoint_options:
        np.save(os.path.join(log_path, f"x{iter}.npy"), x)
        np.save(os.path.join(log_path, f"f{iter}.npy"), f)
        np.save(os.path.join(log_path, f"g{iter}.npy"), g)
        np.save(os.path.join(log_path, f"p{iter}.npy"), p)

    if inv_hess.inv_hess0 is None:
        old_old_fval = _add(f, _norm(g) / 2.0)
    else:
        old_old_fval = None
    line_search_result = _line_search(
        __cost_with_counter,
        __grad_with_counter,
        x,
        p,
        g,
        f,
        old_old_fval,
        **line_search_options,
    )
    step_size: float = line_search_result[0]

    step_size_history: list[float] = [step_size]
    cost_count_history: list[int] = [num_cost_evals]
    grad_count_history: list[int] = [num_grad_evals]

    def __display_iter_info():
        if print_level > 0:
            print(
                "Iter:",
                iter,
                ", cost:",
                np.format_float_scientific(f, precision=3, unique=False),
                ", |g|:",
                np.format_float_scientific(gradnorm, precision=3, unique=False),
                ", step_size:",
                (
                    None
                    if step_size is None
                    else np.format_float_scientific(
                        step_size, precision=3, unique=False
                    )
                ),
                ", using inv_hess0:",
                inv_hess.inv_hess0 is not None,
            )

    __display_iter_info()
    f0: float = f
    gradnorm0: float = gradnorm

    termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.MAXITER_REACHED
    while iter < max_iter:
        if step_size is None:
            termination_reason = LbfgsTerminationReason.LINESEARCH_FAILED
            break

        # One step forward
        iter += 1
        x_old = x
        f_old = f
        g_old = g

        # Update x, f, g
        x = _add(x, _componentwise_scalar_mult(p, step_size))  # x + step_size * p
        if callback is not None:
            callback(x)
        f = line_search_result[1]
        g = __grad_with_counter(x)
        gradnorm = _norm(g)

        cost_history.append(f)
        gradnorm_history.append(gradnorm)
        if checkpoint_options:
            np.save(os.path.join(log_path, f"x{iter}.npy"), x)
            np.save(os.path.join(log_path, f"f{iter}.npy"), f)
            np.save(os.path.join(log_path, f"g{iter}.npy"), g)

        if gradnorm <= rtol * gradnorm0:
            termination_reason: LbfgsTerminationReason = (
                LbfgsTerminationReason.RTOL_ACHIEVED
            )
            break

        if (f_old - f) < stag_tol * f_old:
            termination_reason: LbfgsTerminationReason = (
                LbfgsTerminationReason.DESCENT_STAGNATED
            )
            break

        # Update inv_hess
        if iter == num_initial_iter:
            if isinstance(inv_hess0, LbfgsInverseHessianApproximation):
                inv_hess = inv_hess0
            else:
                if isinstance(inv_hess0, LinearOperator):
                    inv_hess.inv_hess0 = inv_hess0
                else:
                    inv_hess.inv_hess0 = inv_hess0(x)

        inv_hess.add_new_s_y_pair(
            _sub(x, x_old), _sub(g, g_old)
        )  # s = x - x_old, y = g - g_old

        if inv_hess0_update_freq and (iter % inv_hess0_update_freq == 0):
            inv_hess.inv_hess0 = inv_hess0(x)

        # Compute p and line search step
        p = inv_hess.matvec(_neg(g))
        if checkpoint_options:
            np.save(os.path.join(log_path, f"p{iter}.npy"), p)
        line_search_result = _line_search(
            __cost_with_counter,
            __grad_with_counter,
            x,
            p,
            g,
            f,
            None,
            **line_search_options,
        )
        step_size = line_search_result[0]

        step_size_history.append(step_size)
        cost_count_history.append(num_cost_evals)
        grad_count_history.append(num_grad_evals)
        __display_iter_info()

    if print_level > 0:
        print("LBFGS done.")
        print("    Termination reason:", termination_reason.name)
        print("    Iterations:", iter)
        print("    Cost evaluations:", num_cost_evals)
        print("    Gradient evaluations:", num_grad_evals)
        print("    Final cost:", f)
        print("    Final |g|:", gradnorm)

    return LbfgsResult(
        x,
        f,
        g,
        inv_hess,
        iter,
        num_cost_evals,
        num_grad_evals,
        cost_history,
        gradnorm_history,
        step_size_history,
        termination_reason,
        cost_count_history,
        grad_count_history,
    )


class LbfgsTerminationReason(Enum):
    MAXITER_REACHED = 0
    RTOL_ACHIEVED = 1
    DESCENT_STAGNATED = 2
    LINESEARCH_FAILED = 3


class LbfgsResult(NamedTuple):
    x: VecType  # solution
    cost: float  # cost(x)
    grad: VecType  # grad f(x)
    inv_hess: LbfgsInverseHessianApproximation  # L-BFGS approximation to the inverse Hessian at x
    iter: int
    num_cost_evals: int
    num_grad_evals: int
    cost_history: list[float]  # [cost(x0), cost(x1), ..., cost(x)]
    gradnorm_history: list[float]  # [grad(x0), grad(x1), ..., grad(x)]
    step_size_history: list[
        float
    ]  # [a0, a1, ...], where x1 = x0 + a0*p, x2 = x1 + a1*p, ...
    termination_reason: LbfgsTerminationReason
    cost_count_history: list[int]
    grad_count_history: list[int]


@dataclass
class LbfgsInverseHessianApproximation:
    """See Nocedal and Wright page 177-179."""

    m: int  # max vector pairs stored
    ss: Deque[
        VecType
    ]  # GETS MODIFIED! ss=[s_(k-1), s_(k-2), ..., s_(k-m)], s_i = x_(i+1) - x_i. Eq 7.18, left, on page 177
    yy: Deque[
        VecType
    ]  # GETS MODIFIED! yy=[y_(k-1), y_(k-2), ..., y_(k-m)], y_i = grad f_(i+1) - grad f_i. Eq 7.18, right, on page 177
    inv_hess0: Callable[[VecType], VecType] = (
        None  # Initial inverse Hessian approximation
    )
    print_level: int = 1
    high_pass_filter: LinearOperator = None
    gamma_type: int = 2
    initial_gamma: float = 1.0

    def __post_init__(self) -> None:
        assert self.m >= 0
        assert len(self.ss) == len(self.yy)
        while len(self.ss) > self.m:
            self.ss.pop()
        while len(self.yy) > self.m:
            self.yy.pop()

    def add_new_s_y_pair(self, s: VecType, y: VecType) -> None:
        self.ss.appendleft(s)
        if len(self.ss) > self.m:
            self.ss.pop()

        self.yy.appendleft(y)
        if len(self.yy) > self.m:
            self.yy.pop()

    def apply_inv_hess0_k(self, x: VecType) -> VecType:
        if self.inv_hess0 is not None:
            if self.ss:
                if self.gamma_type == 0:
                    gamma_k = 1.0
                elif self.gamma_type == 1:
                    gamma_k = _inner_product(self.ss[0], self.yy[0]) / _inner_product(
                        self.yy[0], self.yy[0]
                    )
                elif self.gamma_type == 2:
                    gamma_k = _inner_product(self.ss[0], self.yy[0]) / _inner_product(
                        self.yy[0], self.inv_hess0 * self.yy[0]
                    )
                # if self.print_level > 0:
                #     print("(y,s)/(y,y) = ", gamma1)
                #     print("(y,s)/(y,Py) = ", gamma2)
                # if self.high_pass_filter is not None:
                #     F = self.high_pass_filter
                #     Fs = F * self.ss[0]
                #     Fy = F * self.yy[0]
                #     fs = self.ss[0] - Fs
                #     fy = self.yy[0] - Fy
                #     gamma3 = (Fs * Fy).sum() / (Fy * (self.inv_hess0 * Fy)).sum()
                #     gamma4 = (fs * fy).sum() / (fy * (self.inv_hess0 * fy)).sum()
                # if self.print_level > 0:
                #     print("(Fy,Fs)/(Fy,PFy) = ", gamma3)
                #     print("(fy,fs)/(fy,Pfy) = ", gamma4)
            else:
                gamma_k = self.initial_gamma
            if self.print_level > 0:
                print("gamma_k =", gamma_k)
            return _componentwise_scalar_mult(self.inv_hess0 * x, gamma_k)
        else:
            if self.ss:
                gamma_k = _inner_product(self.ss[0], self.yy[0]) / _inner_product(
                    self.yy[0], self.yy[0]
                )  # <s_(k-1), y_(k-1)> / <y_(k-1), y_(k-1)>
            else:
                gamma_k = 1.0
            if self.print_level > 0:
                print("gamma_k =", gamma_k)
            return _componentwise_scalar_mult(x, gamma_k)  # H0_k = gamma_k*I

    def matvec(self, q: VecType) -> VecType:
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

    # def cost_1D(s):
    #     return cost(_add(x, _componentwise_scalar_mult(p, s)))

    # def grad_1D(s):
    #     return _inner_product(p, grad(_add(x, _componentwise_scalar_mult(p, s))))

    # g0 = _inner_product(p, g)
    # stp, fval, old_fval = ls.scalar_search_wolfe1(
    #     cost_1D, grad_1D, f, old_old_fval, g0, **kwargs
    # )
    # return stp, fval
