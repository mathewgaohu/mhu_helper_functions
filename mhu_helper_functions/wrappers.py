import functools
from typing import Callable

import numpy as np


def count_calls(func):
    """Decorator to count the number of times a function is called.

    Usage:
    @count_calls
    def f(x):
        ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.ncalls += 1  # Increment call count
        return func(*args, **kwargs)

    wrapper.ncalls = 0  # Initialize the call counter
    return wrapper


def memoize_with_tol(tol=1e-15):
    """Decorator to memoize a function's last result if input is within tolerance.

    If the function is called with a new input `x` that is within a relative tolerance `tol`
    compared to the last input `x_last`, it returns the cached result instead of recomputing.

    The relative tolerance is computed as:
        norm(x - last_x) <= tol * max(norm(x), norm(last_x))

    Usage:
    @memoize_with_tol(tol=0.01)
    def f(x):
        ...
    """

    def compute_norm(x: np.ndarray) -> float:
        if x.ndim == 1:
            return np.linalg.norm(x)
        else:
            return np.linalg.norm(x, "fro")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(x: np.ndarray):
            last_x = getattr(wrapper, "last_x", None)
            last_result = getattr(wrapper, "last_result", None)

            if last_x is not None:
                norm_x = compute_norm(x)
                norm_last_x = compute_norm(last_x)
                norm_diff = compute_norm(x - last_x)

                if norm_diff <= tol * max(norm_x, norm_last_x):
                    return last_result  # Return cached result

            # Compute and store new result
            wrapper.ncalls += 1  # Increment call count
            wrapper.last_x = x.copy()
            wrapper.last_result = func(x)
            return wrapper.last_result

        wrapper.ncalls = 0  # Initialize the call counter
        return wrapper

    return decorator


class FunctionWithMemory:
    def __init__(self, func: Callable, tol: float = 1e-15):
        self.func = func
        self.tol = tol
        self.last_x: np.ndarray = None
        self.last_out = None
        self.count: int = 0

    def __call__(self, x: np.ndarray):
        if self.last_x is not None:
            if l2norm(x - self.last_x) <= self.tol * np.max(
                [l2norm(x), l2norm(self.last_x)]
            ):
                return self.last_out
        self.last_x = x.copy()
        self.last_out = self.func(x)
        self.count += 1
        return self.last_out


def l2norm(x: np.ndarray) -> float:
    return np.sqrt(np.sum(x * x))
