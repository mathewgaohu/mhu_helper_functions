"""Clean results of iterative solvers for optimization."""

from pathlib import Path

import numpy as np


def clean_map(output_dir: str):
    """Clean results of iterative solvers (lbfgs and newton).

    1. Concatenate all checkpoints x, f, g, and p.
    Note that the number of x is often one more than the number of p (when converged).
    Their number are same if line search fails (not a good result)
    """
    path = Path(output_dir)
    save = lambda x, name: np.save(path / name, x)
    load = lambda name: np.load(path / f"{name}.npy")

    N = len([p for p in path.glob("x*.npy") if p.stem[1:].isdigit()])
    print(f"Found {N} x arrays.")
    if N > 0:
        xx = np.asarray([load(f"x{i}") for i in range(N)])
        ff = np.asarray([load(f"f{i}") for i in range(N)])
        gg = np.asarray([load(f"g{i}") for i in range(N)])
        save(xx, "xx")
        save(ff, "ff")
        save(gg, "gg")
    for i in range(N):
        (path / f"x{i}.npy").unlink()
        (path / f"f{i}.npy").unlink()
        (path / f"g{i}.npy").unlink()

    N = len([p for p in path.glob("p*.npy") if p.stem[1:].isdigit()])
    print(f"Found {N} p arrays.")
    if N > 0:
        pp = np.asarray([load(f"p{i}") for i in range(N)])
        save(pp, "pp")
    for i in range(N):
        (path / f"p{i}.npy").unlink()
