"""Clean results of iterative solvers for optimization."""

from pathlib import Path

import numpy as np


def stack_vectors(folder_path: str, name: str):
    """Stack all name*.npy under the path. Then remove singe files."""
    path = Path(folder_path)
    N = len([p for p in path.glob(f"{name}*.npy") if p.stem[len(name) :].isdigit()])
    print(f"Found {N} {name} numpy files.")
    if N > 0:
        xx = np.stack([np.load(path / f"{name}{i}.npy") for i in range(N)])
        np.save(path / f"{name}{name}.npy", xx)
    for i in range(N):
        (path / f"{name}{i}.npy").unlink()


def clean_map(output_dir: str):
    stack_vectors(output_dir, "x")
    stack_vectors(output_dir, "f")
    stack_vectors(output_dir, "g")
    stack_vectors(output_dir, "p")
