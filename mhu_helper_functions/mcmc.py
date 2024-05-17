"""MCMC with progess bar.

Inspired by hippylib.mcmc
"""

from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm

from .mcmc_diagonstics import integratedAutocorrelationTime


class SampleStruct:
    """A structure to store the sample and related computation, to avoid repeated computation."""

    def __init__(self):
        self.m: np.ndarray = None

    def assign(self, other: SampleStruct):
        self.cost = other.cost
        self.m = other.m.copy()


class MCMC:
    def __init__(
        self,
        kernel: gpCNKernel,
        nsamples: int = 2000,
        nburnin: int = 0,
        ncheck: int = 100,
        print_level: int = 1,
    ):
        """Create MCMC sampling model.

        Args:
            kernel (gpCNKernel): The kernel.
            nsamples (int, optional): Number of samples. Defaults to 2000.
            nburnin (int, optional): Number of burn-in samples. Defaults to 0.
            ncheck (int, optional): Number of samples between prints. Defaults to 100.
            print_level (int, optional): 0: no print; >0: print. Defaults to 1.
        """
        self.kernel = kernel
        self.nsamples = nsamples
        self.nburnin = nburnin
        self.ncheck = ncheck
        self.print_level = print_level

    def run(self, m0: np.ndarray, qoi=None, tracer=None):
        if qoi is None:
            qoi = NullQoi()
        if tracer is None:
            tracer = NullTracer()

        current = SampleStruct()
        proposed = SampleStruct()
        current.m = m0.copy()
        self.kernel.init_sample(current)

        if self.print_level > 0:
            print(f"Burn {self.nburnin} samples.")
        sample_count = 0
        naccept = 0
        while sample_count < self.nburnin:
            naccept += self.kernel.sample(current, proposed)
            sample_count += 1
            if sample_count % self.ncheck == 0 and self.print_level > 0:
                print(
                    "{0:2.1f} % completed, Acceptance ratio {1:2.1f} %".format(
                        float(sample_count) / float(self.nburnin) * 100,
                        float(naccept) / float(sample_count) * 100,
                    )
                )

        if self.print_level > 0:
            print(f"Generate {self.nsamples} samples")
        sample_count = 0
        pbar = tqdm(total=self.nsamples)  # Add tqdm progress bar
        naccept = 0
        while sample_count < self.nsamples:
            naccept += self.kernel.sample(current, proposed)
            q = qoi.eval(current)
            tracer.append(current, q)
            sample_count += 1
            if sample_count % self.ncheck == 0:
                pbar.update(self.ncheck)
                if self.print_level > 0:
                    print(
                        "{0:2.1f} % completed, Acceptance ratio {1:2.1f} %".format(
                            float(sample_count) / float(self.nsamples) * 100,
                            float(naccept) / float(sample_count) * 100,
                        )
                    )
        pbar.close()  # Close pqdm progress bar
        return naccept

    def consume_random(self, nsamples: int):
        for ii in range(nsamples):
            self.kernel.consume_random()


class NullQoi:
    def eval(self, current: SampleStruct):
        return 0.0


class NullTracer:
    def append(self, current: SampleStruct, q):
        pass


class FullTracer:
    """A tracer with checkpoints."""

    def __init__(
        self,
        nsamples: int,
        dim: int,
        checkpoint_interval: int = None,
        checkpoint_path: str = None,
        load_existing: bool = False,
    ):
        self.nsamples = nsamples
        self.dim = dim
        self.interval = checkpoint_interval
        self.path = checkpoint_path
        if self.path is not None:
            os.makedirs(self.path, exist_ok=True)

        self.i: int = 0
        self.data: np.ndarray = np.zeros((nsamples, dim), dtype=float)

        if load_existing:
            self.load_existing(self.search_for_existing())

    def _file_path(self, start_index: int) -> str:
        file_name = f"tracer_{start_index}_{start_index+self.interval-1}.npy"
        return os.path.join(self.path, file_name)

    def append(self, current: SampleStruct, q):
        self.data[self.i, :] = current.m
        self.i += 1
        if self.interval is not None:
            if self.i % self.interval == 0:
                sta = self.i - self.interval
                end = self.i
                np.save(
                    self._file_path(sta),
                    self.data[sta:end, :],
                )

    def load_existing(self, nsamples: int) -> int:
        self.i = 0
        while self.i < nsamples:
            sta = self.i
            end = self.i + self.interval
            path = self._file_path(sta)
            if os.path.isfile(path):
                self.data[sta:end, :] = np.load(path)
                self.i += self.interval
            else:
                break
        print(f"Load {self.i} existing samples.")
        return self.i

    def search_for_existing(self) -> int:
        n = 0
        while True:
            if os.path.isfile(self._file_path(n)):
                n += self.interval
            else:
                break
        return n

    def load(self, i: int) -> np.ndarray:
        return self.data[i]


class FullTracerSmallMemory(FullTracer):
    """A tracer with checkpoints and small memory"""

    def __init__(
        self,
        nsamples: int,
        dim: int,
        checkpoint_interval: int,
        checkpoint_path: str,
        load_existing: bool = True,
    ):
        self.nsamples = nsamples
        self.dim = dim
        self.interval = checkpoint_interval
        self.path = checkpoint_path
        os.makedirs(self.path, exist_ok=True)

        self.i: int = 0
        self.data: np.ndarray = np.zeros((self.interval, self.dim), dtype=float)

        if load_existing:
            self.i = self.search_for_existing()

    def append(self, current: SampleStruct, q):
        self.data[self.i % self.interval, :] = current.m
        self.i += 1

        if self.i % self.interval == 0:
            sta = self.i - self.interval
            end = self.i
            np.save(self._file_path(sta), self.data)
            self.data[:] = 0.0

    def load_existing(self, nsamples: int) -> int:
        self.i = nsamples
        print(f"Load {self.i} existing samples.")
        return self.i

    def load(self, i: int) -> np.ndarray:
        path = self._file_path(i // self.interval * self.interval)
        data = np.load(path)
        return data[i % self.interval]


class GaussianPrior:

    def __init__(self, R, sqrtRinv, mean):
        self.R = spla.aslinearoperator(R)
        self.sqrtRinv = spla.aslinearoperator(sqrtRinv)
        self.mean = np.asarray(mean)

    def cost(self, m: np.ndarray):
        dm = m - self.mean
        return 0.5 * np.dot(self.R * dm, dm)

    def sample(self, add_mean: bool = True):
        noise = np.random.randn(self.sqrtRinv.shape[1])
        if add_mean:
            return self.sqrtRinv * noise + self.mean
        else:
            return self.sqrtRinv * noise


class gpCNKernel:
    """hippylib.mcmc.kernels"""

    def __init__(
        self,
        total_cost: Callable[[np.ndarray], float],
        nu: GaussianPrior,
        s: float = 0.1,
    ):
        """Create a gpCN kernel with posterior and approximated posterior infromation.

        Args:
            total_cost (Callable[[np.ndarray], float]): The negative log-posterior, i.e. the total cost.
            nu (GaussianPrior): Laplacian approximation posterior.
            s (float): update rate
        """
        self.total_cost = total_cost
        self.nu = nu
        self.s = s

    def init_sample(self, sample: SampleStruct):
        sample.cost = self.total_cost(sample.m)

    def sample(self, current: SampleStruct, proposed: SampleStruct) -> bool:
        proposed.m = self.proposal(current.m)
        self.init_sample(proposed)
        alpha = self.delta(current) - self.delta(proposed)
        if alpha > np.log(np.random.rand()):
            current.assign(proposed)
            return True
        else:
            return False

    def delta(self, sample: SampleStruct) -> float:
        return sample.cost - self.nu.cost(sample.m)

    def proposal(self, m: np.ndarray) -> np.ndarray:
        # Generate sample from the approximated posterior
        w = self.nu.sample(add_mean=False)
        # do pCN linear combination with current sample
        return (
            self.s * w
            + np.sqrt(1.0 - self.s * self.s) * (m - self.nu.mean)
            + self.nu.mean
        )

    def consume_random(self):
        # assuming that applying sqrtRinv doesn't involve random.
        np.random.randn(self.nu.sqrtRinv.shape[1])
        np.random.rand()


# Analysis of samples
def plot_trace(q, *args, **kwargs):
    plt.plot(q, "*", *args, **kwargs)
    plt.title("Trace plot")
    plt.xlabel("samples")


def plot_hist(q, **kwargs):
    sns.histplot(q, kde=False, bins=20, stat="density", **kwargs)
    mu, std = norm.fit(q)
    x = np.linspace(q.min(), q.max(), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=1)
    plt.title("Histogram")


def plot_autocorrelation(q, max_lag=300, **kwargs):
    IAT, lags, acorrs = integratedAutocorrelationTime(q, max_lag=max_lag)
    plt.plot(lags, acorrs, "-", **kwargs)
    plt.title("Autocorrelation")
    plt.ylim([0.0, 1.0])
    print(f"Autocorrelation, IAT = {IAT:.2f}")


class pCNKernel:
    """hippylib.mcmc.kernels"""

    def __init__(
        self,
        misfit_cost: Callable[[np.ndarray], float],
        nu: GaussianPrior,
        s: float = 0.1,
    ):
        """Create a gpCN kernel with posterior and approximated posterior infromation.

        Args:
            misfit_cost (Callable[[np.ndarray], float]): The negative log-likelihood, i.e. the misfit cost.
            nu (GaussianPrior): Prior.
            s (float): update rate
        """
        self.misfit_cost = misfit_cost
        self.nu = nu
        self.s = s

    def init_sample(self, sample: SampleStruct):
        sample.cost = self.misfit_cost(sample.m)

    def sample(self, current: SampleStruct, proposed: SampleStruct) -> bool:
        proposed.m = self.proposal(current.m)
        self.init_sample(proposed)
        alpha = current.cost - proposed.cost
        if alpha > np.log(np.random.rand()):
            current.assign(proposed)
            return True
        else:
            return False

    def proposal(self, m: np.ndarray) -> np.ndarray:
        # Generate sample from the approximated posterior
        w = self.nu.sample(add_mean=False)
        # do pCN linear combination with current sample
        return (
            self.s * w
            + np.sqrt(1.0 - self.s * self.s) * (m - self.nu.mean)
            + self.nu.mean
        )

    def consume_random(self):
        # assuming that applying sqrtRinv doesn't involve random.
        np.random.randn(self.nu.sqrtRinv.shape[1])
        np.random.rand()


# Analysis of samples
def plot_trace(q, *args, **kwargs):
    plt.plot(q, "*", *args, **kwargs)
    plt.title("Trace plot")
    plt.xlabel("samples")


def plot_hist(q, **kwargs):
    sns.histplot(q, kde=False, bins=20, stat="density", **kwargs)
    mu, std = norm.fit(q)
    x = np.linspace(q.min(), q.max(), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=1)
    plt.title("Histogram")


def plot_autocorrelation(q, max_lag=300, **kwargs):
    IAT, lags, acorrs = integratedAutocorrelationTime(q, max_lag=max_lag)
    plt.plot(lags, acorrs, "-", **kwargs)
    plt.title("Autocorrelation")
    plt.ylim([0.0, 1.0])
    print(f"Autocorrelation, IAT = {IAT:.2f}")
