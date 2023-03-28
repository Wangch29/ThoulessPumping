import numpy as np
import scipy.linalg as linalg
from typing import Tuple


class ThoulessModel:
    h_0: float
    delta_0: float
    t_0: float
    period: float
    # The number of sites.
    sites: int

    def __init__(self, h_0: float, delta_0: float, t_0: float, period: float, sites: int):
        self.h_0 = h_0
        self.delta_0 = delta_0
        self.t_0 = t_0
        self.period = period
        self.sites = sites
        assert sites > 0 and sites % 2 == 0

    '''
    Return the cosine part of J1/J2.
    '''
    def delta(self, t: float) -> float:
        return self.delta_0 * np.sin(2 * np.pi * t / self.period)

    '''
    Return the staggered on-site energy offset.
    '''
    def h_st(self, t: float) -> float:
        return self.h_0 * np.cos(2 * np.pi * t / self.period)

    '''
    Return the Hamiltonian of this model, in form of matrix, in time t.
    '''
    def single_hamiltonian(self, t: float) -> np.ndarray:
        hamiltonian = np.zeros((self.sites, self.sites))
        # Fill the diagonal elements.
        for i in range(0, self.sites):
            hamiltonian[i, i] = (-1)**i * self.h_st(t)
        # Fill the non-diagonal elements.
        for i in range(0, self.sites - 1):
            hamiltonian[i, i + 1] = (self.t_0 + self.delta(t) * (-1)**i)
            hamiltonian[i + 1, i] = np.conj(hamiltonian[i, i+1])
        return hamiltonian

    '''
    Diagonalize the single-hamiltonian matrix.
    Return a Tuple containing energy_levels(eigenvalues) and vectors(eigenvectors).
    '''
    def single_exact_diagonalization(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        single_hamiltonian: np.ndarray = self.single_hamiltonian(t)
        energy_levels, vectors = linalg.eigh(single_hamiltonian)
        return energy_levels, vectors.T

    '''
    Calculate bulk levels, and return them in form of a two-dimensional array: [E+_array, E-_array].
    '''
    def bulk_levels(self, t: float) -> np.ndarray:
        k_points = np.linspace(-np.pi, np.pi, self.sites // 2 + 1)
        levels = np.sqrt(self.h_st(t) ** 2 + self.delta(t) ** 2 *
                         np.sin(k_points / 2) ** 2 + self.t_0 ** 2 * np.cos(k_points / 2) ** 2)
        return np.concatenate(
            (levels, -levels)
        )
