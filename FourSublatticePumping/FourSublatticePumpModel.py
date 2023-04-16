import numpy as np
import scipy.linalg as linalg
from typing import Tuple
from math import *


class ThoulessModel4:
    h_0: float
    delta_0: float
    J0: float
    period: float
    # The number of sites.
    sites: int
    omega: float

    def __init__(self, h_0: float, delta_0: float, J0: float, period: float, sites: int):
        self.h_0 = h_0
        self.delta_0 = delta_0
        self.J0 = J0
        self.period = period
        self.sites = sites
        self.omega = 2 * pi / self.period
        assert sites > 0 and sites % 4 == 0  # The number of sites is a multiple of four.

    '''
    Return the staggered on-site energy offset.
    '''

    def h_st(self, t: float, i: int) -> float:
        return self.h_0 * np.cos(self.omega * t + pi * i / 2)

    '''
    Return the Hamiltonian of this model, in form of matrix, in time t.
    '''

    def single_hamiltonian(self, t: float) -> np.ndarray:
        hamiltonian = np.zeros((self.sites, self.sites))
        # Fill the diagonal elements:
        for i in range(0, self.sites):
            hamiltonian[i, i] = self.h_st(t, i)
        # Fill the non-diagonal elements.
        for i in range(0, self.sites - 1):
            hamiltonian[i, i + 1] = self.J0
            hamiltonian[i + 1, i] = np.conj(hamiltonian[i, i + 1])

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

    def bulk_levels(self, t) -> np.ndarray:
        k_points = np.linspace(-np.pi, np.pi, self.sites // 4 + 1)
        eigenvalues = np.zeros((self.sites // 4 + 1, 4))
        i = 0
        for k in k_points:
            values = sorted(np.real(np.linalg.eigvals(self.bulk_hamiltonian(k, t))))
            for n in range(4):
                eigenvalues[i][n] = values[n]
            i += 1

        return eigenvalues

    def bulk_hamiltonian(self, k, t):
        omega = 2 * np.pi * t / self.period

        matrix = np.zeros((4, 4), dtype=complex)
        # Fill the diagonal elements.
        matrix[0, 0] = self.h_0 * cos(omega)
        matrix[1, 1] = -self.h_0 * sin(omega)
        matrix[2, 2] = -self.h_0 * cos(omega)
        matrix[3, 3] = self.h_0 * sin(omega)
        # Fill the non-diagonal elements
        matrix[0, 1] = e ** (1j * k)
        matrix[1, 0] = np.conj(matrix[0, 1])
        matrix[1, 2] = e ** (1j * k)
        matrix[2, 1] = np.conj(matrix[1, 2])
        matrix[2, 3] = e ** (1j * k)
        matrix[3, 2] = np.conj(matrix[2, 3])
        matrix[3, 0] = e ** (1j * k)
        matrix[0, 3] = np.conj(matrix[3, 0])

        return matrix
