import numpy as np
import scipy.linalg as linalg
from typing import Tuple
from math import *


class TwoDimensionalModel:
    h_0: float
    delta_0: float
    J0: float
    period: float
    # The number of sites.
    sites: int
    w: float

    J_inter: float
    D: float
    d: float

    def __init__(self, h_0: float, delta_0: float, J0: float, period: float, sites: int,
                 J_inter: float, D: float, d: float):
        self.h_0 = h_0
        self.delta_0 = delta_0
        self.J0 = J0
        self.period = period
        self.sites = sites
        # Two Dimensional elements
        self.D = D
        self.d = d
        self.J_inter = J_inter
        #
        self.w = 2 * pi / self.period
        assert sites > 0 and sites % 2 == 0  # The number of sites is a multiple of two.

    '''
    Return the staggered on-site energy offset.
    '''

    def h_st(self, t: float, i: int) -> float:
        return self.h_0 * np.cos(self.w * t + pi * i / 2)

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
        kx_points = np.linspace(-np.pi, np.pi, self.sites // 2 + 1)
        ky_points = np.linspace(-np.pi, np.pi, self.sites // 2 + 1)
        eigenvalues = np.zeros((self.sites // 2 + 1, self.sites // 2 + 1, 4))
        i = 0
        j = 0
        for kx in kx_points:
            for ky in ky_points:
                values = sorted(np.real(np.linalg.eigvals(self.bulk_hamiltonian(kx, ky, t))))
                for n in range(4):
                    eigenvalues[i][j][n] = values[n]
                j += 1
            i += 1
            j = 0

        return eigenvalues

    def bulk_hamiltonian(self, kx, ky, t):
        matrix = np.zeros((4, 4), dtype=complex)
        # Fill the diagonal elements.
        matrix[0, 0] = self.h_0 * cos(self.w * t)
        matrix[1, 1] = -self.h_0 * sin(self.w * t)
        matrix[2, 2] = -self.h_0 * cos(self.w * t)
        matrix[3, 3] = self.h_0 * sin(self.w * t)
        # Fill the non-diagonal elements
        matrix[0, 1] = matrix[2][3] \
            = self.J_inter * e ** (-1j * kx * self.D) + self.J0 * e ** (1j * kx * self.d)
        matrix[1, 0] = np.conj(matrix[0, 1])
        matrix[3, 2] = np.conj(matrix[2, 3])
        matrix[0, 3] = matrix[1][2] \
            = self.J0 * e ** (-1j * ky * self.d) + self.J_inter * e ** (1j * ky * self.D)
        matrix[3, 0] = np.conj(matrix[0, 3])
        matrix[2, 1] = np.conj(matrix[1, 2])

        return matrix
