from math import *
import numpy as np
from matplotlib import pyplot as plt
import ThoulessModel
from scipy.linalg import expm


class Simulator:
    model: ThoulessModel.ThoulessModel  # The model
    init_state: np.ndarray  # The initial state
    Sites: int

    def __init__(self, thouless_model: ThoulessModel.ThoulessModel, init_state: np.ndarray):
        self.model = thouless_model
        self.init_state = init_state
        self.times = np.arange(150)
        self.Sites = len(init_state)

    def __generate_matrix(self):
        matrix = np.zeros((self.Sites, len(self.times)))

        init_p = self.init_state * np.conj(self.init_state)
        for i in range(0, self.Sites):
            matrix[i][0] = init_p[i]

        for j in range(1, len(self.times)):
            t = self.times[j] * self.model.period / 100
            self.init_state = np.dot(expm(-1j * self.model.single_hamiltonian(t) * self.model.period/100), self.init_state)
            current_p = self.init_state * np.conj(self.init_state)
            for i in range(0, self.Sites):
                matrix[i][j] = np.real(current_p[i])

        return matrix

    def __x_expected(self, vector: np.ndarray) -> float:
        x = 0.0
        for i in range(len(vector)):
            x += i * vector[i]
        return x

    '''
    Generate the animation plot
    '''
    def generate_plot(self):
        plt.figure()

        plt.subplot(2, 1, 1)
        matrix = self.__generate_matrix()
        plt.imshow(matrix, interpolation='None', cmap='hot', origin='lower', aspect='auto')
        plt.colorbar(shrink=0.9)
        plt.xticks(np.arange(0, len(self.times), 25))

        plt.subplot(2, 1, 2)
        x_expected = np.zeros(len(self.times))
        for i in range(len(self.times)):
            x_expected[i] = self.__x_expected(matrix[:, i])
        plt.scatter(self.times/100, x_expected)

        plt.ylim(5, 9)
        plt.ylabel("sites")
        plt.xlabel("t/Tp")

        plt.show()



