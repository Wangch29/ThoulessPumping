from math import *
import numpy as np
from matplotlib import pyplot as plt
import FourSublatticePumpModel as ThoulessModel
from scipy.linalg import expm


class Simulator:
    model: ThoulessModel.ThoulessModel4  # The model
    init_state: np.ndarray  # The initial state
    Sites: int
    PERIOD: float
    DIV_NUMBER = 100000
    Time_Div: float
    TIMES: np.ndarray
    INIT_COORD: int

    def __init__(self, thouless_model: ThoulessModel.ThoulessModel4, init_state: np.ndarray):
        self.model = thouless_model
        self.PERIOD = self.model.period
        self.init_state = init_state
        self.Sites = len(init_state)
        self.Time_Div = self.PERIOD / self.DIV_NUMBER
        self.TIMES = np.arange(0, self.PERIOD, self.Time_Div)
        # Find initial place
        for i in range(0, self.Sites):
            if init_state[i] == 1:
                self.INIT_COORD = i
                break


    def __generate_matrix(self):
        matrix = np.zeros((self.Sites, len(self.TIMES)))

        init_p = self.init_state * np.conj(self.init_state)
        for i in range(0, self.Sites):
            matrix[i][0] = init_p[i]

        for j in range(1, len(self.TIMES)):
            t = self.TIMES[j]
            self.init_state = np.dot(expm(-1j * self.model.single_hamiltonian(t) * self.Time_Div),
                                     self.init_state)
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

        # First plot
        plt.subplot(2, 1, 1)
        matrix = self.__generate_matrix()
        plt.imshow(matrix, interpolation='None', cmap='hot', origin='lower', aspect='auto')
        plt.colorbar(shrink=0.9)

        plt.xticks(np.arange(0, self.DIV_NUMBER, self.DIV_NUMBER/4), np.arange(0, 1, 0.25))
        plt.xlabel("t/Tp")
        plt.ylim(0, self.Sites)
        plt.ylabel("Site")

        # Second plot
        plt.subplot(2, 1, 2)
        x_expected = np.zeros(len(self.TIMES))
        for i in range(len(self.TIMES)):
            x_expected[i] = self.__x_expected(matrix[:, i])
        plt.scatter(self.TIMES / self.PERIOD, x_expected - self.INIT_COORD, s=1)

        plt.xlim(0, 1)
        plt.ylim(0, 5)
        plt.ylabel(r'$\Delta P/d$')
        plt.xlabel("t/Tp")

        print("The step for a period is %.2f" % (x_expected[self.DIV_NUMBER-1]-x_expected[0]))

        plt.show()
