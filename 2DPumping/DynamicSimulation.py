from math import *

from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt, animation
import TwoDModel as ThoulessModel
from scipy.linalg import expm


class Simulator:
    model: ThoulessModel.TwoDimensionalModel  # The model
    init_state: np.ndarray  # The initial state
    Sites: int
    NUMBER: int
    PERIOD: float
    # Time div
    Time_Div: float
    Div_Number = 100

    def __init__(self, thouless_model: ThoulessModel.TwoDimensionalModel, init_state: np.ndarray):
        self.model = thouless_model
        self.times = np.arange(125)
        self.Sites = len(init_state)
        self.NUMBER = self.Sites * self.Sites
        self.PERIOD = self.model.period
        self.Time_Div = self.PERIOD / self.Div_Number

        # Change two-dimensional index to one-dimensional
        self.init_state = np.zeros((self.NUMBER, 1))
        for i in range(0, self.Sites):
            for j in range(0, self.Sites):
                index = self.index_two_to_one(i, j)
                self.init_state[index] = init_state[i][j]

    '''
    Change two-dimensional index to one-dimensional
    '''

    def index_two_to_one(self, row: int, col: int) -> int:
        return row + col * self.Sites

    '''
        Change one-dimensional index to two-dimensional, in format of row, col.
    '''

    def index_one_to_two(self, index: int):
        return index % self.Sites, index // self.Sites

    def __generate_matrix(self, t: float):
        matrix = np.zeros((self.Sites, self.Sites))

        self.init_state = np.dot(expm(-1j * self.model.single_hamiltonian(t) * self.Time_Div),
                                 self.init_state)

        current_possibility = self.init_state * np.conj(self.init_state)
        for index in range(0, self.NUMBER):
            i, j = self.index_one_to_two(index)
            matrix[i][j] = np.real(current_possibility[index])

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
        fig = plt.figure()
        times = np.linspace(0, 10 * self.PERIOD, 10 * self.Div_Number)

        norm = colors.Normalize(vmin=0, vmax=1)

        # First plot
        # fig1 = fig.add_subplot(2, 1, 1)
        frames = []
        for t in times:
            matrix = self.__generate_matrix(t)
            frame = plt.imshow(matrix, interpolation='None', norm=norm, cmap='hot', origin='lower', aspect='auto',
                               animated='True')
            frames.append([frame])

        ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=50, blit=True)
        plt.colorbar(shrink=0.9)
        plt.xticks(np.arange(0, self.Sites, self.Sites / 4))
        plt.yticks(np.arange(0, self.Sites, self.Sites / 4))

        pw_writer = animation.PillowWriter(fps=20)
        ani.save('2DPumping.gif', writer=pw_writer)

        # Second plot
        # fig2 = fig.add_subplot(2, 1, 2)
        '''
        x_expected = np.zeros(len(self.times))
        for i in range(len(self.times)):
            x_expected[i] = self.__x_expected(matrix[:, i])
        plt.scatter(self.times / 100, x_expected)

        plt.ylim(18, 25)
        plt.ylabel("sites")
        plt.xlabel("t/Tp")

        print("The step for a period is %.2f" % (x_expected[100] - x_expected[0]))
'''
        plt.show()
