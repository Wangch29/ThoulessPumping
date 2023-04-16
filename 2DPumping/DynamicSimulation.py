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
    Div_Number = 1000

    def __init__(self, thouless_model: ThoulessModel.TwoDimensionalModel, init_state: np.ndarray):
        self.model = thouless_model
        self.times = np.arange(3000)
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

    def format_func(self, value, tick_number):
        # N是pi/2的倍数
        N = value
        if N == 0:
            return "0"  # 0点
        elif N == 2:
            return r"$\pi$"  # pi
        elif N == -2:
            return r"$-\pi$"  # -pi

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

    def __x_expected(self, matrix: np.ndarray) -> float:
        X = 0.0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                X += i * matrix[i][j]
        return X

    def __y_expected(self, matrix: np.ndarray) -> float:
        Y = 0.0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                Y += i * matrix[j][i]
        return Y

    '''
    Generate the animation plot
    '''

    def generate_plot(self):
        fig = plt.figure()
        times = np.linspace(0, 3 * self.PERIOD, 3 * self.Div_Number)

        plt.style.use('ggplot')

        norm = colors.Normalize(vmin=0, vmax=1)

        # First plot
        # fig1 = fig.add_subplot(2, 1, 1)
        '''
        frames = []
        for t in times:
            matrix = self.__generate_matrix(t)
            frame = plt.imshow(matrix, interpolation='None', norm=norm, cmap='hot', origin='lower', aspect='auto',
                               animated='True')
            plt.title('%.2f' % t)
            frames.append([frame])

        ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=50, blit=True)
        plt.colorbar(shrink=0.9)
        plt.xticks(np.arange(0, self.Sites, self.Sites / 4))
        plt.yticks(np.arange(0, self.Sites, self.Sites / 4))

        pw_writer = animation.HTMLWriter(fps=20)
        ani.save('2DPumping.html', writer=pw_writer)
'''
        # Second plot
        fig2 = fig.add_subplot(2, 1, 1)
        x_expected = []
        y_expected = []
        for t in times:
            matrix = self.__generate_matrix(t)
            x_expected.append(self.__x_expected(matrix))
            # y_expected.append(self.__y_expected(matrix))

        fig2.scatter(self.times / self.Div_Number, x_expected, s=1)

        plt.ylim(9, 13)
        plt.ylabel("X")
        plt.xlabel("t/Tp")

        '''
        fig3 = fig.add_subplot(2, 1, 2)
        fig3.scatter(self.times / self.Div_Number, y_expected, s=1)

        plt.ylim(9, 13)
        plt.ylabel("Y")
        plt.xlabel("t/Tp")
 '''
        plt.show()
