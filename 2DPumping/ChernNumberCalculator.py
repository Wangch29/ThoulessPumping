import time
import numpy as np
import matplotlib.pyplot as plt
from math import *
import cmath
import TwoDModel as ThoulessModel

'''
Return the hamiltonian of the model.
'''

# The pumping period
period: float
# The length of one cell
d: float
# The Thouless Model
MODEL: ThoulessModel.TwoDimensionalModel


def initialization(model: ThoulessModel.TwoDimensionalModel):
    global period, MODEL, d

    period = model.period
    MODEL = model
    # d = (model.d + model.D) / 2
    d = 2

    main()


def hamiltonian(kx, ky):
    return MODEL.bulk_hamiltonian(kx, kx, ky)


def main():
    start_time = time.time()
    n = 100  # 积分密度
    delta = 1e-9  # 求导的偏离量
    chern_number = 0  # 陈数初始化
    for kx in np.arange(-pi / d, pi / d, 2 * pi / (d * n)):
        for ky in np.arange(-period / 2, period / 2, period / n):
            H = hamiltonian(kx, ky)
            eigenvalue, eigenvector = np.linalg.eig(H)
            vector = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 价带波函数

            H_delta_kx = hamiltonian(kx + delta, ky)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_kx)
            vector_delta_kx = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离kx的波函数

            H_delta_ky = hamiltonian(kx, ky + delta)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_ky)
            vector_delta_ky = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离ky的波函数

            H_delta_kx_ky = hamiltonian(kx + delta, ky + delta)
            eigenvalue, eigenvector = np.linalg.eig(H_delta_kx_ky)
            vector_delta_kx_ky = eigenvector[:, np.argsort(np.real(eigenvalue))[0]]  # 略偏离kx和ky的波函数

            index = np.argmax(np.abs(vector))
            precision = 0.0001
            vector = find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=precision, index=index)
            vector_delta_kx = find_vector_with_fixed_gauge_by_making_one_component_real(vector_delta_kx,
                                                                                        precision=precision,
                                                                                        index=index)
            vector_delta_ky = find_vector_with_fixed_gauge_by_making_one_component_real(vector_delta_ky,
                                                                                        precision=precision,
                                                                                        index=index)
            vector_delta_kx_ky = find_vector_with_fixed_gauge_by_making_one_component_real(vector_delta_kx_ky,
                                                                                           precision=precision,
                                                                                           index=index)

            # 价带的波函数的贝里联络(berry connection) # 求导后内积
            A_x = np.dot(vector.transpose().conj(), (vector_delta_kx - vector) / delta)  # 贝里联络Ax（x分量）
            A_y = np.dot(vector.transpose().conj(), (vector_delta_ky - vector) / delta)  # 贝里联络Ay（y分量）

            A_x_delta_ky = np.dot(vector_delta_ky.transpose().conj(),
                                  (vector_delta_kx_ky - vector_delta_ky) / delta)  # 略偏离ky的贝里联络Ax
            A_y_delta_kx = np.dot(vector_delta_kx.transpose().conj(),
                                  (vector_delta_kx_ky - vector_delta_kx) / delta)  # 略偏离kx的贝里联络Ay

            # 贝里曲率(berry curvature)
            F = (A_y_delta_kx - A_y) / delta - (A_x_delta_ky - A_x) / delta

            # 陈数(chern number)
            chern_number = chern_number + F * 2 * pi / (d * n) * period / n
    chern_number = chern_number / (2 * pi * 1j)
    print('Chern number = ', chern_number)
    end_time = time.time()
    print('Running time = %.3fs' % (end_time - start_time))


def find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.005, index=None):
    vector = np.array(vector)
    if index == None:
        index = np.argmax(np.abs(vector))
    sign_pre = np.sign(np.imag(vector[index]))
    for phase in np.arange(0, 2 * np.pi, precision):
        sign = np.sign(np.imag(vector[index] * cmath.exp(1j * phase)))
        if np.abs(np.imag(vector[index] * cmath.exp(1j * phase))) < 1e-9 or sign == -sign_pre:
            break
        sign_pre = sign
    vector = vector * cmath.exp(1j * phase)
    if np.real(vector[index]) < 0:
        vector = -vector
    return vector


if __name__ == '__main__':
    main()
