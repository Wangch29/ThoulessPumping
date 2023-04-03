from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import TwoDModel as Tm
import ChernNumberCalculator
import DynamicSimulation

# Initialize the Thouless Model
h_0 = 10
J_0 = 1
w = 0.03
period = 2 * np.pi / w
sites = 20  # The number of sub-lattices for one dimension
J_inter = 0.01  # The interacting parameter between cells
D = 1  # The distance between cells
d = 1  # The distance between sub-lattices inside a cell
model = Tm.TwoDimensionalModel(h_0, J_0, period, sites, J_inter, D, d)

# Create the time array.
steps = 100
duration = period
times = np.linspace(0, duration, steps)

'''
Get the exact energy levels and vectors according to times.

Every time node in 'exact_levels_and_vectors_by_time' array is a tuple containing two sub-tuples: 
first tuple contains the energy levels, 
second tuple contains eigenvectors.

exact_levels_and_vectors_by_time = [model.single_exact_diagonalization(t) for t in times]

# Get the energy levels of eigenvectors.
exact_levels_by_time = np.array([levels_and_vectors[0]
                                 for levels_and_vectors in exact_levels_and_vectors_by_time])
# Get the eigenvectors.
exact_vectors_by_time = [levels_and_vectors[1]
                         for levels_and_vectors in exact_levels_and_vectors_by_time]
'''

# ----------------------------------------------------------------------------------------
# Chern Number Calculation


# ----------------------------------------------------------------------------------------
# Three-dimensional plot

plt.figure()


plt.subplot(2, 1, 1)
bulk_levels_ax2 = plt.axes(projection='3d')

for i in range(4):
    bulk_levels = np.array(model.bulk_levels(20)[:, :, i])
    xLine = np.linspace(-np.pi, np.pi, sites // 2 + 1)
    yLine = np.linspace(-np.pi, np.pi, sites // 2 + 1)
    X, Y = np.meshgrid(xLine, yLine)
    energy_levels = bulk_levels
    bulk_levels_ax2.plot_surface(X, Y, energy_levels)

bulk_levels_ax2.set_xlabel('kx')
bulk_levels_ax2.set_ylabel('ky')
bulk_levels_ax2.set_zlabel('E')
plt.show()

# ----------------------------------------------------------------------------------------

plt.subplot(2, 1, 2)
bulk_levels_ax1 = plt.axes(projection='3d')

for i in range(4):
    bulk_levels = np.array([model.bulk_levels(t)[1, :, i] for t in times])  # 默认kx， ky对称
    xLine = np.linspace(-np.pi, np.pi, sites // 2 + 1)
    X, Y = np.meshgrid(xLine, times / period)
    energy_levels = bulk_levels
    bulk_levels_ax1.plot_surface(X, Y, energy_levels)

bulk_levels_ax1.set_xlabel('ky')
bulk_levels_ax1.set_ylabel('Time/Period')
bulk_levels_ax1.set_zlabel('E')
plt.show()

# ----------------------------------------------------------------------------------------
# Dynamic Simulation
'''
init_state = np.zeros((sites, sites))
init_state[10][10] = 1

simulator = DynamicSimulation.Simulator(model, init_state)

simulator.generate_plot()
'''



