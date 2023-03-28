from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import FourSublatticePumpModel as Tm
from FourSublatticePumping import DynamicSimulation


# Initialize the Thouless Model
h_0 = 20
delta_0 = 0
t_0 = 1
w = 0.03
period = 2 * np.pi / w
sites = 36  # The number of sub-lattices
model = Tm.ThoulessModel4(h_0, delta_0, t_0, period, sites)

# Create the time array.
steps = 100
duration = period
times = np.linspace(0, duration, steps)

'''
Get the exact energy levels and vectors according to times.

Every time node in 'exact_levels_and_vectors_by_time' array is a tuple containing two sub-tuples: 
first tuple contains the energy levels, 
second tuple contains eigenvectors.
'''
exact_levels_and_vectors_by_time = [model.single_exact_diagonalization(t) for t in times]

# Get the energy levels of eigenvectors.
exact_levels_by_time = np.array([levels_and_vectors[0]
                                 for levels_and_vectors in exact_levels_and_vectors_by_time])
# Get the eigenvectors.
exact_vectors_by_time = [levels_and_vectors[1]
                         for levels_and_vectors in exact_levels_and_vectors_by_time]

# Get the middle eigenvector.
occupied_edge_vector_by_time = np.array([np.abs(exact_vectors[len(exact_vectors) // 2 - 1]) ** 2
                                         for exact_vectors in exact_vectors_by_time])

unoccupied_edge_vector_by_time = np.array([np.abs(exact_vectors[len(exact_vectors) // 2]) ** 2
                                           for exact_vectors in exact_vectors_by_time])

# Get the bulk energy levels by time.
# bulk_levels_by_time = np.array([model.bulk_levels(k_points, t) for t in times])

# Three-dimensional plot

bulk_levels_fig1 = plt.figure()
bulk_levels_ax1 = plt.axes(projection='3d')

for i in range(4):
    bulk_levels_by_time = np.array([model.bulk_levels(t)[:, i] for t in times])
    xLine = np.linspace(-np.pi, np.pi, sites // 4 + 1)
    X, Y = np.meshgrid(xLine, times/period)
    energy_levels = bulk_levels_by_time
    bulk_levels_ax1.plot_surface(X, Y, energy_levels)

bulk_levels_ax1.set_xlabel('k')
bulk_levels_ax1.set_ylabel('Time')
bulk_levels_ax1.set_zlabel('E')
plt.show()


# ----------------------------------------------------------------------------------------
# Dynamic Simulation
init_state = np.zeros((sites, 1))
init_state[19] = 1

simulator = DynamicSimulation.Simulator(model, init_state)

simulator.generate_plot()


