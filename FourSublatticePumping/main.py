from math import *
import numpy as np
from matplotlib import colors
import matplotlib
import matplotlib.pyplot as plt
import FourSublatticePumpModel as Tm
from FourSublatticePumping import DynamicSimulation
import ChernNumberCalculator

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
ChernNumberCalculator.initialization(model)

# ----------------------------------------------------------------------------------------
# Three-dimensional plot
bulk_levels_fig1 = plt.figure()
bulk_levels_ax1 = plt.axes(projection='3d')

# Normalize the color
max_data = h_0
min_data = -h_0
norm = matplotlib.colors.Normalize(vmin=min_data, vmax=max_data)

for i in range(4):
    bulk_levels_by_time = np.array([model.bulk_levels(t)[:, i] for t in times])
    xLine = np.linspace(-np.pi, np.pi, sites // 4 + 1)
    X, Y = np.meshgrid(xLine, times / period)
    energy_levels = bulk_levels_by_time
    sc = bulk_levels_ax1.plot_surface(X, Y, energy_levels, norm=norm, rstride=1, cstride=1,
                                      cmap=plt.get_cmap('rainbow'), antialiased=True)
    # Create color bar
    if i == 0:
        bulk_levels_fig1.colorbar(sc, shrink=0.5)

# Limit z axe range
bulk_levels_ax1.set_zlim3d(min_data - 2, max_data + 2)

# Set the labels
bulk_levels_ax1.set_xlabel('k')
bulk_levels_ax1.set_ylabel('Time/Period')
bulk_levels_ax1.set_zlabel('E')

plt.show()

# ----------------------------------------------------------------------------------------
# Dynamic Simulation
init_state = np.zeros((sites, 1))
init_state[19] = 1

simulator = DynamicSimulation.Simulator(model, init_state)

simulator.generate_plot()
