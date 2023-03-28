import math
from matplotlib import colors
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ThoulessModel as Tm
import DynamicSimulation
import ChernNumberCalculator

# Initialize the Thouless Model
h_0 = 20
delta_0 = 1
t_0 = 1
w = 0.08
period = 2 * np.pi / w  # 2 * np.pi
sites = 18
model = Tm.ThoulessModel(h_0, delta_0, t_0, period, sites)

# Create the time array.
steps = 400
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

# Get the bulk energy levels by time.
bulk_levels_by_time = np.array([model.bulk_levels(t) for t in times])

# ----------------------------------------------------------------------------------------
# Chern Number Calculation
ChernNumberCalculator.initialization(model)

# ----------------------------------------------------------------------------------------
# Three-dimensional plot

bulk_levels_fig1 = plt.figure()
bulk_levels_ax1 = Axes3D(bulk_levels_fig1)
xLine = np.linspace(-np.pi, np.pi, sites // 2 + 1)
X, Y = np.meshgrid(xLine, times / period)

# Normalize the color
min_data = np.min(bulk_levels_by_time)
max_data = np.max(bulk_levels_by_time)
norm = matplotlib.colors.Normalize(vmin=min_data, vmax=max_data)

# Fill the datas
energy_levels_1 = bulk_levels_by_time[:, 0:sites // 2 + 1]
bulk_levels_ax1.plot_surface(X, Y, energy_levels_1, norm=norm, rstride=1, cstride=1,
                             cmap=plt.get_cmap('rainbow'), antialiased=True)
energy_levels_2 = bulk_levels_by_time[:, sites // 2 + 1:]
sc = bulk_levels_ax1.plot_surface(X, Y, energy_levels_2, norm=norm, rstride=1, cstride=1,
                                  cmap=plt.get_cmap('rainbow'), antialiased=True)

# Draw contourf
# bulk_levels_ax1.contourf(X, Y, energy_levels_1, norm=norm, offset=max_data+2, cmap='rainbow')
bulk_levels_ax1.contourf(X, Y, energy_levels_2, norm=norm, offset=min_data - 4, cmap='rainbow')
# Create color bar
bulk_levels_fig1.colorbar(sc, shrink=0.5)
# Limit z axe range
bulk_levels_ax1.set_zlim3d(min_data - 2, max_data + 2)

# Set the labels
bulk_levels_ax1.set_xlabel('k')
bulk_levels_ax1.set_ylabel('t/Tp')
bulk_levels_ax1.set_zlabel('E/J')

plt.show()

# ----------------------------------------------------------------------------------------
# Dynamic Simulation
init_state = np.zeros((sites, 1))
init_state[6] = 1

simulator = DynamicSimulation.Simulator(model, init_state)

simulator.generate_plot()
