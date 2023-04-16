from math import *
import numpy as np
from matplotlib import colors
import matplotlib
import matplotlib.pyplot as plt
import FourSublatticePumpModel as Tm
from FourSublatticePumping import DynamicSimulation
import ChernNumberCalculator


def format_func(value, tick_number):
    # N是pi/2的倍数
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"  # 0点
    elif N == 2:
        return r"$\pi$"  # pi
    elif N == -2:
        return r"$-\pi$"  # -pi


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

# ----------------------------------------------------------------------------------------
# Chern Number Calculation
ChernNumberCalculator.initialization(model)

# ----------------------------------------------------------------------------------------
# Three-dimensional plot
'''
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 11,
             }

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
                                      cmap=plt.get_cmap('coolwarm'), antialiased=True)
    # Create color bar
    if i == 0:
        bulk_levels_fig1.colorbar(sc, shrink=0.5)

# Limit z axe range
bulk_levels_ax1.set_zlim3d(min_data - 2, max_data + 2)

# Set x axis
bulk_levels_ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Set the labels
bulk_levels_ax1.set_xlabel('k', font2)
bulk_levels_ax1.set_ylabel('t/Tp', font2)
bulk_levels_ax1.set_zlabel('E', font2, labelpad=-15)

plt.style.use('ggplot')

plt.show()
'''
# ----------------------------------------------------------------------------------------
# Dynamic Simulation

init_state = np.zeros((sites, 1))
init_state[19] = 1

simulator = DynamicSimulation.Simulator(model, init_state)

simulator.generate_plot()
