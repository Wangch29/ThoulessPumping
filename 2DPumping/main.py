from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import TwoDModel as Tm
import ChernNumberCalculator
import DynamicSimulation
from matplotlib import colors

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
steps = 1000
duration = period
times = np.linspace(0, duration, steps)
'''
# ----------------------------------------------------------------------------------------
# Chern Number Calculation


# ----------------------------------------------------------------------------------------
# Three-dimensional plot
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 11,
             }


fig_1 = plt.figure()

max_data = h_0
min_data = -h_0
norm = matplotlib.colors.Normalize(vmin=min_data, vmax=max_data)

bulk_levels_ax2 = plt.axes(projection='3d')

for i in range(4):
    bulk_levels = np.array(model.bulk_levels(20)[:, :, i])
    xLine = np.linspace(-np.pi, np.pi, sites // 2 + 1)  #
    yLine = np.linspace(-np.pi, np.pi, sites // 2 + 1)
    X, Y = np.meshgrid(xLine, yLine)
    energy_levels = bulk_levels
    sc = bulk_levels_ax2.plot_surface(X, Y, energy_levels,  norm=norm, rstride=1, cstride=1,
                                      cmap=plt.get_cmap('coolwarm'), antialiased=True)

    if i == 0:
        fig_1.colorbar(sc, shrink=0.5)

# Set x axis
bulk_levels_ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

bulk_levels_ax2.set_xlabel('kx')
bulk_levels_ax2.set_ylabel('ky')
bulk_levels_ax2.set_zlabel('E', labelpad=-15)
plt.show()

# ----------------------------------------------------------------------------------------

fig_2 = plt.figure()
max_data = h_0
min_data = -h_0
bulk_levels_ax1 = plt.axes(projection='3d')

norm = matplotlib.colors.Normalize(vmin=min_data, vmax=max_data)

for i in range(4):
    bulk_levels = np.array([model.bulk_levels(t)[1, :, i] for t in times])  # 默认kx， ky对称
    xLine = np.linspace(-np.pi, np.pi, sites // 2 + 1)
    X, Y = np.meshgrid(xLine, times / period)
    energy_levels = bulk_levels
    sc = bulk_levels_ax1.plot_surface(X, Y, energy_levels,  norm=norm, rstride=1, cstride=1,
                                      cmap=plt.get_cmap('coolwarm'), antialiased=True)

    if i == 0:
        fig_2.colorbar(sc, shrink=0.5)

# Set x axis
bulk_levels_ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

bulk_levels_ax1.set_xlabel('ky')
bulk_levels_ax1.set_ylabel('t/Tp')
bulk_levels_ax1.set_zlabel('E', labelpad=-15)
plt.show()
'''
# ----------------------------------------------------------------------------------------
# Dynamic Simulation

init_state = np.zeros((sites, sites))
init_state[10][10] = 1

simulator = DynamicSimulation.Simulator(model, init_state)

simulator.generate_plot()




