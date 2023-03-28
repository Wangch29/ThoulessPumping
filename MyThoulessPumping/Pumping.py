import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython import display
from matplotlib import animation


class Pump:
    sites: int
    steps: int
    duration: int
    occupied_edge_vector_by_time: np.ndarray
    unoccupied_edge_vector_by_time: np.ndarray

    def __init__(self, sites, steps, duration, occupied_edge_vector_by_time, unoccupied_edge_vector_by_time):
        self.sites = sites
        self.steps = steps
        self.duration = duration
        self.occupied_edge_vector_by_time = occupied_edge_vector_by_time
        self.unoccupied_edge_vector_by_time = unoccupied_edge_vector_by_time

    def plot_density(self, time_index):
        density_fig, density_ax = plt.subplots()
        density_ax.set_title(f'Edge states at time {time_index / self.steps * self.duration: .3},'
                             f' 'f'period {self.duration}.')
        density_ax.set_xlabel('Relative position')
        density_ax.set_ylabel('Density summed on two sites')
        occupied_x, occupied_y = np.linspace(0, 1, self.sites)[::2], np.sum(
            self.occupied_edge_vector_by_time[time_index].reshape((-1, 2)), axis=1)
        unoccupied_x, unoccupied_y = np.linspace(0, 1, self.sites)[::2], np.sum(
            self.unoccupied_edge_vector_by_time[time_index].reshape((-1, 2)), axis=1)
        density_ax.fill_between(occupied_x, occupied_y, label='Occupied')
        density_ax.plot(occupied_x, occupied_y, '--')
        density_ax.fill_between(unoccupied_x, unoccupied_y, alpha=0.2, label='Unoccupied')
        density_ax.plot(unoccupied_x, unoccupied_y, '--', alpha=0.2)
        density_ax.legend()
        plt.show()

    def animate_density(self, animation_ax, animation_plot_occupied, animation_plot_unoccupied,
                        frame, *fargs):
        time_index = frame
        animation_ax.set_ylim([0, 1])
        animation_ax.set_title(f'Edge states at time {time_index / self.steps * self.duration: .3}, period {self.duration}.')
        animation_ax.set_xlabel('Relative position')
        animation_ax.set_ylabel('Density summed on two sites')

        occupied_x, occupied_y = np.linspace(0, 1, self.sites)[::2], np.sum(
            self.occupied_edge_vector_by_time[time_index].reshape((-1, 2)), axis=1)
        unoccupied_x, unoccupied_y = np.linspace(0, 1, self.sites)[::2], np.sum(
            self.unoccupied_edge_vector_by_time[time_index].reshape((-1, 2)), axis=1)

        animation_ax.collections.clear()

        animation_plot_occupied.set_xdata(occupied_x)
        animation_plot_occupied.set_ydata(occupied_y)

        animation_fill_occupied = animation_ax.fill_between(occupied_x, occupied_y, label='Occupied', color='blue')

        animation_plot_unoccupied.set_xdata(unoccupied_x)
        animation_plot_unoccupied.set_ydata(unoccupied_y)

        animation_fill_unoccupied = animation_ax.fill_between(unoccupied_x, unoccupied_y, alpha=0.2, label='Unoccupied',
                                                              color='orange')

        animation_ax.legend()

        return animation_fill_occupied, animation_plot_occupied, animation_fill_unoccupied, animation_plot_unoccupied

    def animate_density_init(self, animation_ax, animation_plot_occupied, animation_plot_unoccupied):
        return self.animate_density(animation_ax, animation_plot_occupied, animation_plot_unoccupied, 0)

    def play(self):
        interact(self.plot_density, time_index=widgets.IntSlider(
            value=0,
            min=0,
            max=self.steps,
            step=1)
                 )

        animation_fig, animation_ax = plt.subplots()

        animation_plot_occupied = animation_ax.plot([], [], '--')
        animation_plot_unoccupied = animation_ax.plot([], [], '--', alpha=0.2)

        density_animation = animation.FuncAnimation(animation_fig, self.animate_density,
                                                    np.arange(self.steps),
                                                    init_func=self.animate_density_init(animation_ax, animation_plot_occupied, animation_plot_unoccupied),
                                                    interval=25, blit=True)
        density_video = density_animation.to_html5_video()
        display.display(display.HTML(density_video))

