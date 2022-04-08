import matplotlib.pyplot as plt


def setup_plot(ax=None, title=None, ylabel=None, xlabel=None, xlim=None, ylim=None, grid=True):
    ax = ax or plt.gca()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid()
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
