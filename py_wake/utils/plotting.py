import matplotlib.pyplot as plt


def setup_plot(ax=None, title=None, ylabel=None, xlabel=None, xlim=None, ylim=None, grid=True, axis=None, figsize=None):
    ax = ax or plt.gca()
    if figsize:
        ax.figure.set_size_inches(figsize)
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
    if axis:
        ax.axis(axis)
    plt.tight_layout()
