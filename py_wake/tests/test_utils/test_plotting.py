import matplotlib.pyplot as plt
from py_wake.utils.plotting import setup_plot


def test_setup_plot():
    plt.plot([0, 1, 2], [0, 1, 0], label='test')
    setup_plot(title='Test', ylabel="ylabel", xlabel='xlabel', xlim=[0, 5], ylim=[0, 2])
    if 0:
        plt.show()
