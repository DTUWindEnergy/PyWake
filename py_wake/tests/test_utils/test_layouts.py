import matplotlib.pyplot as plt
from py_wake.utils.layouts import rectangle, square
from py_wake.examples.data.hornsrev1 import V80


def test_square():
    wt = V80()
    x, y = square(25, wt.diameter() * 5)
    if 0:
        wt.plot(x, y)
        plt.show()
    assert len(x) == 25
    assert x[-1] == y[-1] == 80 * 5 * (5 - 1)


def test_rectangle():
    wt = V80()
    x, y = rectangle(8, 5, wt.diameter() * 5)
    if 0:
        wt.plot(x, y)
        plt.show()
    assert len(x) == 8
    assert x[-1] == 800
    assert y[-1] == 400
