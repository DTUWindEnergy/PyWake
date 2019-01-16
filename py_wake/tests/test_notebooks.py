
import os
from _notebooks.notebook import Notebook
import py_wake
import pytest


def get_notebooks():
    path = os.path.dirname(py_wake.__file__) + "/../_notebooks/elements/"
    return [Notebook(path + f) for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]]


@pytest.mark.parametrize("notebook", get_notebooks())
def test_notebooks(notebook):
    import matplotlib.pyplot as plt

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    notebook.check_code()
    notebook.check_links()
