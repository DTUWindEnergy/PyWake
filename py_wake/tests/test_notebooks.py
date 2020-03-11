
import os

import py_wake
import pytest
from py_wake.tests.notebook import Notebook


def get_notebooks():
    path = os.path.dirname(py_wake.__file__) + "/../docs/notebooks/"
    return [Notebook(path + f) for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]]


@pytest.mark.parametrize("notebook", get_notebooks())
def test_notebooks(notebook):
    import matplotlib.pyplot as plt

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    try:
        notebook.check_code()
        notebook.check_links()
        pass
    except Exception as e:
        raise Exception(notebook.filename + " failed") from e
