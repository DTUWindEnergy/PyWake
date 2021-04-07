import os

import pytest

from py_wake.tests.notebook import Notebook
import py_wake
from py_wake.flow_map import Grid


def get_notebooks():
    def get(path):
        return [Notebook(path + f) for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]]
    path = os.path.dirname(py_wake.__file__) + "/../docs/notebooks/"
    return get(path) + get(path + "exercises/")


notebooks = get_notebooks()


@pytest.mark.parametrize("notebook", notebooks, ids=[os.path.basename(nb.filename) for nb in notebooks])
def test_notebooks(notebook):
    import matplotlib.pyplot as plt

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    try:
        default_resolution = Grid.default_resolution
        Grid.default_resolution = 100
        plt.rcParams.update({'figure.max_open_warning': 0})
        notebook.check_code()
        notebook.check_links()
        notebook.remove_empty_end_cell()
        notebook.check_pip_header()
        pass
    except Exception as e:
        raise Exception(notebook.filename + " failed") from e
    finally:
        Grid.default_resolution = default_resolution
        plt.close('all')
        plt.rcParams.update({'figure.max_open_warning': 20})


if __name__ == '__main__':
    print("\n".join([f.filename for f in get_notebooks()]))
