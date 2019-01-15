
import os
from _notebooks.notebook import Notebook
import py_wake


def test_notebooks():
    path = os.path.dirname(py_wake.__file__) + "/../_notebooks/elements/"
    for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]:
        nb = Notebook(path + f)
        nb.check_code()
        nb.check_links()
