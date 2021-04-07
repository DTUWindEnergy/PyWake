import importlib
import os
import pkgutil
import warnings

import pytest

import sys
import py_wake
from unittest import mock
from py_wake.flow_map import Grid


def get_main_modules():
    package = py_wake
    modules = []
    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = importlib.import_module(modname)
            except Exception:
                print(f"!!!!!! failed to import {modname}")
                raise

        if 'main' in dir(m):
            modules.append(m)
    return modules


def print_main_modules():
    print("\n".join([m.__file__ for m in get_main_modules()]))


@pytest.mark.parametrize("module", get_main_modules())
def test_main(module):
    # check that all main module examples run without errors
    if os.name == 'posix' and "DISPLAY" not in os.environ:
        pytest.xfail("No display")

    import matplotlib.pyplot as plt

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    def no_print(*_):
        pass
    default_resolution = Grid.default_resolution
    Grid.default_resolution = 100
    try:
        with mock.patch.object(module, "print", no_print):  # @UndefinedVariable
            # To count 'if __name__=="__main__": main()' in cov
            with mock.patch.object(module, "__name__", "__main__"):  # @UndefinedVariable
                getattr(module, 'main')()

    except Exception as e:
        raise type(e)(str(e) +
                      ' in %s.main' % module.__name__).with_traceback(sys.exc_info()[2])
    finally:
        Grid.default_resolution = default_resolution
        plt.close('all')


if __name__ == '__main__':
    print_main_modules()
