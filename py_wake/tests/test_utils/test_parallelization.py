import numpy as np
from memory_profiler import _get_memory
from py_wake.utils.profiling import get_memory_usage
from py_wake.utils.parallelization import gc_func


def test_gc_function():
    def f():
        np.full((1, 1024**2, 128), 1.)  # allocate 1gb

    mem_before = get_memory_usage()
    gc_func(f)()

    # assert memory increase is less than 5mb (on linux an increase occurs)
    assert get_memory_usage() - mem_before < 5
