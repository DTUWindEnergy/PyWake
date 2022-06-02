import time
from py_wake import np
from py_wake.utils.profiling import profileit, line_timeit, compare_lineprofile, timeit
from py_wake.tests import npt
from py_wake.utils.numpy_utils import Numpy32


def test_profile_it():
    def f(t, mem):
        start = time.time()
        a = np.arange(1024 * 1024 * mem, dtype=np.byte)
        while time.time() < t + start:
            pass

    for i in range(10):
        try:
            # check 3 successive runs (all should take .1 s and 10MB
            for _ in range(3):
                r, t, m = profileit(f)(.1, 100)
                npt.assert_allclose(t, 0.1, rtol=.2)
                npt.assert_allclose(m, 100, rtol=.2)
            return
        except AssertionError:
            # fail test if not succeeding in 10 attempts
            if i == 9:
                raise


def test_line_profile():
    def f():
        time.sleep(0.09)
        time.sleep(0.01)
    timeit(f, line_profile=1, verbose=1)()


def test_compare_lineprofile():
    def f2(x):
        return np.cos(x)

    def f(a):
        for _ in range(10):
            b = np.sin(a)
            c = f2(b)
        return np.tan(c)

    N = 1000000
    profile_funcs = [f2]
    res, lp64 = line_timeit(f, profile_funcs)(np.full(N, 1))

    with Numpy32():
        res, lp32 = line_timeit(f, profile_funcs)(np.full(N, 1))

    compare_lineprofile(lp64, lp32)
