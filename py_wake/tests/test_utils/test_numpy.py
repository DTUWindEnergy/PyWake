from py_wake import np
from py_wake.utils.numpy_utils import Numpy32
import pytest
from py_wake.tests import npt
from py_wake.utils.profiling import profileit, timeit
from py_wake.utils.gradients import autograd
from autograd.numpy.numpy_boxes import ArrayBox
from numpy import newaxis as na
import os


@pytest.mark.parametrize('v,dtype,dtype32', [(5., float, np.float32),
                                             (5 + 0j, complex, np.complex64)])
def test_32bit_precision(v, dtype, dtype32):

    assert np.array(v).dtype == dtype
    with Numpy32():
        assert np.array(v).dtype == dtype32
    assert np.array(v).dtype == dtype


def test_members():
    import numpy
    with Numpy32():
        assert np.pi == numpy.pi
        npt.assert_array_equal(np.r_[3, 4], numpy.r_[3, 4])
        np.array([3]).dtype
        np.iscomplexobj([3])


def test_autograd32():
    def f(x):
        assert isinstance(x, ArrayBox)
        assert x.dtype == np.float32
        return x**2

    with Numpy32():
        autograd(f)([1, 2])


def test_speed_mem():
    if os.name == 'posix':
        pytest.xfail("Memory tests behave differently on Linux")

    def f(x):
        return (x**2).sum()

    N = 1000
    x = np.arange(N, dtype=float) * np.arange(1024**2 / 8)[:, na] + 1
    t64, mem64 = profileit(f)(x)[1:]

    with Numpy32():
        x = np.asarray(x)
        t32, mem32 = profileit(f)(x)[1:]

    assert t32 / t64 < .6
    assert mem32 / mem64 < .6

# def test_speed():
#     site = Hornsrev1Site()
#     windTurbines = V80()
#     wfm = BastankhahGaussian(site, windTurbines)
#     x, y = square(200, 5 * 80)
#     timeit(wfm.__call__, verbose=1)(x, y)
#     with Numpy32():
#         timeit(wfm.__call__, verbose=1)(x, y)

# def test_speed_gradients():
#     site = Hornsrev1Site()
#     windTurbines = V80()
#     wfm = BastankhahGaussian(site, windTurbines)
#     x, y = square(200, 5 * 80)
#     with Numpy32():
#         timeit(wfm.aep_gradients, verbose=1)(autograd, wrt_arg=['x', 'y'], wd_chunks=4, x=x, y=y)
#     timeit(wfm.aep_gradients, verbose=1)(autograd, wrt_arg=['x', 'y'], wd_chunks=4, x=x, y=y)
