import multiprocessing
from py_wake.examples.data.hornsrev1 import HornsrevV80, Hornsrev1Site, wt_x, wt_y
from py_wake import IEA37SimpleBastankhahGaussian
from py_wake.tests.check_speed import timeit
import numpy as np
from py_wake.tests import npt


wt = HornsrevV80()
site = Hornsrev1Site()
wf_model = IEA37SimpleBastankhahGaussian(site, wt)
wd_lst = np.arange(0, 360)


def aep_wd(args):
    x, y, wd = args
    return wf_model(x, y, wd=wd, ws=None).aep()


def aep_all_multiprocessing(pool, x, y):
    return np.sum(pool.map(aep_wd, [(x, y, i) for i in wd_lst]))


def test_multiprocessing():
    pool = multiprocessing.Pool()
    aep1, t_lst1 = timeit(aep_wd, min_runs=1)((wt_x, wt_y, wd_lst))
    aep2, t_lst2 = timeit(aep_all_multiprocessing, min_runs=1)(pool, wt_x, wt_y)
    t1, t2 = np.mean(t_lst1), np.mean(t_lst2)
    print("1 CPU: %.2fs, %d CPUs: %.2fs, speedup: %d%%" % (t1, pool._processes, t2, (t1 - t2) / t1 * 100))
    npt.assert_almost_equal(aep1, aep2)
