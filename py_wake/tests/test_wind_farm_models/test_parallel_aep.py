import multiprocessing
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, wt_x, wt_y
from py_wake import IEA37SimpleBastankhahGaussian
from py_wake.tests.check_speed import timeit
import numpy as np
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines
from py_wake.examples.data import wtg_path


wt = WindTurbines.from_WAsP_wtg(wtg_path + "Vestas-V80.wtg")
site = Hornsrev1Site()
wf_model = IEA37SimpleBastankhahGaussian(site, wt)
wd_lst = np.arange(0, 360, 10)


def aep_wd(args):
    x, y, wd = args
    return wf_model(x, y, wd=wd, ws=None).aep().sum()


def aep_all_multiprocessing(pool, x, y):
    return np.sum(pool.map(aep_wd, [(x, y, i) for i in wd_lst]))


def aep_wfm_xy(args):
    wfm, x, y = args
    return wfm(x, y, wd=wd_lst).aep().sum()


def aep_xy(args):
    x, y = args
    return wf_model(x, y, wd=wd_lst).aep().sum()


def test_multiprocessing_wd():
    pool = multiprocessing.Pool(4)
    aep1, t_lst1 = timeit(aep_wd, min_runs=1)((wt_x, wt_y, wd_lst))
    aep2, t_lst2 = timeit(aep_all_multiprocessing, min_runs=1)(pool, wt_x, wt_y)
    t1, t2 = np.mean(t_lst1), np.mean(t_lst2)
    print("1 CPU: %.2fs, %d CPUs: %.2fs, speedup: %d%%" % (t1, pool._processes, t2, (t1 - t2) / t1 * 100))
    npt.assert_almost_equal(aep1, aep2 / 36)


def test_multiprocessing_wfm_xy():
    pool = multiprocessing.Pool(2)
    arg_lst = [(wf_model, np.array(wt_x) + i, wt_y) for i in range(4)]
    aep1, t_lst1 = timeit(lambda arg_lst: [aep_wfm_xy(arg) for arg in arg_lst], min_time=0, min_runs=1)(arg_lst)
    aep2, t_lst2 = timeit(lambda arg_lst: pool.map(aep_wfm_xy, arg_lst), min_time=0, min_runs=1)(arg_lst)
    t1, t2 = np.mean(t_lst1), np.mean(t_lst2)
    print("1 CPU: %.2fs, %d CPUs: %.2fs, speedup: %d%%" % (t1, pool._processes, t2, (t1 - t2) / t1 * 100))
    npt.assert_almost_equal(aep1, aep2)


def test_multiprocessing_xy():
    pool = multiprocessing.Pool(2)
    arg_lst = [(np.array(wt_x) + i, wt_y) for i in range(4)]
    aep1, t_lst1 = timeit(lambda arg_lst: [aep_xy(arg) for arg in arg_lst], min_time=0, min_runs=1)(arg_lst)
    aep2, t_lst2 = timeit(lambda arg_lst: pool.map(aep_xy, arg_lst), min_time=0, min_runs=1)(arg_lst)
    t1, t2 = np.mean(t_lst1), np.mean(t_lst2)
    print("1 CPU: %.2fs, %d CPUs: %.2fs, speedup: %d%%" % (t1, pool._processes, t2, (t1 - t2) / t1 * 100))
    npt.assert_almost_equal(aep1, aep2)
