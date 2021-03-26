from datetime import datetime
import functools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from py_wake import NOJ
from py_wake.deficit_models import fuga
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.examples.data.hornsrev1 import wt_x, wt_y, HornsrevV80, Hornsrev1Site
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from pandas.plotting import register_matplotlib_converters
import sys
register_matplotlib_converters()


def timeit(func, min_time=0, min_runs=1, verbose=False, line_profile=False, profile_funcs=[]):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        if line_profile and getattr(sys, 'gettrace')() is None:
            from line_profiler import LineProfiler
            lp = LineProfiler()
            lp.timer_unit = 1e-6
            for f in profile_funcs:
                lp.add_function(f)
            lp_wrapper = lp(func)
            t = time.time()
            res = lp_wrapper(*args, **kwargs)
            t = time.time() - t
            if verbose:
                lp.print_stats()
            return res, [t]
        else:
            t_lst = []
            for i in range(100000):
                startTime = time.time()
                res = func(*args, **kwargs)
                t_lst.append(time.time() - startTime)
                if sum(t_lst) > min_time and len(t_lst) >= min_runs:
                    if hasattr(func, '__name__'):
                        fn = func.__name__
                    else:
                        fn = "Function"
                    if verbose:
                        print('%s: %f +/-%f (%d runs)' % (fn, np.mean(t_lst), np.std(t_lst), i + 1))
                    return res, t_lst
    return newfunc


def check_speed_Hornsrev(WFModel):
    assert getattr(sys, 'gettrace')() is None, "Skipping speed check, In debug mode!!!"
    wt = HornsrevV80()
    site = Hornsrev1Site()
    wf_model = WFModel(site, wt)
    aep, t_lst = timeit(lambda x, y: wf_model(x, y).aep().sum(), min_runs=3)(wt_x, wt_y)

    fn = tfp + "speed_check/%s.txt" % WFModel.__name__
    if os.path.isfile(fn):
        with open(fn) as fid:
            lines = fid.readlines()

        # check aep
        npt.assert_almost_equal(float(lines[-1].split(";")[1]), aep)

        timings = np.array([(np.mean(eval(l.split(";")[2])), np.std(eval(l.split(";")[2]))) for l in lines])
        dates = [np.datetime64(l.split(";")[0]) for l in lines]
        dates = np.r_[dates, datetime.now()]
        y = np.r_[timings[:, 0], np.mean(t_lst)]

        error = np.r_[timings[:, 1], np.std(t_lst)]
        fig, axes = plt.subplots(2, 1)
        fig.suptitle(WFModel.__name__)
        for x, ax in zip([dates, np.arange(len(dates))], axes):
            ax.fill_between(x, y - 2 * error, y + 2 * error)
            ax.plot(x, y, '.-k')
            ax.axhline(y[:-1].mean() + 2 * error[:-1].mean(), ls='--', color='gray')

        if y[-1] > (y[:-1].mean() + 2 * error[:-1].mean()):
            raise Exception("Simulation time too slow, %f > %f" % (y[-1], (y[:-1].mean() + 2 * error[:-1].mean())))

    if getattr(sys, 'gettrace')() is None:
        with open(fn, 'a') as fid:
            fid.write("%s;%.10f;%s\n" % (datetime.now(), aep, t_lst))


def test_check_speed():
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'

    def Fuga(site, wt):
        return fuga.Fuga(path, site, wt)

    for WFModel in [NOJ, IEA37SimpleBastankhahGaussian, Fuga]:
        try:
            check_speed_Hornsrev(WFModel)
        except Exception as e:
            print(e)
            raise e
    if 1:
        plt.show()


if __name__ == '__main__':
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'

    def Fuga(site, wt):
        return fuga.Fuga(path, site, wt)

    for WFModel in [NOJ, IEA37SimpleBastankhahGaussian, Fuga]:
        try:
            check_speed_Hornsrev(WFModel)
        except Exception as e:
            print(e)
            raise e

    plt.show()
