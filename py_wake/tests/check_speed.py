from datetime import datetime
import functools
import os
import time

import matplotlib.pyplot as plt
from py_wake import np
from py_wake import NOJ
from py_wake.deficit_models import fuga
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.examples.data.hornsrev1 import wt_x, wt_y, HornsrevV80, Hornsrev1Site
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
import sys
from py_wake.utils.profiling import timeit

path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'


def Fuga(site, wt):
    return fuga.Fuga(path, site, wt)


test_lst = [(NOJ, 1.2), (IEA37SimpleBastankhahGaussian, 1.5), (Fuga, 1)]


def check_speed_Hornsrev(WFModel, max_min):
    assert getattr(sys, 'gettrace')() is None, "Skipping speed check, In debug mode!!!"
    wt = HornsrevV80()
    site = Hornsrev1Site()
    wf_model = WFModel(site, wt)
    aep, t_lst = timeit(lambda x, y: wf_model(x, y).aep().sum(), min_runs=3)(wt_x, wt_y)
    assert min(t_lst) < max_min, f'{WFModel},{t_lst}'
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

    for WFModel, max_min in test_lst:
        try:
            check_speed_Hornsrev(WFModel, max_min)
        except Exception as e:
            print(WFModel)
            print(e)
            raise e


if __name__ == '__main__':
    test_check_speed()
    plt.show()
