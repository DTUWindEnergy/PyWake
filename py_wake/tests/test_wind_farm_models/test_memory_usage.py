from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site, wt16_x, wt16_y
from py_wake.deficit_models.noj import NOJ, NOJDeficit
import os
import psutil
import matplotlib.pyplot as plt
import numpy as np
import gc
from py_wake.wind_farm_models.engineering_models import All2AllIterative
import memory_profiler
from py_wake.tests import npt
import pytest


def get_memory_usage():
    pid = os.getpid()
    python_process = psutil.Process(pid)
    return python_process.memory_info()[0] / 1024**2


def test_memory_usage():
    if os.name == 'posix':
        pytest.skip('Memory usage seems not to work on linux')
    gc.collect()
    initial_mem_usage = get_memory_usage()
    wt = V80()
    site = Hornsrev1Site()
    x, y = site.initial_position.T

    for wfm, mem_min, mem_max in [(NOJ(site, wt), 65, 90),
                                  (All2AllIterative(site, wt, wake_deficitModel=NOJDeficit()), 530, 600)]:
        mem_usage, _ = memory_profiler.memory_usage(
            (wfm, (x, y), {'wd': np.arange(0, 360, 4)}), interval=0.001, max_usage=True, retval=True)

        mem_usage -= initial_mem_usage
        print(initial_mem_usage, mem_usage)
        assert mem_min < mem_usage < mem_max, (initial_mem_usage, mem_usage)

    return


def test_memory_leak():

    N = 10

    wt = V80()
    site = Hornsrev1Site()

    wfm_lst = [NOJ(site, wt), All2AllIterative(site, wt, wake_deficitModel=NOJDeficit())]
    memory_usage = np.zeros((len(wfm_lst), N))
    for i, wfm in enumerate(wfm_lst):
        memory_usage[i, 0] = get_memory_usage()
        for j in range(1, N):
            wfm(wt16_x, wt16_y, ws=10, wd=np.arange(0, 360, 30))
            gc.collect()
            memory_usage[i, j] = get_memory_usage()
    npt.assert_array_less(memory_usage - memory_usage[:, :1], 1)  # at most 1mb more than initial usage
    if 0:
        for i, wfm in enumerate(wfm_lst):
            plt.plot(memory_usage[i], label=str(wfm.__class__.__name__))
        plt.legend()
        plt.show()
