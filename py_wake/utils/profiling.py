import time
import functools
import sys
import numpy as np
import gc
import os
import psutil
import memory_profiler


def timeit(func, min_time=0, min_runs=1, verbose=False, line_profile=False, profile_funcs=[]):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):  # pragma: no cover
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


def get_memory_usage():
    gc.collect()
    pid = os.getpid()
    python_process = psutil.Process(pid)
    return python_process.memory_info()[0] / 1024**2


def check_memory_usage(f, subtract_initial=True):
    def wrap(*args, **kwargs):
        initial_mem_usage = get_memory_usage()
        mem_usage, res = memory_profiler.memory_usage((f, args, kwargs), interval=.02, max_usage=True, retval=True)
        if subtract_initial:
            mem_usage -= initial_mem_usage
        return res, mem_usage
    return wrap
