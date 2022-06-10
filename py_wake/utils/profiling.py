import time
import functools
import sys
from py_wake import np
import gc
import os
import psutil
import memory_profiler
import ctypes
from pathlib import Path
import linecache


def timeit(func, min_time=0, min_runs=1, verbose=False, line_profile=False, profile_funcs=[]):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        if line_profile and getattr(sys, 'gettrace')() is None:  # pragma: no cover
            lp_wrapper = line_timeit(func, profile_funcs)
            t = time.time()
            res, lp = lp_wrapper(*args, **kwargs)
            t = time.time() - t
            if verbose:
                lp.print_stats()
            return res, [t]
        else:
            t_lst = []
            time_start = time.time()
            for i in range(100000):
                t0 = time.time_ns()
                res = func(*args, **kwargs)
                t_lst.append((time.time_ns() - t0) * 1e-9)
                if (time.time() - time_start) > min_time and len(t_lst) >= min_runs:
                    break

            if verbose:  # pragma: no cover
                if hasattr(func, '__name__'):
                    fn = func.__name__
                else:
                    fn = "Function"
                print('%s: %f +/-%f (%d runs)' % (fn, np.mean(t_lst), np.std(t_lst), i + 1))
            return res, t_lst
    return newfunc


def line_timeit(func, profile_funcs=[]):  # pragma: no cover
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp.timer_unit = 1e-6

    for f in profile_funcs:
        lp.add_function(f)
    if getattr(sys, 'gettrace')() is None:
        lp_wrapper = lp(func)
    else:
        # in debug mode
        lp_wrapper = func
    return lambda *args, lp=lp, **kwargs: (lp_wrapper(*args, **kwargs), lp)


def compare_lineprofile(lp1, lp2, include_gt_pct=None):  # pragma: no cover
    stats1, stats2 = lp1.get_stats(), lp2.get_stats()
    for ((fn1, lineno1, name1), timings1), ((fn2, lineno2, name2),
                                            timings2) in zip(sorted(stats1.timings.items()), sorted(stats2.timings.items())):
        assert fn1 == fn2
        assert lineno1 == lineno2
        assert name1 == name2

        if os.path.exists(fn1):
            # Clear the cache to ensure that we get up-to-date results.
            linecache.clearcache()
        lines = [l.rstrip() for l in linecache.getlines(fn1)]
        template = '%6s %9s %12s %12s %8s %8s %8s  %-s'

        total1 = np.sum([v[2] for v in timings1])
        total2 = np.sum([v[2] for v in timings2])
        print('Total time: %g s' % (total1 * stats1.unit))
        print(f'File: {fn1}:{lineno1}')
        print(f'Function: {name1} at line {lineno1}\n')
        header = template % ('Line #', 'Hits', 'Time A', 'Time B', '% time', '% diff', '% diff', 'Line Contents')
        print(header)
        print('=' * len(header))

        print(template % (lineno1, "", "", "", "", "", "", lines[lineno1 - 1]))
        for (lineno1, hits1, time1), (lineno2, hits2, time2) in zip(timings1, timings2):
            pct_time = (time1 / total1 * 100)
            if include_gt_pct is None or pct_time > include_gt_pct:
                print(template % (lineno1, hits1, time1 / 1000, time2 / 1000,
                                  '%5.1f' % pct_time,
                                  '%5.1f' % ((time2 - time1) / time1 * 100),
                                  '%5.1f' % ((time2 - time1) / total1 * 100),
                                  lines[lineno1 - 1]))
        print(template % ("", "", "--------", "--------", "", "", "----", ""))
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'divide by zero encountered in double_scalars')
            print(
                template %
                ("", "", total1 / 1e6, total2 / 1e6, "", "", "%5.1f" %
                 ((total2 - total1) / total1 * 100), ""))


def get_memory_usage():
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
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


def profileit(f):
    def wrap(*args, **kwargs):
        (res, t), mem_usage = check_memory_usage(timeit(f), subtract_initial=True)(*args, **kwargs)
        return res, t[0], mem_usage
    return wrap
