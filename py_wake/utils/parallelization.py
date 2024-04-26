import multiprocessing
import atexit
import platform
import gc
import os

pool_dict = {}


def get_pool(processes=multiprocessing.cpu_count()):
    if processes not in pool_dict:
        # close pools
        for pool in pool_dict.values():
            pool.close()
        pool_dict.clear()

        if platform.system() == 'Darwin':  # pragma: no cover
            pool_dict[processes] = multiprocessing.get_context('fork').Pool(processes)
        else:
            pool_dict[processes] = multiprocessing.Pool(processes)
    return pool_dict[processes]


class gc_func():
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        r = self.func(*args, **kwargs)
        gc.collect()
        return r


def get_pool_map(processes=multiprocessing.cpu_count()):
    pool = get_pool(processes)

    def gc_map(func, iterable, chunksize=None):
        return pool.map(gc_func(func), iterable, chunksize)
    return gc_map


def get_pool_starmap(processes=multiprocessing.cpu_count()):
    pool = get_pool(processes)

    def gc_map(func, iterable, chunksize=None):
        return pool.starmap(gc_func(func), iterable, chunksize)
    return gc_map


def close_pools():  # pragma: no cover
    for k, pool in pool_dict.items():
        pool.close()


atexit.register(close_pools)
