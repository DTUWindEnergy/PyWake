import multiprocessing
import atexit
import platform

pool_dict = {}


def get_pool(processes=multiprocessing.cpu_count()):
    if processes not in pool_dict:
        if platform.system() == 'Darwin':  # pragma: no cover
            pool_dict[processes] = multiprocessing.get_context('fork').Pool(processes)
        else:
            pool_dict[processes] = multiprocessing.Pool(processes)
    return pool_dict[processes]


def close_pools():  # pragma: no cover
    for k, pool in pool_dict.items():
        pool.close()


atexit.register(close_pools)
