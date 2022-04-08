import multiprocessing
import atexit

pool_dict = {}


def get_pool(processes=multiprocessing.cpu_count()):
    if processes not in pool_dict:
        pool_dict[processes] = multiprocessing.Pool(processes)
    return pool_dict[processes]


def close_pools():  # pragma: no cover
    for k, pool in pool_dict.items():
        pool.close()


atexit.register(close_pools)
