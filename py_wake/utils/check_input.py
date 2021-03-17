import numpy as np
import pytest


def check_input(input_space_lst, input_lst, input_keys=None):
    if input_keys is None:
        input_keys = ["index_%d" % i for i in range(len(input_space_lst))]

    for input, input_space, key in zip(input_lst, input_space_lst, input_keys):
        if np.min(input) < np.min(input_space):
            v = np.min(input)
        elif np.max(input) > np.max(input_space):
            v = np.max(input)
        else:
            continue  # pragma: no cover # Is covered but not registered
        mi, ma = np.min(input_space), np.max(input_space)
        raise ValueError(f"Input, {key}, with value, {v} outside range {mi}-{ma}")
