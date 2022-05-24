import numpy as np
from py_wake.utils import gradients
import xarray as xr


def mean_deg(dir, axis=0):
    """Mean of angles in degrees

    Parameters
    ----------
    dir : array_like
        Angles in degrees
    axis : int
        if dir is 2d array_like, axis defines which axis to take the mean of

    Returns
    -------
    mean_deg : float
        Mean angle
    """
    return gradients.rad2deg(mean_rad(gradients.deg2rad(dir), axis))


def mean_rad(dir, axis=0):
    """Mean of angles in radians

    Parameters
    ----------
    dir : array_like
        Angles in radians
    axis : int
        if dir is 2d array_like, axis defines which axis to take the mean of

    Returns
    -------
    mean_rad : float
        Mean angle
    """
    return gradients.arctan2(np.mean(np.sin(dir[:]), axis), np.mean(np.cos(dir[:]), axis))


# class ilk_array(np.ndarray):
#     def __new__(cls, input_array):
#         if isinstance(input_array, xr.DataArray):
#             input_array = input_array.ilk()
#         obj = np.asarray(input_array).view(cls)
#         assert len(obj.shape) == 3, f"ilk_array must have 3 dimensions but has shape {obj.shape}"
#         return obj
#
#     def __array_finalize__(self, obj):
#         if obj is None:
#             return
#
#     def __getitem__(self, indices):
#         try:
#             return np.array(self).__getitem__(indices)
#         except IndexError:
#             ilk_indices = (indices + (slice(None), slice(None), slice(None)))[:3]
#
#             def get_index(i, dim):
#                 if isinstance(i, int):
#                     if [self.shape[dim] == 1]:
#                         i = 0
#                     return slice(i, i + 1)
#                 return i
#
#             ilk_indices = tuple([get_index(ilk, dim) for dim, ilk in enumerate(ilk_indices)])
#             return np.ndarray.__getitem__(self, ilk_indices)
