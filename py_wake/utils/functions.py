import numpy as np
from py_wake.utils import gradients


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
