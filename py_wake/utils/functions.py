from py_wake import np
from py_wake.utils import gradients
from numpy import newaxis as na
import warnings


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
    return gradients.rad2deg(mean_rad(gradients.deg2rad(np.asarray(dir)), axis))


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
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'invalid value encountered in divide')
        return gradients.arctan2(np.mean(np.sin(dir[:]), axis), np.mean(np.cos(dir[:]), axis))


def arg2ilk(k, v, I, L, K):
    v = np.asarray(v)
    if v.shape == ():
        v = v[na, na, na]
    elif v.shape in [(I,), (1,)]:
        v = v[:, na, na]
    elif v.shape in [(I, L), (1, L), (I, 1), (1, 1)]:
        v = v[:, :, na]
    elif v.shape in {(I, L, K),
                     (1, L, K), (I, 1, K), (I, L, 1),
                     (1, 1, K), (1, L, 1), (I, 1, 1),
                     (1, 1, 1)}:
        pass
    elif v.shape == (L,):
        v = v[na, :, na]
    elif v.shape in [(L, K), (L, 1), (1, K)]:
        v = v[na, :, :]
    elif v.shape == (K,):
        v = v[na, na]
    else:
        valid_shapes = f"(), ({I}), ({I},{L}), ({I},{L},{K}), ({L},), ({L}, {K})"
        raise ValueError(
            f"Argument, {k}(shape={v.shape}), has unsupported shape. Valid shapes are {valid_shapes} (interpreted in this order)")
    return v
