from py_wake.examples.data.hornsrev1 import Hornsrev1Site
import numpy as np
import pytest
import pandas as pd


@pytest.mark.parametrize(['ti', 'dims'], [
    (0.1, ()),
    (np.full((2), 0.1), ('i',)),
    (np.full((2, 360), 0.1), ('i', 'wd')),
    (np.full((2, 360, 23), 0.1), ('i', 'wd', 'ws')),
    (np.full((360), 0.1), ('wd',)),
    (np.full((360, 23), 0.1), ('wd', 'ws')),
    (np.full((23), 0.1), ('ws',)),
])
def test_add_ilk(ti, dims):
    site = Hornsrev1Site()
    wt, wd, ws = np.arange(2), np.arange(360), np.arange(3, 26)
    lw = site.local_wind(wt * 1000, wt * 0, wt * 0 + 70, wd, ws, time=False)
    lw.add_ilk('TI', ti)
    assert lw.TI.dims == dims


@pytest.mark.parametrize(['ti', 'dims'], [
    (0.1, ()),
    (np.full((2), 0.1), ('i',)),
    (np.full((2, 100), 0.1), ('i', 'time')),
    (np.full((100), 0.1), ('time',)),
])
def test_add_ilk_time(ti, dims):
    site = Hornsrev1Site()
    wt, wd, ws = np.arange(2), np.arange(100), np.arange(100) % 20 + 3
    t = pd.date_range("2000-01-01", freq="10T", periods=100)

    lw = site.local_wind(wt * 1000, wt * 0, wt * 0 + 70, wd, ws, time=t)
    lw.add_ilk('TI', ti)
    assert lw.TI.dims == dims


@pytest.mark.parametrize(['shape'], [
    [(360, 2)],
    [(23, 360)],
    [(23, 2)],
    [(2, 23, 360)],
    [(360, 2, 23)],
    [(360, 23, 2)],
    [(23, 360, 2)],
    [(23, 2, 360)],
])
def test_add_ilk_wrong_dim(shape):
    site = Hornsrev1Site()
    wt, wd, ws = np.arange(2), np.arange(360), np.arange(3, 26)
    lw = site.local_wind(wt * 1000, wt * 0, wt * 0 + 70, wd, ws, time=False)
    with pytest.raises(ValueError):
        lw.add_ilk('TI', np.full(shape, 0.1))


@pytest.mark.parametrize(['shape'], [
    [(100, 2)],
    [(2, 100, 23)],
])
def test_add_ilk_time_wrong_dim(shape):
    site = Hornsrev1Site()
    wt, wd, ws = np.arange(2), np.arange(100), np.arange(100) % 20 + 3
    t = pd.date_range("2000-01-01", freq="10T", periods=100)
    lw = site.local_wind(wt * 1000, wt * 0, wt * 0 + 70, wd, ws, time=t)

    with pytest.raises(ValueError):
        lw.add_ilk('TI', np.full(shape, 0.1))
