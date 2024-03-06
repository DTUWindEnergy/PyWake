from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake import np
import pytest
import pandas as pd
import xarray as xr
from py_wake.site.xrsite import XRSite
from py_wake.tests import npt


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
    lw.add_ilk('TI_ilk', ti)
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
    lw.add_ilk('TI_ilk', ti)
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


def test_time_dims():
    time = np.arange(5)
    WS = np.array([[9, 8, 10, 11, 12],
                   [9, 8, 10, 11, 12]])

    # local WD in shape x, time (2, 5)
    WD = np.array([[20, 25, 30, 45, 50],
                   [20, 25, 30, 45, 50]])

    ds = xr.Dataset(
        data_vars=dict(TI=0.06, WS=(['x', "time"], WS), WD=(['x', "time"], WD), P=1 / len(time)),
        coords=dict(x=("x", [0, 1000]), time=time),
    )

    site = XRSite(ds=ds, interp_method='linear')
    lw = site.local_wind(x=0, y=0, wd=[0, 0, 0], ws=[0, 0, 0], time=[1, 2, 0])
    npt.assert_array_equal(lw['WS_ilk'].flatten(), [8, 10, 9])
    npt.assert_array_equal(lw['WD_ilk'].flatten(), [25, 30, 20])
    lw = site.local_wind(x=0, y=0, wd=[0, 0, 0], ws=[0, 0, 0], time=[1.25])
    npt.assert_array_equal(lw['WS_ilk'].flatten(), [8.5])
    npt.assert_array_equal(lw['WD_ilk'].flatten(), [26.25])
