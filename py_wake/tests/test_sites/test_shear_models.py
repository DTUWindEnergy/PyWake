from py_wake.site.shear import PowerShear
import numpy as np
from py_wake.tests import npt
from py_wake.site._site import UniformSite
import xarray as xr
import matplotlib.pyplot as plt


def test_power_shear():
    shear = PowerShear(70, alpha=[.1, .2])
    h_lst = np.arange(10, 100, 10)
    site = UniformSite([1], .1)
    wref = site.wref([0, 180], [10])
    h = xr.DataArray(np.arange(10, 100, 10), [('i', np.arange(9))])
    u = shear(wref.WS, wref.WD, h)

    if 0:
        plt.plot(u.sel(wd=0), h, label='alpha=0.1')
        plt.plot((h_lst / 70)**0.1 * 10, h_lst, ':')
        plt.plot(u.sel(wd=180), h, label='alpha=0.2')
        plt.plot((h_lst / 70)**0.2 * 10, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(u.sel(wd=0, ws=10), [8.23, 8.82, 9.19, 9.46, 9.67, 9.85, 10., 10.13, 10.25], 2)
    npt.assert_array_almost_equal(u.sel(wd=180, ws=10), [6.78, 7.78, 8.44, 8.94, 9.35, 9.7, 10., 10.27, 10.52], 2)


if __name__ == '__main__':
    test_power_shear()
