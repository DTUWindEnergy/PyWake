from py_wake.site.shear import PowerShear, LogShear
import numpy as np
from py_wake.tests import npt
from py_wake.site._site import UniformSite
import xarray as xr
import matplotlib.pyplot as plt


def test_power_shear():
    h_lst = np.arange(10, 100, 10)
    site = UniformSite([1], .1, shear=PowerShear(70, alpha=[.1, .2]))
    WS = site.local_wind(x_i=h_lst * 0, y_i=h_lst * 0, h_i=h_lst, wd=[0, 180], ws=[10, 12]).WS

    if 0:
        plt.plot(WS.sel(wd=0, ws=10), WS.h, label='alpha=0.1')
        plt.plot((h_lst / 70)**0.1 * 10, h_lst, ':')
        plt.plot(WS.sel(wd=180, ws=12), WS.h, label='alpha=0.2')
        plt.plot((h_lst / 70)**0.2 * 12, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_equal(WS.sel(wd=0, ws=10), (h_lst / 70)**0.1 * 10)
    npt.assert_array_equal(WS.sel(wd=180, ws=12), (h_lst / 70)**0.2 * 12)


def test_log_shear():

    h_lst = np.arange(10, 100, 10)
    site = UniformSite([1], .1, shear=LogShear(70, z0=[.02, 2]))
    WS = site.local_wind(x_i=h_lst * 0, y_i=h_lst * 0, h_i=h_lst, wd=[0, 180], ws=[10, 12]).WS

    if 0:
        plt.plot(WS.sel(wd=0, ws=10), WS.h, label='z0=0.02')
        plt.plot(np.log(h_lst / 0.02) / np.log(70 / 0.02) * 10, h_lst, ':')
        plt.plot(WS.sel(wd=180, ws=12), WS.h, label='z0=2')
        plt.plot(np.log(h_lst / 2) / np.log(70 / 2) * 12, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_equal(WS.sel(wd=0, ws=10), np.log(h_lst / 0.02) / np.log(70 / 0.02) * 10)
    npt.assert_array_equal(WS.sel(wd=180, ws=12), np.log(h_lst / 2) / np.log(70 / 2) * 12)


def test_log_shear_constant_z0():
    h_lst = np.arange(10, 100, 10)
    site = UniformSite([1], .1, shear=LogShear(70, z0=.02))
    WS = site.local_wind(x_i=h_lst * 0, y_i=h_lst * 0, h_i=h_lst, wd=[0, 180], ws=[10, 12]).WS

    if 0:
        plt.plot(WS.sel(ws=10), WS.h, label='z0=0.02')
        plt.plot(np.log(h_lst / 0.02) / np.log(70 / 0.02) * 10, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_equal(WS.sel(ws=10), np.log(h_lst / 0.02) / np.log(70 / 0.02) * 10)


def test_custom_shear():
    def my_shear(WS, WD, h):
        return WS * (0.02 * (h - 70) + 1) * (WD * 0 + 1)
    h_lst = np.arange(10, 100, 10)

    site = UniformSite([1], .1, shear=my_shear)
    WS = site.local_wind(x_i=h_lst * 0, y_i=h_lst * 0, h_i=h_lst, wd=[0, 180], ws=[10, 12]).WS

    if 0:
        plt.plot(WS.sel(wd=0, ws=10), WS.h, label='2z-2')
        plt.plot((h_lst - 70) * 0.2 + 10, h_lst, ':')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(WS.sel(wd=0, ws=10), (h_lst - 70) * 0.2 + 10)


if __name__ == '__main__':
    test_power_shear()
