from py_wake.site._site import UniformWeibullSite
import numpy as np
from py_wake.tests import npt


def test_local_wind():
    f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
         0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
    A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
         9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
    k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
         2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
    ti = .1
    site = UniformWeibullSite(f, A, k, ti)

    x_i = y_i = np.arange(5)
    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x_i, y_i, wdir_lst, wsp_lst)
    npt.assert_array_equal(WS_ilk.shape, (5, 4, 3))
    WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x_i, y_i)
    npt.assert_array_equal(WS_ilk.shape, (5, 360, 23))
    npt.assert_array_equal(site.elevation(x_i, y_i), np.zeros_like(x_i))

    npt.assert_equal(site.local_wind(x_i, y_i, [0], [10])[-1] * 2,
                     site.local_wind(x_i, y_i, [0], [10], wd_bin_size=2)[-1])
    npt.assert_equal(site.local_wind(x_i, y_i, [0], [9, 10, 11])[-1].sum(),
                     site.local_wind(x_i, y_i, [0], [10], ws_bin_size=3)[-1])
