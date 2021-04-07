from py_wake.site._site import UniformWeibullSite, UniformSite
import numpy as np
from py_wake.tests import npt
import pytest
from py_wake.site.shear import PowerShear
import matplotlib.pyplot as plt
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.site.xrsite import XRSite

f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
     0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
     9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
     2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
ti = .1


@pytest.fixture(autouse=True)
def close_plots():
    yield
    try:
        plt.close('all')
    except Exception:
        pass


@pytest.fixture
def site():
    return UniformWeibullSite(f, A, k, ti, shear=PowerShear(50, alpha=np.zeros_like(f) + .3))


def test_local_wind(site):
    x_i = y_i = np.arange(5)
    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    lw = site.local_wind(x_i=x_i, y_i=y_i, h_i=50, wd=wdir_lst, ws=wsp_lst)
    npt.assert_array_equal(lw.WS_ilk.shape, (1, 4, 3))

    lw = site.local_wind(x_i=x_i, y_i=y_i, h_i=50)
    npt.assert_array_equal(lw.WS_ilk.shape, (1, 360, 23))

    # check probability local_wind()[-1]
    npt.assert_equal(site.local_wind(x_i=x_i, y_i=y_i, h_i=50, wd=[0], ws=[10], wd_bin_size=1).P_ilk,
                     site.local_wind(x_i=x_i, y_i=y_i, h_i=50, wd=[0], ws=[10], wd_bin_size=2).P_ilk / 2)
    npt.assert_almost_equal(site.local_wind(x_i=x_i, y_i=y_i, h_i=50, wd=[0], ws=[9, 10, 11]).P_ilk.sum((1, 2)),
                            site.local_wind(x_i=x_i, y_i=y_i, h_i=50, wd=[0], ws=[10], ws_bins=3).P_ilk[:, 0, 0], 5)

    z = np.arange(1, 100)
    zero = [0] * len(z)

    ws = site.local_wind(x_i=zero, y_i=zero, h_i=z, wd=[0], ws=[10]).WS_ilk[:, 0, 0]
    site2 = UniformWeibullSite(f, A, k, ti, shear=PowerShear(70, alpha=np.zeros_like(f) + .3))
    ws70 = site2.local_wind(x_i=zero, y_i=zero, h_i=z, wd=[0], ws=[10]).WS_ilk[:, 0, 0]
    if 0:
        plt.plot(ws, z)
        plt.plot(ws70, z)
        plt.show()
    npt.assert_array_equal(10 * (z / 50)**.3, ws)
    npt.assert_array_equal(10 * (z / 70)**.3, ws70)


def test_elevation(site):
    x_i = y_i = np.arange(5)
    npt.assert_array_equal(site.elevation(x_i=x_i, y_i=y_i), np.zeros_like(x_i))


def test_missing_interp_method():
    with pytest.raises(AssertionError, match='interp_method "missing_method" not implemented. Must be "linear" or "nearest"'):
        site = UniformWeibullSite([1], [10], [2], .75, interp_method='missing_method')


def test_ws_bins(site):
    npt.assert_array_equal(site.ws_bins([3, 4, 5]).ws_lower, [2.5, 3.5, 4.5])
    npt.assert_array_equal(site.ws_bins([3, 4, 5]).ws_upper, [3.5, 4.5, 5.5])
    npt.assert_array_equal(site.ws_bins(4).ws_lower, [3.5])
    npt.assert_array_equal(site.ws_bins(4).ws_upper, [4.5])
    npt.assert_array_equal(site.ws_bins([3, 4, 5], [2.5, 3.5, 4.5, 5.5]).ws_lower, [2.5, 3.5, 4.5])
    npt.assert_array_equal(site.ws_bins([3, 4, 5], [2.5, 3.5, 4.5, 5.5]).ws_upper, [3.5, 4.5, 5.5])


def test_plot_ws_distribution(site):
    p = site.plot_ws_distribution(wd=[0, 90, 180, 270])
    npt.assert_array_almost_equal(p.ws[p.argmax('ws')], [7.35, 8.25, 7.95, 9.75])
    npt.assert_array_almost_equal(p.max('ws'), [0.01063067, 0.01048959, 0.01081972, 0.00893762])

    plt.figure()
    p = site.plot_ws_distribution(wd=[0, 90, 180, 270], include_wd_distribution=True)
    npt.assert_array_almost_equal(p.ws[p.argmax('ws')], [7.75, 8.25, 8.15, 9.55])
    npt.assert_array_almost_equal(p.max('ws'), [0.001269, 0.002169, 0.002771, 0.003548])

    with pytest.raises(AssertionError, match="Wind directions must be equidistant"):
        p = site.plot_ws_distribution(wd=[0, 180, 270], include_wd_distribution=True)

    plt.figure()
    p = site.plot_ws_distribution(wd=[90, 180, 270, 0], include_wd_distribution=True)
    npt.assert_array_almost_equal(p.ws[p.argmax('ws')], [8.25, 8.15, 9.55, 7.75, ])
    npt.assert_array_almost_equal(p.max('ws'), [0.002182, 0.002771, 0.003548, 0.001257])

    if 0:
        plt.show()
    plt.close('all')


def test_plot_wd_distribution(site):
    p1 = site.plot_wd_distribution(n_wd=12, ax=plt)
    npt.assert_array_almost_equal(p1, f, 4)
    plt.figure()
    site.plot_wd_distribution(n_wd=12, ax=plt.gca())
    plt.figure()
    p2 = site.plot_wd_distribution(n_wd=360)
    npt.assert_array_almost_equal(np.array(p2)[::30] * 30, f, 4)
    # UniformWeibullSite(f, A, k, ti, 'spline').plot_wd_distribution(n_wd=360)
    UniformWeibullSite(f, A, k, ti, 'linear').plot_wd_distribution(n_wd=360)

    if 0:
        plt.show()
    plt.close('all')


def test_plot_wd_distribution_uniformSite():
    site = IEA37Site(16)

    p1 = site.plot_wd_distribution(n_wd=12, ax=plt)
    if 0:
        plt.show()
    plt.close('all')


def test_plot_wd_distribution_with_ws_levels(site):
    p = site.plot_wd_distribution(n_wd=12, ws_bins=[0, 5, 10, 15, 20, 25])
    # print(np.round(p, 4).tolist())
    npt.assert_array_almost_equal(p, [[0.0075, 0.0179, 0.0091, 0.0014, 0.0001],
                                      [0.0069, 0.0188, 0.0115, 0.0022, 0.0001],
                                      [0.0098, 0.025, 0.0142, 0.0025, 0.0001],
                                      [0.0109, 0.0339, 0.0214, 0.0036, 0.0001],
                                      [0.0114, 0.0411, 0.0271, 0.004, 0.0001],
                                      [0.0108, 0.0324, 0.0185, 0.0026, 0.0001],
                                      [0.0147, 0.0434, 0.0247, 0.0035, 0.0001],
                                      [0.0164, 0.0524, 0.0389, 0.0092, 0.0007],
                                      [0.0185, 0.0595, 0.0524, 0.0184, 0.0026],
                                      [0.0153, 0.0564, 0.054, 0.0191, 0.0024],
                                      [0.0103, 0.0386, 0.0369, 0.0127, 0.0015],
                                      [0.0092, 0.0231, 0.0152, 0.0038, 0.0004]], 4)

    if 0:
        plt.show()
    plt.close('all')


def test_plot_wd_distribution_with_ws_levels_xr(site):
    import xarray as xr
    ds = xr.Dataset(
        data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k)},
        coords={'wd': np.linspace(0, 360, len(f), endpoint=False)})
    site2 = XRSite(ds, shear=PowerShear(h_ref=100, alpha=.2), interp_method='nearest')

    p = site2.plot_wd_distribution(n_wd=12, ws_bins=[0, 5, 10, 15, 20, 25])
    if 0:
        plt.show()
    # print(np.round(p.values, 4).tolist())
    npt.assert_array_almost_equal(p, [[0.0075, 0.0179, 0.0091, 0.0014, 0.0001],
                                      [0.0069, 0.0188, 0.0115, 0.0022, 0.0001],
                                      [0.0098, 0.025, 0.0142, 0.0025, 0.0001],
                                      [0.0109, 0.0339, 0.0214, 0.0036, 0.0001],
                                      [0.0114, 0.0411, 0.0271, 0.004, 0.0001],
                                      [0.0108, 0.0324, 0.0185, 0.0026, 0.0001],
                                      [0.0147, 0.0434, 0.0247, 0.0035, 0.0001],
                                      [0.0164, 0.0524, 0.0389, 0.0092, 0.0007],
                                      [0.0185, 0.0595, 0.0524, 0.0184, 0.0026],
                                      [0.0153, 0.0564, 0.054, 0.0191, 0.0024],
                                      [0.0103, 0.0386, 0.0369, 0.0127, 0.0015],
                                      [0.0092, 0.0231, 0.0152, 0.0038, 0.0004]], 4)

    plt.close('all')


def test_plot_wd_distribution_with_ws_levels2(site):
    p = site.plot_wd_distribution(n_wd=12, ws_bins=6)
    # print(np.round(p, 3).tolist())
    npt.assert_array_almost_equal(p, [[0.011, 0.02, 0.005, 0.0, 0.0],
                                      [0.01, 0.022, 0.007, 0.0, 0.0],
                                      [0.014, 0.028, 0.008, 0.0, 0.0],
                                      [0.017, 0.039, 0.013, 0.001, 0.0],
                                      [0.018, 0.049, 0.016, 0.001, 0.0],
                                      [0.017, 0.038, 0.011, 0.0, 0.0],
                                      [0.022, 0.049, 0.014, 0.001, 0.0],
                                      [0.025, 0.063, 0.026, 0.002, 0.0],
                                      [0.028, 0.074, 0.041, 0.006, 0.0],
                                      [0.024, 0.073, 0.044, 0.007, 0.0],
                                      [0.016, 0.051, 0.03, 0.004, 0.0],
                                      [0.013, 0.028, 0.011, 0.001, 0.0]], 3)
    if 0:
        plt.show()
    plt.close('all')


def test_plot_ws_distribution_iea37():
    from py_wake.examples.data.iea37 import IEA37Site

    n_wt = 16  # must be 16, 32 or 64
    site = IEA37Site(n_wt)
    p = site.plot_ws_distribution(wd=[0])
    npt.assert_almost_equal(p, [[1 / 300] * 300])

    if 0:
        plt.show()


def test_iea37_distances():
    from py_wake.examples.data.iea37 import IEA37Site

    n_wt = 16  # must be 9, 16, 36, 64
    site = IEA37Site(n_wt)
    x, y = site.initial_position.T
    lw = site.local_wind(x_i=x, y_i=y,
                         wd=site.default_wd,
                         ws=site.default_ws)
    site.distance.setup(x, y, np.zeros_like(x))
    dw_iil, hcw_iil, _ = site.wt2wt_distances(wd_il=lw.WD_ilk.mean(2))
    # Wind direction.
    wdir = np.rad2deg(np.arctan2(hcw_iil, dw_iil))
    npt.assert_allclose(
        wdir[:, 0, 0],
        [180, -90, -18, 54, 126, -162, -90, -54, -18, 18, 54, 90, 126, 162, -162, -126],
        atol=1e-4)

    if 0:
        ax = plt.subplots()[1]
        ax.scatter(x, y)
        for i, txt in enumerate(np.arange(len(x))):
            ax.annotate(txt, (x[i], y[i]), fontsize='large')


def test_uniform_site_probability():
    """check that the uniform site recovers probability"""
    p_wd = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
    site = UniformSite(p_wd, ti=1)
    lw = site.local_wind(0, 0, 0, wd=np.linspace(0, 360, p_wd.size, endpoint=False), ws=12)
    npt.assert_array_almost_equal(lw.P, p_wd)
