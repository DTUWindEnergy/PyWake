from py_wake.site._site import UniformWeibullSite, UniformSite
import numpy as np
from numpy import newaxis as na
from py_wake.tests import npt
import pytest
from py_wake.site.shear import PowerShear

f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
     0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
     9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
     2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
ti = .1


@pytest.fixture
def site():
    return UniformWeibullSite(f, A, k, ti, shear=PowerShear(50, alpha=np.zeros_like(f) + .3))


def test_local_wind(site):
    x_i = y_i = np.arange(5)
    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x_i=x_i, y_i=y_i, wd=wdir_lst, ws=wsp_lst)
    npt.assert_array_equal(WS_ilk.shape, (5, 4, 3))

    WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x_i=x_i, y_i=y_i)
    npt.assert_array_equal(WS_ilk.shape, (5, 360, 23))

    # check probability local_wind()[-1]
    npt.assert_equal(site.local_wind(x_i=x_i, y_i=y_i, wd=[0], ws=[10], wd_bin_size=1)[-1],
                     site.local_wind(x_i=x_i, y_i=y_i, wd=[0], ws=[10], wd_bin_size=2)[-1] / 2)
    npt.assert_almost_equal(site.local_wind(x_i=x_i, y_i=y_i, wd=[0], ws=[9, 10, 11])[-1].sum((1, 2)),
                            site.local_wind(x_i=x_i, y_i=y_i, wd=[0], ws=[10], ws_bins=3)[-1][:, 0, 0], 5)

    z = np.arange(1, 100)
    zero = [0] * len(z)

    ws = site.local_wind(x_i=zero, y_i=zero, h_i=z, wd=[0], ws=[10])[1][:, 0, 0]
    site2 = UniformWeibullSite(f, A, k, ti, shear=PowerShear(70, alpha=np.zeros_like(f) + .3))
    ws70 = site2.local_wind(x_i=zero, y_i=zero, h_i=z, wd=[0], ws=[10])[1][:, 0, 0]
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(ws, z)
        plt.plot(ws70, z)
        plt.show()
    npt.assert_array_equal(10 * (z / 50)**.3, ws)
    npt.assert_array_equal(10 * (z / 70)**.3, ws70)


def test_elevation(site):
    x_i = y_i = np.arange(5)
    npt.assert_array_equal(site.elevation(x_i=x_i, y_i=y_i), np.zeros_like(x_i))


def test_site():
    with pytest.raises(NotImplementedError, match="interp_method=missing_method not implemeted yet."):
        site = UniformWeibullSite([1], [10], [2], .75, interp_method='missing_method')


def test_plot_ws_distribution(site):
    site.plot_ws_distribution(wd=[0, 90, 180, 270])
    site.plot_ws_distribution(wd=[0, 90, 180, 270], include_wd_distribution=True)
    if 0:
        import matplotlib.pyplot as plt
        plt.show()


def test_plot_wd_distribution(site):
    import matplotlib.pyplot as plt
    p1 = site.plot_wd_distribution(n_wd=12, ax=plt)
    npt.assert_array_almost_equal(p1, f, 4)
    plt.figure()
    site.plot_wd_distribution(n_wd=12, ax=plt.gca())
    plt.figure()
    p2 = site.plot_wd_distribution(n_wd=360)
    npt.assert_array_almost_equal(np.array(p2)[::30] * 30, f, 4)
    UniformWeibullSite(f, A, k, ti, 'spline').plot_wd_distribution(n_wd=360)
    UniformWeibullSite(f, A, k, ti, 'linear').plot_wd_distribution(n_wd=360)

    if 0:
        plt.show()
    plt.close()


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
        import matplotlib.pyplot as plt
        plt.show()


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
        import matplotlib.pyplot as plt
        plt.show()


def test_plot_ws_distribution_iea37():
    from py_wake.examples.data.iea37 import IEA37Site

    n_wt = 16  # must be 16, 32 or 64
    site = IEA37Site(n_wt)
    p = site.plot_ws_distribution(wd=[0])
    npt.assert_almost_equal(p, [1 / 300] * 300)

    if 0:
        import matplotlib.pyplot as plt
        plt.show()


def test_iea37_distances():
    from py_wake.examples.data.iea37 import IEA37Site

    n_wt = 16  # must be 9, 16, 36, 64
    site = IEA37Site(n_wt)
    x, y = site.initial_position.T
    WD_ilk, _, _, _ = site.local_wind(x_i=x, y_i=y,
                                      wd=site.default_wd,
                                      ws=site.default_ws)
    dw_iil, hcw_iil, _, _ = site.wt2wt_distances(
        x_i=x, y_i=y,
        h_i=np.zeros_like(x),
        wd_il=WD_ilk.mean(2))
    # Wind direction.
    wdir = np.rad2deg(np.arctan2(hcw_iil, dw_iil))
    npt.assert_allclose(
        wdir[:, 0, 0],
        [180, -90, -18, 54, 126, -162, -90, -54, -18, 18, 54, 90, 126, 162, -162, -126],
        atol=1e-4)

    if 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(np.arange(len(x))):
            ax.annotate(txt, (x[i], y[i]), fontsize='large')


def test_uniform_site_probability():
    """check that the uniform site recovers probability"""
    p_wd = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
    wd_l = np.linspace(0, 360, p_wd.size, endpoint=False)
    ws_k = np.array([12])
    site = UniformSite(p_wd, ti=1)
    actual = site.probability(0, 0, 0, wd_l[na, :, na], ws_k[na, na], 360 / p_wd.size, None)
    npt.assert_array_almost_equal(actual.squeeze(), p_wd)
