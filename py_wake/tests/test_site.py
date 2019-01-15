from py_wake.site._site import UniformWeibullSite, UniformSite
import numpy as np
from py_wake.tests import npt
import pytest

f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
     0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
     9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
     2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
ti = .1


@pytest.fixture
def site():
    return UniformWeibullSite(f, A, k, ti, h_ref=50, alpha=.3)


def test_local_wind(site):
    x_i = y_i = np.arange(5)
    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x_i=x_i, y_i=y_i, wd=wdir_lst, ws=wsp_lst)
    npt.assert_array_equal(WS_ilk.shape, (5, 4, 3))

    WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x_i=x_i, y_i=y_i)
    npt.assert_array_equal(WS_ilk.shape, (5, 360, 23))

    # check probability local_wind()[-1]
    npt.assert_equal(site.local_wind(x_i=x_i, y_i=y_i, wd=[0], ws=[10])[-1],
                     site.local_wind(x_i=x_i, y_i=y_i, wd=[0], ws=[10], wd_bin_size=2)[-1] * 180)
    npt.assert_almost_equal(site.local_wind(x_i=x_i, y_i=y_i, wd=[0], ws=[9, 10, 11])[-1].sum(),
                            site.local_wind(x_i=x_i, y_i=y_i, wd=[0], ws=[10], ws_bin_size=3)[-1], 5)

    z = np.arange(1, 100)
    zero = [0] * len(z)

    ws = site.local_wind(x_i=zero, y_i=zero, h_i=z, wd=[0], ws=[10])[1][:, 0, 0]
    site.h_ref = 70
    ws70 = site.local_wind(x_i=zero, y_i=zero, h_i=z, wd=[0], ws=[10])[1][:, 0, 0]
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
    site.plot_wd_distribution(12, ax=plt)
    plt.figure()
    site.plot_wd_distribution(12, ax=plt.gca())
    plt.figure()
    site.plot_wd_distribution(360)
    UniformWeibullSite(f, A, k, ti, 'spline').plot_wd_distribution(360)
    UniformWeibullSite(f, A, k, ti, 'linear').plot_wd_distribution(360)

    if 0:
        import matplotlib.pyplot as plt
        plt.show()


def test_plot_wd_distribution_with_ws_levels(site):
    site.plot_wd_distribution(12, [0, 5, 10, 15, 20, 25])

    if 0:
        import matplotlib.pyplot as plt
        plt.show()


def test_plot_wd_distribution_with_ws_levels2(site):
    import matplotlib.pyplot as plt
    site.plot_wd_distribution(12, 6)

    if 0:
        import matplotlib.pyplot as plt
        plt.show()


def test_plot_ws_distribution_iea37():
    from py_wake.examples.data.iea37 import IEA37Site

    n_wt = 16  # must be 16, 32 or 64
    site = IEA37Site(n_wt)
    site.plot_ws_distribution(wd=[0])
    if 0:
        import matplotlib.pyplot as plt
        plt.show()
