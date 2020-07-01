import numpy as np
from py_wake.site.shear import PowerShear
from py_wake.site.xrsite import XRSite
import xarray as xr
from py_wake.tests import npt
import pytest
import matplotlib.pyplot as plt
from numpy import newaxis as na
from py_wake.tests.test_files import tfp
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, wt9_x, wt9_y
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.wind_turbines import WindTurbines
from py_wake.deficit_models.gaussian import BastankhahGaussian, BastankhahGaussianDeficit
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.superposition_models import LinearSum
from py_wake.tests.check_speed import timeit
from py_wake.site._site import LocalWind, UniformWeibullSite


f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
     0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
     9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
     2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
ti = .1


@pytest.fixture
def uniform_site():
    ti = 0.1
    ds = xr.Dataset(
        data_vars={'WS': 10, 'P': ('wd', f), 'TI': ti},
        coords={'wd': np.linspace(0, 360, len(f), endpoint=False)})
    return XRSite(ds, shear=PowerShear(h_ref=100, alpha=.2))


@pytest.fixture
def uniform_weibull_site():
    ti = 0.1
    ds = xr.Dataset(
        data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ti},
        coords={'wd': np.linspace(0, 360, len(f), endpoint=False)})
    return XRSite(ds, shear=PowerShear(h_ref=100, alpha=.2))


@pytest.fixture
def complex_fixed_pos_site():
    ti = 0.1

    ds = xr.Dataset(
        data_vars={'Speedup': ('i', np.arange(.8, 1.3, .1)),
                   'Turning': ('i', np.arange(-2, 3)),
                   'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ti},
        coords={'i': np.arange(5), 'wd': np.linspace(0, 360, len(f), endpoint=False)})
    x_i = np.arange(5)
    return XRSite(ds, initial_position=np.array([x_i, x_i + 1]).T, shear=PowerShear(h_ref=100, alpha=.2))


@pytest.fixture
def complex_grid_site():
    ti = 0.1
    ds = xr.Dataset(
        data_vars={'Speedup': (['x', 'y'], np.arange(.8, 1.4, .1).reshape((3, 2))),
                   'Turning': (['x', 'y'], np.arange(-2, 4, 1).reshape((3, 2))),
                   'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ti},
        coords={'x': [0, 5, 10], 'y': [0, 5], 'wd': np.linspace(0, 360, len(f), endpoint=False)})
    return XRSite(ds, shear=PowerShear(h_ref=100, alpha=.2), interp_method='linear')


def test_uniform_local_wind(uniform_site):
    site = uniform_site
    x_i = y_i = np.arange(5)

    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    lw = site.local_wind(x_i=x_i, y_i=y_i, h_i=100, wd=wdir_lst, ws=wsp_lst)
    npt.assert_array_equal(lw.WS, 10)
    npt.assert_array_equal(lw.WD, wdir_lst)
    npt.assert_array_equal(lw.TI, 0.1)
    npt.assert_array_equal(lw.P, np.array([0.035972, 0.070002, 0.086432, 0.147379]) * 3)
    npt.assert_array_equal(site.elevation(x_i, y_i), 0)

    lw = site.local_wind(x_i=x_i, y_i=y_i, h_i=100)
    npt.assert_array_equal(lw.WD_ilk.shape, (1, 360, 1))

    z = np.arange(1, 200)
    zero = [0] * len(z)

    ws100_2 = site.local_wind(x_i=zero, y_i=zero, h_i=z, wd=[0], ws=[10]).WS_ilk[:, 0, 0]

    site.shear = PowerShear(70, alpha=.3)
    ws70_3 = site.local_wind(x_i=zero, y_i=zero, h_i=z, wd=[0], ws=[10]).WS_ilk[:, 0, 0]
    if 0:
        plt.plot(ws100_2, z)
        plt.plot(ws70_3, z)
        plt.show()
    npt.assert_array_equal(10 * (z / 100)**.2, ws100_2)
    npt.assert_array_equal(10 * (z / 70)**.3, ws70_3)


def test_uniform_weibull_local_wind(uniform_weibull_site):
    site = uniform_weibull_site
    x_i = y_i = np.arange(5)

    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    lw = site.local_wind(x_i=x_i, y_i=y_i, h_i=100, wd=wdir_lst, ws=wsp_lst)

    npt.assert_array_equal(lw.WS, [3, 4, 5])
    npt.assert_array_equal(lw.WD, wdir_lst)
    npt.assert_array_equal(lw.TI, 0.1)

#     ref_site = UniformWeibullSite(p_wd=f, a=A, k=k, ti=0.1, shear=site.shear)
#     lw_ref = ref_site.local_wind(x_i=x_i, y_i=y_i, h_i=100, wd=wdir_lst, ws=wsp_lst)
#     print(lw_ref.P)
    npt.assert_array_almost_equal(lw.P, [[0.00553222, 0.00770323, 0.00953559],
                                         [0.0078508, 0.01177761, 0.01557493],
                                         [0.0105829, 0.01576518, 0.02066746],
                                         [0.01079997, 0.01656828, 0.02257487]])


def test_complex_fixed_pos_local_wind(complex_fixed_pos_site):
    site = complex_fixed_pos_site
    x_i, y_i = site.initial_position.T
    npt.assert_array_equal(x_i, np.arange(5))
    npt.assert_array_equal(y_i, np.arange(5) + 1)

    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    lw = site.local_wind(x_i=x_i, y_i=y_i, h_i=100, wd=wdir_lst, ws=wsp_lst)

    npt.assert_array_equal(lw.WS, [3, 4, 5] * np.arange(.8, 1.3, .1)[:, na])
    npt.assert_array_equal(lw.WD, (wdir_lst + np.arange(-2, 3)[:, na]) % 360)
    npt.assert_array_equal(lw.TI, 0.1)

    npt.assert_array_almost_equal(lw.P, [[0.00553222, 0.00770323, 0.00953559],
                                         [0.0078508, 0.01177761, 0.01557493],
                                         [0.0105829, 0.01576518, 0.02066746],
                                         [0.01079997, 0.01656828, 0.02257487]])


def test_complex_grid_local_wind(complex_grid_site):
    site = complex_grid_site
    y = np.arange(5)
    x = y * 2
    X, Y = np.meshgrid(x, y)
    x_i, y_i = X.flatten(), Y.flatten()

    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    lw = site.local_wind(x_i=x_i, y_i=y_i, h_i=100, wd=wdir_lst, ws=wsp_lst)
    if 0:
        c = plt.contourf(X, Y, lw.WS.sel(ws=5).data.reshape(X.shape))
        plt.colorbar(c)
        plt.figure()
        c = plt.contourf(X, Y, lw.WD.sel(wd=90).data.reshape(X.shape))
        plt.colorbar(c)
        plt.show()

    lw = site.local_wind(x_i=[2.5, 7.5], y_i=[2.5, 2.5], h_i=100, wd=wdir_lst, ws=wsp_lst)
    npt.assert_array_almost_equal(lw.WS, [3, 4, 5] * np.array([0.95, 1.15])[:, na])
    npt.assert_array_equal(lw.WD, (wdir_lst + np.array([-.5, 1.5])[:, na]) % 360)
    npt.assert_array_equal(lw.TI, 0.1)

    npt.assert_array_almost_equal(lw.P, [[0.00553222, 0.00770323, 0.00953559],
                                         [0.0078508, 0.01177761, 0.01557493],
                                         [0.0105829, 0.01576518, 0.02066746],
                                         [0.01079997, 0.01656828, 0.02257487]])


def test_wd_independent_site():
    ti = 0.1
    ds = xr.Dataset(
        data_vars={
            'WS': 10, 'Sector_frequency': 1,
            'Weibull_A': 4, 'Weibull_k': 2, 'TI': ti},
        coords={})
    site = XRSite(ds, shear=None)
    npt.assert_equal(site.ds.sector_width, 360)


def test_load_save(complex_grid_site):
    complex_grid_site.save(tfp + "tmp.nc")
    site = XRSite.load(tfp + "tmp.nc", interp_method='linear')
    test_complex_grid_local_wind(site)


def test_elevation():
    ti = 0.1
    ds = xr.Dataset(
        data_vars={'Elevation': (['x', 'y'], np.arange(.8, 1.4, .1).reshape((3, 2))),
                   'P': 1, 'TI': ti},
        coords={'x': [0, 5, 10], 'y': [0, 5]})
    site = XRSite(ds)
    npt.assert_array_almost_equal(site.elevation([2.5, 7.5], [2.5, 2.5]), [0.95, 1.15])


def test_cyclic_interpolation(uniform_site):
    site = uniform_site
    lw = site.local_wind([0], [0], 100, wd=np.arange(360), ws=10)
    if 0:
        plt.plot(np.linspace(0, 360, len(f) + 1), np.r_[f, f[0]], '.')
        plt.plot(lw.wd, lw.P)
        plt.show()
    npt.assert_array_almost_equal(lw.P[::30], np.array(f) / 30)


def test_from_flow_box_2wt():
    site = Hornsrev1Site()
    windTurbines = V80()

    # simulate current and neighbour wt
    wfm = BastankhahGaussian(site, windTurbines)
    wd = np.arange(30)
    sim_res = wfm([0, 0], [0, 500], wd=wd)
    ref_aep = sim_res.aep().sel(wt=0)

    wt_x, wt_y = [0], [0]
    neighbour_x, neighbour_y = [0], [500]

    # make site with effects of neighbour wt
    sim_res = wfm(neighbour_x, neighbour_y, wd=wd)
    e = 100
    box = sim_res.flow_box(x=np.linspace(min(wt_x) - e, max(wt_x) + e, 21),
                           y=np.linspace(min(wt_y) - e, max(wt_y) + e, 21),
                           h=windTurbines.hub_height(windTurbines.types()))
    site = XRSite.from_flow_box(box)

    # Simujlate current wt and compare aep
    wfm = BastankhahGaussian(site, windTurbines)
    sim_res = wfm(wt_x, wt_y, wd=wd)
    aep = sim_res.aep()

    if 0:
        site.ds.WS.sel(ws=10, wd=3).plot()
        windTurbines.plot(wt_x, wt_y)
        windTurbines.plot(neighbour_x, neighbour_y)
        plt.show()

    npt.assert_array_almost_equal(ref_aep, aep.sel(wt=0))


def test_neighbour_farm_speed():
    # import and setup site and windTurbines
    site = IEA37Site(16)

    # setup current, neighbour and all positions
    wt_x, wt_y = site.initial_position.T
    neighbour_x, neighbour_y = wt_x - 4000, wt_y
    all_x, all_y = np.r_[wt_x, neighbour_x], np.r_[wt_y, neighbour_y]

    windTurbines = WindTurbines.from_WindTurbines([IEA37_WindTurbines(), IEA37_WindTurbines()])
    windTurbines._names = ["Current wind farm", "Neighbour wind farm"]
    types = [0] * len(wt_x) + [1] * len(neighbour_x)

    wf_model = PropagateDownwind(site, windTurbines,
                                 wake_deficitModel=BastankhahGaussianDeficit(use_effective_ws=True),
                                 superpositionModel=LinearSum())
    # Consider wd=270 +/- 30 deg only
    wd_lst = np.arange(240, 301)

    sim_res, t = timeit(wf_model, verbose=False)(all_x, all_y, type=types, ws=9.8, wd=wd_lst)
    if 1:
        ext = 100
        flow_box = wf_model(neighbour_x, neighbour_y, wd=wd_lst).flow_box(
            x=np.linspace(min(wt_x) - ext, max(wt_x) + ext, 53),
            y=np.linspace(min(wt_y) - ext, max(wt_y) + ext, 51),
            h=[100, 110, 120])

        wake_site = XRSite.from_flow_box(flow_box)
        wake_site.save('tmp.nc')
    else:
        wake_site = XRSite.load('tmp.nc')

    wf_model_wake_site = PropagateDownwind(wake_site, windTurbines,
                                           wake_deficitModel=BastankhahGaussianDeficit(use_effective_ws=True),
                                           superpositionModel=LinearSum())

    sim_res_wake_site, _ = timeit(wf_model_wake_site, verbose=False)(wt_x, wt_y, ws=9.8, wd=wd_lst)
    npt.assert_allclose(sim_res.aep().sel(wt=np.arange(len(wt_x))).sum(), sim_res_wake_site.aep().sum(), rtol=0.0005)
    npt.assert_array_almost_equal(sim_res.aep().sel(wt=np.arange(len(wt_x))), sim_res_wake_site.aep(), 2)


@pytest.mark.parametrize('h,wd,ws,h_i,wd_l,ws_k', [
    ([100, 110, 120], range(360), [9.8], 110, range(360), [9.8]),
    ([100, 110, 120], range(5, 25), [9.8], 110, range(10, 20), [9.8]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, range(10, 20), [10]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, range(10, 20), [10, 11]),
    ([100, 110, 120], range(5, 25), [8, 10, 11], 110, range(10, 20), [9.8]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, range(10, 20), [9.8]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, range(10, 20), [9.8, 10.2]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, [11.5, 12.5], [10]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, [11.5, 12.5], [10]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, [11.5, 12.5], [9.8, 10.2]),
])
def test_interp(h, wd, ws, h_i, wd_l, ws_k):
    ds = xr.Dataset({
        'TI': 1,
        'P': 1,
        'XYHLK': (['x', 'y', 'h', 'wd', 'ws'], np.random.rand(10, 20, len(h), len(wd), len(ws))),
        'XYHL': (['x', 'y', 'h', 'wd'], np.random.rand(10, 20, len(h), len(wd))),
        'XYHK': (['x', 'y', 'h', 'ws'], np.random.rand(10, 20, len(h), len(ws))),
        'K': (['ws'], np.random.rand(len(ws))),
        'L': (['wd'], np.random.rand(len(wd))),
        'KL': (['wd', 'ws'], np.random.rand(len(wd), len(ws))),
        'XY': (['x', 'y'], np.random.rand(10, 20)),
        'H': (['h'], np.random.rand(len(h))),
        'XYH': (['x', 'y', 'h'], np.random.rand(10, 20, len(h))),
        'XYL': (['x', 'y', 'wd'], np.random.rand(10, 20, len(wd))),
        'XYK': (['x', 'y', 'ws'], np.random.rand(10, 20, len(ws))),
        'I': (['i'], np.random.rand(2)),
        'IL': (['i', 'wd'], np.random.rand(2, len(wd))),
        'IK': (['i', 'ws'], np.random.rand(2, len(ws))),
        'ILK': (['i', 'wd', 'ws'], np.random.rand(2, len(wd), len(ws))),
    },
        coords={'x': np.linspace(0, 100, 10),
                'y': np.linspace(200, 300, 20),
                'h': h,
                'wd': wd,
                'ws': ws,
                'i': [0, 1]}
    )
    site = XRSite(ds)
    lw = LocalWind(x_i=[25, 50], y_i=[225, 250], h_i=h_i, wd=wd_l, ws=ws_k, wd_bin_size=1)
    for n in ['XYHLK', 'XYHL', 'XYHK', 'K', 'L', 'KL', 'XY', 'H', 'XYH', 'XYL', 'XYK', 'I', 'IL', 'IK', 'ILK']:
        ip1 = site.interp(site.ds[n], lw.coords)
        ip2 = ds[n].sel_interp_all(lw.coords)
        npt.assert_array_equal(ip1.shape, ip2.shape)
        npt.assert_array_almost_equal(ip1.data, ip2.data)


def test_interp_special_cases():
    wd = np.arange(5)
    ws = np.arange(10)

    ds = xr.Dataset({
        'TI': (['i', 'wd', 'ws'], np.random.rand(10, len(wd), len(ws))),
        'P': 1
    },
        coords={'i': np.arange(10),
                'wd': wd,
                'ws': ws}
    )
    site = XRSite(ds)
    with pytest.raises(ValueError, match=r"Number of points, i\(=10\), in site data variable, TI, must match "):
        lw = LocalWind(x_i=[25, 50], y_i=[225, 250], h_i=110, wd=wd, ws=ws, wd_bin_size=1)
        site.interp(site.ds.TI, lw.coords)

    x = y = np.arange(10)
    lw = LocalWind(x_i=x, y_i=y, h_i=110, wd=wd, ws=ws, wd_bin_size=1)
    ip1 = site.interp(site.ds.TI, lw.coords)
    ip2 = ds.TI.sel_interp_all(lw.coords)
    npt.assert_array_equal(ip1.shape, ip2.shape)
    npt.assert_array_almost_equal(ip1.data, ip2.data)
