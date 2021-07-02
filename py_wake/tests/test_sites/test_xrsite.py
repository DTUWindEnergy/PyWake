import numpy as np
from py_wake.site.shear import PowerShear
from py_wake.site.xrsite import XRSite, GlobalWindAtlasSite
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
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.tests.check_speed import timeit
from py_wake.site._site import LocalWind
from py_wake.utils import weibull
from py_wake.deficit_models.noj import NOJ, NOJDeficit
from py_wake.flow_map import XYGrid, Points


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


@pytest.fixture
def complex_ws_site():
    ti = 0.1
    P = weibull.cdf(np.array([3, 5, 7, 9, 11, 13]), 10, 2) - weibull.cdf(np.array([0, 3, 5, 7, 9, 11]), 10, 2)
    ds = xr.Dataset(
        data_vars={'Speedup': (['ws'], np.arange(.8, 1.4, .1)),
                   'P': (('wd', 'ws'), P[na, :] * np.array(f)[:, na]), 'TI': ti},
        coords={'ws': [1.5, 4, 6, 8, 10, 12], 'wd': np.linspace(0, 360, len(f), endpoint=False)})
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


@pytest.mark.parametrize('wfm', [PropagateDownwind, All2AllIterative])
def test_turning_mean(complex_grid_site, wfm):

    ds = xr.Dataset(
        data_vars={'Turning': (['x', 'y'], np.arange(-2, 4, 1).reshape((2, 3)).T),
                   'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': .1},
        coords={'x': [0, 500, 1000], 'y': [0, 500], 'wd': np.linspace(0, 360, len(f), endpoint=False)})
    site = XRSite(ds)

    wt = V80()
    wfm = wfm(site, wt, NOJDeficit())
    sim_res = wfm([500, 500], [100, 400], wd=0, ws=10)
    print(sim_res.Power)
    if 0:
        sim_res.flow_map(XYGrid(y=np.linspace(0, 500, 100))).plot_wake_map()
        plt.show()
    assert sim_res.WS_eff.sel(wt=0).item() < sim_res.WS_eff.sel(wt=1).item()
    fm = sim_res.flow_map(Points([500, 500], [100, 400], [70, 70]))
    assert fm.WS_eff.sel(i=0).item() < fm.WS_eff.sel(i=1).item()


def test_GlobalWindAtlasSite():
    ref = Hornsrev1Site()
    lat, long = 55.52972, 7.906111  # hornsrev

    site = GlobalWindAtlasSite(lat, long, height=70, roughness=0.001, ti=0.075)
    ref_mean = weibull.mean(ref.ds.Weibull_A, ref.ds.Weibull_k)
    gwa_mean = weibull.mean(site.ds.Weibull_A, site.ds.Weibull_k)

    if 0:
        plt.figure()
        plt.plot(ref.ds.wd, ref_mean, label='HornsrevSite')
        plt.plot(site.ds.wd, gwa_mean, label='HornsrevSite')
        for r in [0, 1.5]:
            for h in [10, 200]:
                A, k = [site.gwc_ds[v].sel(roughness=r, height=h) for v in ['Weibull_A', 'Weibull_k']]
                plt.plot(site.gwc_ds.wd, weibull.mean(A, k), label=f'{h}, {r}')
        plt.legend()

        plt.show()

    npt.assert_allclose(gwa_mean, ref_mean, atol=1.4)
    for var, atol in [('Sector_frequency', 0.03), ('Weibull_A', 1.6), ('Weibull_k', 0.4)]:
        npt.assert_allclose(site.ds[var], ref.ds[var], atol=atol)


def test_wrong_height():
    ti = 0.1
    ds = xr.Dataset(
        data_vars={'Speedup': (['x', 'y', 'h'], np.arange(.8, 1.4, .1).reshape((3, 2, 1))),
                   'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ti},
        coords={'x': [0, 5, 10], 'y': [0, 5], 'h': [100], 'wd': np.linspace(0, 360, len(f), endpoint=False)})
    site = XRSite(ds, shear=PowerShear(h_ref=100, alpha=.2), interp_method='linear')

    y = np.arange(5)
    x = y * 2
    X, Y = np.meshgrid(x, y)
    x_i, y_i = X.flatten(), Y.flatten()

    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    lw = site.local_wind(x_i=x_i, y_i=y_i, h_i=100, wd=wdir_lst, ws=wsp_lst)


def test_wd_independent_site():
    ti = 0.1
    ds = xr.Dataset(
        data_vars={
            'WS': 10, 'Sector_frequency': 1,
            'Weibull_A': 4, 'Weibull_k': 2, 'TI': ti},
        coords={})
    site = XRSite(ds, shear=None)
    npt.assert_equal(site.ds.sector_width, 360)


def test_i_dependent_WS():
    ds = xr.Dataset(
        data_vars={'WS': ('i', [8, 9, 10]), 'P': ('wd', f)},
        coords={'wd': np.linspace(0, 360, len(f), endpoint=False)})
    site = XRSite(ds)
    lw = site.local_wind([0, 200, 400], [0, 0, 0], [70, 70, 70], wd=0, ws=10)
    npt.assert_array_equal(lw.WS, [8, 9, 10])

    WS = np.arange(6).reshape(3, 2) + 9
    ds = xr.Dataset(
        data_vars={'WS': (('i', 'ws'), WS), 'Sector_frequency': ('wd', f),
                   'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k)},
        coords={'wd': np.linspace(0, 360, len(f), endpoint=False), 'ws': [9, 10], 'i': [0, 1, 2]})
    site = XRSite(ds)
    lw = site.local_wind([0, 200, 400], [0, 0, 0], [70, 70, 70], wd=0, ws=10)
    npt.assert_array_equal(lw.WS.squeeze(), [10, 12, 14])


def test_i_time_dependent_WS():
    t = np.arange(4)
    WS_it = t[na] / 10 + np.array([9, 10])[:, na]
    ds = xr.Dataset(
        data_vars={'WS': (('i', 'time'), WS_it), 'P': ('wd', f), 'TI': 0.1},
        coords={'wd': np.linspace(0, 360, len(f), endpoint=False)})
    site = XRSite(ds)
    wfm = NOJ(site, V80())
    sim_res = wfm([0, 200], [0, 0], ws=WS_it.mean(0), wd=np.zeros(4), time=t)
    npt.assert_array_equal(sim_res.WS, WS_it)


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


def test_plot_wd_distribution(complex_ws_site):
    res = complex_ws_site.plot_wd_distribution()
    ref = [0.0312, 0.0332, 0.043, 0.0568, 0.0648, 0.0567, 0.0718, 0.0967, 0.1199, 0.1154, 0.0809, 0.045]
    if 0:
        print(np.round(res.squeeze().values, 4).tolist())
        plt.show()
    plt.close('all')
    npt.assert_array_almost_equal(res.squeeze(), ref, 4)


def test_local_wind_P_diff_ws(complex_ws_site):
    with pytest.raises(ValueError, match='Cannot interpolate ws-dependent P to other range of ws'):
        complex_ws_site.local_wind([0], [0], 100, wd=np.arange(360), ws=9)


def test_local_wind_P_same_ws(complex_ws_site):
    complex_ws_site.local_wind([0], [0], 100, wd=np.arange(360), ws=10)
    complex_ws_site.local_wind([0], [0], 100, wd=np.arange(360), ws=[8, 10])


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

    windTurbines = WindTurbines.from_WindTurbine_lst([IEA37_WindTurbines(), IEA37_WindTurbines()])
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
    ([110], range(5, 25), [9, 10, 11], 110, [11.5, 12.5], [9.8, 10.2]),
    ([100, 110, 120], range(360), [9.8], 110, range(360), [9.8]),
    ([100, 110, 120], range(5, 25), [9.8], 110, range(10, 20), [9.8]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, range(10, 20), [10]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, range(10, 20), [10, 11]),
    ([100, 110, 120], range(5, 25), [8, 10, 11], 110, range(10, 20), [9.8]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, range(10, 20), [9.8]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, range(10, 20), [9.8, 10.2]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, [11.5, 12.5], [10]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, [11.5, 12.5], [10]),
    ([100, 110, 120], range(5, 25), [9, 10, 11], 110, [11.5, 12.5], [9.8, 10.2])
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
    lw = LocalWind(x_i=[25, 50], y_i=[225, 250], h_i=h_i, wd=wd_l, ws=ws_k, time=False, wd_bin_size=1)

    for n in ['XYHLK', 'XYHL', 'XYHK', 'K', 'L', 'KL', 'XY', 'H', 'XYH', 'XYL', 'XYK', 'I', 'IL', 'IK', 'ILK']:
        ip1 = site.interp(site.ds[n], lw.coords)
        ip2 = ds[n].sel_interp_all(lw.coords)
        npt.assert_array_equal(ip1.shape, ip2.shape)
        if not np.isnan(ip2).sum():
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
        lw = LocalWind(x_i=[25, 50], y_i=[225, 250], h_i=110, wd=wd, ws=ws, time=False, wd_bin_size=1)
        site.interp(site.ds.TI, lw.coords)

    x = y = np.arange(10)
    lw = LocalWind(x_i=x, y_i=y, h_i=110, wd=wd, ws=ws, time=False, wd_bin_size=1)
    ip1 = site.interp(site.ds.TI, lw.coords)
    ip2 = ds.TI.sel_interp_all(lw.coords)
    npt.assert_array_equal(ip1.shape, ip2.shape)
    npt.assert_array_almost_equal(ip1.data, ip2.data)
