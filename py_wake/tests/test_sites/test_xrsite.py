import numpy as np
from py_wake.site.shear import PowerShear
from py_wake.site.xrsite import XRSite, GlobalWindAtlasSite
import xarray as xr
from py_wake.tests import npt
import pytest
import matplotlib.pyplot as plt
from numpy import newaxis as na
from py_wake.tests.test_files import tfp
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
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
import warnings
from urllib.error import HTTPError
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.utils.gradients import fd, autograd, cs


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


@pytest.fixture
def pywasp_pwc_point():
    pwc_file = tfp + "pwc_parquo_fictio_small.nc"
    return xr.open_dataset(pwc_file)


@pytest.fixture
def pywasp_pwc_cuboid():
    pwc_file = tfp + "pwc_cuboid_small.nc"
    return xr.open_dataset(pwc_file)


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


def test_complex_fixed_pos_flow_map(complex_fixed_pos_site):
    site = complex_fixed_pos_site
    x_i, y_i = site.initial_position.T
    sim_res = NOJ(site, V80())(x_i, y_i, wd=270, ws=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim_res.flow_map(XYGrid(resolution=3))


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


@pytest.mark.parametrize('k', ['WD', 'Turning'])
def test_turning_mean(complex_grid_site, k):

    ds = xr.Dataset(
        data_vars={k: (['x', 'y'], [[350, 10], [-30, 30]]),
                   'P': 1, 'TI': .1},
        coords={'x': [0, 500], 'y': [0, 400], 'wd': np.linspace(0, 360, len(f), endpoint=False)})
    site = XRSite(ds)

    wt = V80()
    wfm = NOJ(site, wt)
    sim_res = wfm([], [], wd=0, ws=10)
    # print(sim_res.Power)
    s = 100
    WD = sim_res.flow_map(XYGrid(x=[0, 500], y=np.arange(0, 400 + s, s))).WD.squeeze()

    if 0:
        for wd in WD.T:
            ((wd + 180) % 360 - 180).plot()
        plt.plot([0, 400], [-10, 10])
        plt.plot([0, 400], [-30, 30])
        plt.show()

    npt.assert_array_almost_equal(WD, [[350., 330.],
                                       [355, 345],
                                       [0., 0.],
                                       [5, 15],
                                       [10., 30.]])


def test_GlobalWindAtlasSite():
    ref = Hornsrev1Site()
    lat, long = 55.52972, 7.906111  # hornsrev

    try:
        site = GlobalWindAtlasSite(lat, long, height=70, roughness=0.001, ti=0.075)
    except HTTPError:
        pytest.xfail('HTTPError in GlobalWindAtlasSite')
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


def test_from_pywasp_pwc_point(pywasp_pwc_point):

    site = XRSite.from_pywasp_pwc(pywasp_pwc_point)

    A = np.array([
        [5.757, 6.088, 5.369],
        [5.584, 5.806, 5.358],
        [7.506, 7.707, 7.098],
        [9.383, 10.082, 11.018],
        [8.835, 9.644, 10.244],
        [6.464, 7.071, 7.066],
        [6.265, 6.626, 5.846],
        [7.562, 7.858, 7.289],
        [10.005, 10.42, 10.124],
        [10.046, 11.211, 12.218],
        [8.821, 9.624, 10.19],
        [5.948, 6.504, 6.499]
    ])

    k = np.array([
        [1.912109, 1.919922, 1.990234],
        [2.162109, 2.166016, 2.169922],
        [2.638672, 2.626953, 2.771484],
        [3.033203, 3.044922, 3.044922],
        [2.884766, 2.880859, 2.697266],
        [2.666016, 2.666016, 2.666016],
        [2.513672, 2.501953, 2.470703],
        [2.529297, 2.548828, 2.509766],
        [2.533203, 2.537109, 2.533203],
        [2.310547, 2.333984, 2.337891],
        [1.986328, 1.986328, 1.900391],
        [1.767578, 1.767578, 1.767578]
    ])

    wdfreq = np.array([
        [0.0544071, 0.0532056, 0.04288784],
        [0.0376036, 0.0355262, 0.03168268],
        [0.05400841, 0.05099307, 0.04538284],
        [0.08345144, 0.08324289, 0.09760682],
        [0.09817267, 0.102156, 0.1072249],
        [0.06296581, 0.06525171, 0.06451698],
        [0.04901042, 0.04791974, 0.03865756],
        [0.06785324, 0.0656789, 0.05674313],
        [0.1384152, 0.1309593, 0.1174582],
        [0.1622368, 0.1655659, 0.1941505],
        [0.1219364, 0.1269784, 0.1319477],
        [0.06993906, 0.07252219, 0.0717409]
    ])

    speedups = np.array([
        [0.94574503, 0.96176405, 0.97405529, 0.85145872, 0.86463324,
         0.91415641, 0.94562806, 0.962134, 0.9601336, 0.82210133,
         0.86466306, 0.91451415, 0.94574503],
        [1., 1., 1., 0.9150481, 0.94375361,
         1., 1., 1., 1., 0.91755896,
         0.94337573, 1., 1.],
        [0.8811313, 0.92283872, 0.92266141, 1., 1.,
         0.99929289, 0.88201829, 0.92721526, 0.97155348, 1.,
         1., 0.99923124, 0.8811313]
    ])

    sector = np.linspace(0.0, 330.0, 12)

    x = np.array([263655.0, 263891.1, 264022.2])
    y = np.array([6506601.0, 6506394.0, 6506124.0])
    h = np.array([70] * 3)

    wd = np.linspace(0.0, 360.0, 13)
    i = np.arange(3)

    assert "i" in site.ds.dims

    npt.assert_array_equal(site.ds.Weibull_A.values[..., :12], A.T)
    npt.assert_array_equal(site.ds.Weibull_k.values[..., :12], k.T)
    npt.assert_array_equal(site.ds.Sector_frequency.values[..., :12], wdfreq.T)
    npt.assert_array_equal(site.ds.x.values, x)
    npt.assert_array_equal(site.ds.y.values, y)
    npt.assert_array_equal(site.ds.h.values, h)
    npt.assert_array_equal(site.ds.i.values, i)
    npt.assert_array_equal(site.ds.wd.values, wd)
    npt.assert_allclose(site.ds.Speedup.values, speedups, rtol=1e-8)


def test_from_pywasp_pwc_cuboid(pywasp_pwc_cuboid):

    site = XRSite.from_pywasp_pwc(pywasp_pwc_cuboid)

    assert all(d in site.ds.dims for d in ["x", "y", "h"])
    assert site.ds.wd.size == 13
    npt.assert_almost_equal(site.ds.Weibull_A.values[0, 0, 0, 0], 6.208063)
    npt.assert_almost_equal(site.ds.Weibull_k.values[1, 1, 0, 1], 2.4550781)
    npt.assert_almost_equal(site.ds.Sector_frequency.values[0, 0, 0, 0], 0.05975334)


def test_gradients():
    site = ParqueFicticioSite()
    x, y = site.initial_position[0]

    data_vars = ['WS', 'WD', 'ws_lower', 'ws_upper', 'Weibull_A', 'Weibull_k', 'Sector_frequency', 'P', 'TI']
    for data_var in data_vars[1:]:
        def t(x):
            return site.local_wind(x, y, h_i=[100], wd=270, ws=10)[data_var].values
        ddx_lst = [grad(t)(x) for grad in [fd, cs, autograd]]
        npt.assert_allclose(ddx_lst[0], ddx_lst[1], rtol=1e-4)
        npt.assert_allclose(ddx_lst[1], ddx_lst[2])
