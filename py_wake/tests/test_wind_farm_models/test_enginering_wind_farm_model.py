import numpy as np
import xarray as xr
import pytest
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site, IEA37WindTurbinesDeprecated
from py_wake import NOJ, Fuga, examples
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.examples.data.hornsrev1 import HornsrevV80, Hornsrev1Site, wt_x, wt_y, V80
from py_wake.tests.test_files.fuga import LUT_path_2MW_z0_0_03
from py_wake.flow_map import HorizontalGrid, XYGrid
from py_wake.wind_farm_models.engineering_models import All2AllIterative, PropagateDownwind
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.superposition_models import SquaredSum, WeightedSum
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
from py_wake.turbulence_models.stf import STF2005TurbulenceModel
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian, IEA37SimpleBastankhahGaussianDeficit,\
    BastankhahGaussian
from numpy import newaxis as na
import matplotlib.pyplot as plt
from py_wake.utils.gradients import autograd, cs, fd, plot_gradients
from py_wake.deficit_models.fuga import FugaDeficit
from py_wake.superposition_models import LinearSum
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.wind_turbines_deprecated import DeprecatedOneTypeWindTurbines
import pandas as pd
import os
from py_wake.rotor_avg_models.rotor_avg_model import CGIRotorAvg


WindFarmModel.verbose = False


def test_wake_model():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(site, windTurbines)

    with pytest.raises(ValueError, match="Turbines 0 and 1 are at the same position"):
        wake_model([0, 0], [100, 100], wd=np.arange(0, 360, 22.5), ws=[9.8])


def test_wec():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wfm = BastankhahGaussian(site, wts)
    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    flow_map_wec1 = wfm(wt_x, wt_y, 70, wd=[30], ws=[10]).flow_map(HorizontalGrid(x_j, y_j))
    Z_wec1 = flow_map_wec1.WS_eff_xylk[:, :, 0, 0]
    wfm.wec = 2
    flow_map_wec2 = wfm(wt_x, wt_y, 70, wd=[30], ws=[10]).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map_wec1.XY
    Z_wec2 = flow_map_wec2.WS_eff_xylk[:, :, 0, 0]

    if 0:
        print(list(np.round(Z_wec1[140, 100:400:10].values, 2)))
        print(list(np.round(Z_wec2[140, 100:400:10].values, 2)))

        flow_map_wec1.plot_wake_map(levels=np.arange(6, 10.5, .1), plot_colorbar=False)
        plt.plot(X[0], Y[140])
        wts.plot(wt_x, wt_y)
        plt.figure()
        c = flow_map_wec2.plot_wake_map(levels=np.arange(6, 10.5, .1), plot_colorbar=False)
        plt.colorbar(c)
        plt.plot(X[0], Y[140])
        wts.plot(wt_x, wt_y)

        plt.figure()
        plt.plot(X[0], Z_wec1[140, :], label="Z=70m")
        plt.plot(X[0], Z_wec2[140, :], label="Z=70m")
        plt.plot(X[0, 100:400:10], Z_wec1[140, 100:400:10], '.')
        plt.plot(X[0, 100:400:10], Z_wec2[140, 100:400:10], '.')
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(
        Z_wec1[140, 100:400:10],
        [10.0, 10.0, 10.0, 9.99, 9.8, 6.52, 1.47, 9.44, 9.98, 10.0, 10.0, 10.0, 10.0, 9.05, 0.03, 9.11, 10.0,
         10.0, 10.0, 9.97, 9.25, 7.03, 2.35, 6.51, 9.99, 10.0, 10.0, 10.0, 10.0, 10.0], 2)
    npt.assert_array_almost_equal(
        Z_wec2[140, 100:400:10],
        [9.99, 9.96, 9.84, 9.47, 7.82, 2.24, 0.21, 6.21, 9.22, 9.82, 9.98, 9.92, 9.05, 4.45, 0.01, 4.53, 9.35,
         9.95, 9.75, 9.13, 7.92, 5.14, 0.32, 2.2, 8.38, 9.94, 10.0, 10.0, 10.0, 10.0], 2)


def test_str():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wf_model = All2AllIterative(site, windTurbines,
                                wake_deficitModel=NOJDeficit(),
                                superpositionModel=SquaredSum(),
                                blockage_deficitModel=SelfSimilarityDeficit(),
                                deflectionModel=JimenezWakeDeflection(),
                                turbulenceModel=STF2005TurbulenceModel())
    assert str(wf_model) == "All2AllIterative(EngineeringWindFarmModel, NOJDeficit-wake, SelfSimilarityDeficit-blockage, RotorCenter-rotor-average, SquaredSum-superposition, JimenezWakeDeflection-deflection, STF2005TurbulenceModel-turbulence)"


@pytest.mark.parametrize('wake_deficitModel,deflectionModel,superpositionModel',
                         [(NOJDeficit(), None, SquaredSum()),
                          (IEA37SimpleBastankhahGaussianDeficit(), JimenezWakeDeflection(), WeightedSum())])
def test_huge_flow_map(wake_deficitModel, deflectionModel, superpositionModel):
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wake_model = PropagateDownwind(site, windTurbines, wake_deficitModel=wake_deficitModel,
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=STF2005TurbulenceModel())
    n_wt = 2
    flow_map = wake_model(*site.initial_position[:n_wt].T, wd=[0]).flow_map(HorizontalGrid(resolution=810))
    # check that deficit matrix > 10MB (i.e. it enters the memory saving loop)
    assert (np.prod(flow_map.WS_eff_xylk.shape) * n_wt * 8 / 1024**2) > 10
    assert flow_map.WS_eff_xylk.shape == (810, 810, 1, 1)


def test_aep():
    site = UniformSite([1], ti=0)
    windTurbines = IEA37_WindTurbines()
    wfm = NOJ(site, windTurbines)

    sim_res = wfm([0], [0], wd=270)

    npt.assert_almost_equal(sim_res.aep().sum(), 3.35 * 24 * 365 / 1000)
    npt.assert_almost_equal(sim_res.aep(normalize_probabilities=True).sum(), 3.35 * 24 * 365 / 1000)

    npt.assert_equal(sim_res.aep().data.sum(), wfm.aep([0], [0], wd=270))
    npt.assert_almost_equal(sim_res.aep(normalize_probabilities=True).sum(),
                            wfm.aep([0], [0], wd=270, normalize_probabilities=True))
    npt.assert_almost_equal(sim_res.aep(with_wake_loss=False).sum(), wfm.aep([0], [0], wd=270, with_wake_loss=False))


def test_two_wt_aep():
    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(site, windTurbines)
    sim_res1 = wake_model([0], [0], wd=270)
    sim_res2 = wake_model([0, 0], [0, 500], wd=270)

    # one wt, wind from west ~ 5845 hours of full load
    npt.assert_almost_equal(sim_res1.aep(normalize_probabilities=True).sum(), 3.35 * 5.845, 2)

    # No wake, two wt = 2 x one wt
    npt.assert_almost_equal(sim_res1.aep().sum() * 2, sim_res2.aep().sum())

    # same for normalized propabilities
    npt.assert_almost_equal(sim_res1.aep(normalize_probabilities=True).sum() * 2,
                            sim_res2.aep(normalize_probabilities=True).sum())


def test_aep_mixed_type():
    site = UniformSite([1], ti=0)
    wt = WindTurbines.from_WindTurbine_lst([IEA37_WindTurbines(), IEA37_WindTurbines()])

    wfm = NOJ(site, wt)

    sim_res = wfm([0, 500], [0, 0], type=[0, 1], wd=270)

    npt.assert_almost_equal(sim_res.aep(with_wake_loss=False).sum(),
                            2 * wfm([0], [0], wd=270).aep(with_wake_loss=False).sum())


@pytest.mark.parametrize('deflection_model,count',
                         [(None, 1),
                          (JimenezWakeDeflection(), 4)])
def test_All2AllIterativeDeflection(deflection_model, count):

    class FugaDeficitCount(FugaDeficit):
        counter = 0

        def _calc_layout_terms(self, dw_ijlk, hcw_ijlk, h_il, dh_ijlk, D_src_il, **_):
            self.counter += 1
            return FugaDeficit._calc_layout_terms(self, dw_ijlk, hcw_ijlk, h_il, dh_ijlk, D_src_il, **_)

    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    deficit_model = FugaDeficitCount()
    wf_model = All2AllIterative(site, windTurbines,
                                wake_deficitModel=deficit_model,
                                superpositionModel=LinearSum(),
                                blockage_deficitModel=SelfSimilarityDeficit(),
                                rotorAvgModel=CGIRotorAvg(4),
                                deflectionModel=deflection_model, convergence_tolerance=None)
    sim_res = wf_model([0, 500, 1000, 1500], [0, 0, 0, 0], wd=270, ws=10, yaw=[30, -30, 30, -30])
    assert wf_model.wake_deficitModel.counter == count
    if 0:
        sim_res.flow_map(XYGrid(x=np.linspace(-200, 2000, 100))).plot_wake_map()
        plt.show()


def test_dAEP_2wt():
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wsp_cut_in = 4
    wsp_cut_out = 25
    wsp_rated = 9.8
    power_rated = 3350000
    constant_ct = 8 / 9

    def ct(wsp):
        wsp = np.asarray(wsp)
        ct = np.zeros_like(wsp, dtype=float)
        ct[(wsp >= wsp_cut_in) & (wsp <= wsp_cut_out)] = constant_ct
        return ct

    def power(wsp):
        wsp = np.asarray(wsp)
        power = np.where((wsp > wsp_cut_in) & (wsp <= wsp_cut_out),
                         np.minimum(power_rated * ((wsp - wsp_cut_in) / (wsp_rated - wsp_cut_in))**3, power_rated), 0)

        return power

    def dpower(wsp):
        return np.where((wsp > wsp_cut_in) & (wsp <= wsp_rated),
                        3 * power_rated * (wsp - wsp_cut_in)**2 / (wsp_rated - wsp_cut_in)**3,
                        0)

    def dct(wsp):
        return wsp * 0  # constant ct

    wt = DeprecatedOneTypeWindTurbines(name='test', diameter=130, hub_height=110,
                                       ct_func=ct, power_func=power, power_unit='w')
    wt.set_gradient_funcs(dpower, dct)
    wfm = IEA37SimpleBastankhahGaussian(site, wt)
    x, y = iea37_site.initial_position[np.array([0, 2, 5, 8, 14])].T

    # plot 2 wt case
    x, y = np.array([[0, 130 * 4], [0, 0]], dtype=float)
    x_lst = np.array([0., 1.]) * np.arange(1, 600, 10)[:, na]
    kwargs = {'ws': [10], 'wd': [270]}

    _, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.plot(x_lst[:, 1], [wfm.aep(x_, y, **kwargs) for x_ in x_lst])
    ax1.set_xlabel('Downwind distance [m]')
    ax1.set_ylabel('AEP [GWh]')

    x_ = x_lst[20]
    ax1.set_title("Center line")
    for grad in [fd, cs, autograd]:
        dAEPdx = wfm.dAEPdn(0, grad)(x_, y, **kwargs)[1]
        npt.assert_almost_equal(dAEPdx / 360, 3.976975605364392e-06, (10, 5)[grad == fd])
        plot_gradients(wfm.aep(x_, y, **kwargs), dAEPdx, x_[1], grad.__name__, step=100, ax=ax1)
    y_lst = np.array([0, 1.]) * np.arange(-100, 100, 5)[:, na]
    ax2.plot(y_lst[:, 1], [wfm.aep(x, y_, **kwargs) for y_ in y_lst])
    ax2.set_xlabel('Crosswind distance [m]')
    ax2.set_ylabel('AEP [GWh]')
    y_ = y_lst[25]
    ax2.set_title("%d m downstream" % x[1])
    for grad in [fd, cs, autograd]:
        dAEPdy = wfm.dAEPdn(1, grad)(x, y_, **kwargs)[1]
        plot_gradients(wfm.aep(x, y_, **kwargs), dAEPdy, y_[1], grad.__name__, step=50, ax=ax2)
        npt.assert_almost_equal(dAEPdy / 360, 3.794435973860448e-05, (10, 5)[grad == fd])

    if 0:
        plt.legend()
        plt.show()
    plt.close('all')


def test_dAEPdx():
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wt = IEA37_WindTurbines()
    wt.enable_autograd()
    wfm = IEA37SimpleBastankhahGaussian(site, wt)
    x, y = iea37_site.initial_position[np.array([0, 2, 5, 8, 14])].T

    dAEPdxy_autograd = wfm.dAEPdxy(gradient_method=autograd)(x, y)
    dAEPdxy_cs = wfm.dAEPdxy(gradient_method=cs)(x, y)
    dAEPdxy_fd = wfm.dAEPdxy(gradient_method=fd)(x, y)

    npt.assert_array_almost_equal(dAEPdxy_autograd, dAEPdxy_cs, 15)
    npt.assert_array_almost_equal(dAEPdxy_autograd, dAEPdxy_fd, 6)


@pytest.mark.parametrize('wake_deficitModel,blockage_deficitModel', [(FugaDeficit(), None),
                                                                     (NoWakeDeficit(), SelfSimilarityDeficit()),
                                                                     (FugaDeficit(), SelfSimilarityDeficit())])
def test_deficit_symmetry(wake_deficitModel, blockage_deficitModel):
    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()

    wfm = All2AllIterative(site, windTurbines, wake_deficitModel=wake_deficitModel,
                           superpositionModel=LinearSum(),
                           blockage_deficitModel=blockage_deficitModel,
                           deflectionModel=None, turbulenceModel=None)

    power = wfm([0, 0, 500, 500], [0, 500, 0, 500], wd=[0], ws=[8]).power_ilk[:, 0, 0]
    npt.assert_array_almost_equal(power[:2], power[2:])


def test_double_wind_farm_model():
    """Check that a new wind farm model does not change results of previous"""
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit())
    aep_ref = wfm(x, y).aep().sum()
    PropagateDownwind(site, windTurbines, wake_deficitModel=NoWakeDeficit())
    aep = wfm(x, y).aep().sum()
    npt.assert_array_equal(aep, aep_ref)


def test_double_wind_farm_model_All2AllIterative():
    """Check that a new wind farm model does not change results of previous"""
    site = IEA37Site(64)
    x, y = site.initial_position.T
    x, y = wt_x, wt_y
    windTurbines = IEA37_WindTurbines()
    wfm = All2AllIterative(site, windTurbines, wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit())
    aep_ref = wfm(x, y).aep().sum()
    All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit())(x, y)
    aep = wfm(x, y).aep().sum()
    npt.assert_array_equal(aep, aep_ref)


def test_huge_farm():
    site = UniformSite([1], ti=0)
    windTurbines = IEA37_WindTurbines()
    wfm = PropagateDownwind(site, windTurbines, NoWakeDeficit())
    N = 200
    x = np.arange(N) * windTurbines.diameter(0) * 4

    import tracemalloc
    tracemalloc.start()
    wfm(x, x * 0, ws=10)
    current, peak = tracemalloc.get_traced_memory()  # @UnusedVariable
    peak /= 1024**2
    assert peak < 800
    tracemalloc.stop()


def test_time_series_values():
    wt = V80()
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = NOJ(site, wt)
    wd = np.arange(350, 360)
    ws = np.arange(5, 10)
    wd_t, ws_t = [v.flatten() for v in np.meshgrid(wd, ws)]
    sim_res_t = wfm(x, y, ws=ws_t, wd=wd_t, time=True, verbose=False)
    sim_res = wfm(x, y, wd=wd, ws=ws)

    for k in ['WS_eff', 'TI_eff', 'Power', 'CT']:
        npt.assert_array_equal(np.moveaxis(sim_res_t[k].values.reshape((80, 5, 10)), 1, 2), sim_res[k].values)


def test_time_series_dates():
    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws, ws_std = [d[k][:6 * 24] for k in ['wd', 'ws', 'ws_std']]
    ti = np.minimum(ws_std / ws, .5)
    t = pd.date_range("2000-01-01", freq="10T", periods=24 * 6)
    wt = V80()
    site = Hornsrev1Site()
    site.ds['TI'] = xr.DataArray(ti, [('time', t)])

    x, y = site.initial_position.T
    wfm = NOJ(site, wt)
    sim_res = wfm(x, y, ws=ws, wd=wd, time=t, verbose=False)
    npt.assert_array_equal(sim_res.WS, ws)
    npt.assert_array_equal(sim_res.WD, wd)
    npt.assert_array_equal(sim_res.time, t)


def test_time_series_override_WS():
    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws = [d[k][:6 * 24] for k in ['wd', 'ws']]
    t = pd.date_range("2000-01-01", freq="10T", periods=24 * 6)
    WS_it = (np.arange(80) / 100)[:, na] + ws[na]
    wt = V80()
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = NOJ(site, wt)
    sim_res = wfm(x, y, ws=ws, wd=wd, time=t, WS=WS_it, verbose=False)
    npt.assert_array_equal(sim_res.WS, WS_it)
    npt.assert_array_equal(sim_res.WD, wd)
    npt.assert_array_equal(sim_res.time, t)


def test_time_series_override_WD():
    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws = [d[k][:6 * 24] for k in ['wd', 'ws']]
    t = pd.date_range("2000-01-01", freq="10T", periods=24 * 6)
    WD_it = (np.arange(80) / 100)[:, na] + wd[na]
    wt = V80()
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = NOJ(site, wt)
    sim_res = wfm(x, y, ws=ws, wd=wd, time=t, WD=WD_it, verbose=False)
    npt.assert_array_equal(sim_res.WS, ws)
    npt.assert_array_equal(sim_res.WD, WD_it)
    npt.assert_array_equal(sim_res.time, t)


def test_time_series_override_TI():

    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws, ws_std = [d[k][:6 * 24] for k in ['wd', 'ws', 'ws_std']]
    ti = np.minimum(ws_std / ws, .5)
    t = pd.date_range("2000-01-01", freq="10T", periods=24 * 6)
    wt = V80()
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = NOJ(site, wt)
    sim_res = wfm(x, y, ws=ws, wd=wd, time=t, TI=ti, verbose=False)
    npt.assert_array_equal(sim_res.WS, ws)
    npt.assert_array_equal(sim_res.WD, wd)
    npt.assert_array_equal(sim_res.time, t)
    npt.assert_array_equal(sim_res.TI[0], ti)


def test_time_series_aep():

    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws = [d[k][::100] for k in ['wd', 'ws']]
    wt = V80()
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = NOJ(site, wt)
    sim_res = wfm(x, y, ws=ws, wd=wd, time=True, verbose=False)
    npt.assert_allclose(sim_res.aep().sum(), 545, atol=1)


def test_time_series_operating():
    from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular
    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws, ws_std = [d[k][:6 * 24] for k in ['wd', 'ws', 'ws_std']]
    ws += 3
    t = np.arange(6 * 24)
    wt = V80()
    site = Hornsrev1Site()

    # replace powerCtFunction
    wt.powerCtFunction = PowerCtFunctionList(
        key='operating',
        powerCtFunction_lst=[PowerCtTabular(ws=[0, 100], power=[0, 0], power_unit='w', ct=[0, 0]),  # 0=No power and ct
                             wt.powerCtFunction],  # 1=Normal operation
        default_value=1)
    wfm = NOJ(site, wt)
    x, y = site.initial_position.T
    operating = (t < 48) | (t > 72)
    sim_res = wfm(x, y, ws=ws, wd=wd, time=t, operating=operating)
    npt.assert_array_equal(sim_res.operating[0], operating)
    npt.assert_array_equal(sim_res.Power[:, operating == 0], 0)
    npt.assert_array_equal(sim_res.Power[:, operating != 0] > 0, True)

    operating = np.ones((80, 6 * 24))
    operating[1] = (t < 48) | (t > 72)
    sim_res = wfm(x, y, ws=ws, wd=wd, time=t, operating=operating)
    npt.assert_array_equal(sim_res.operating, operating)
    npt.assert_array_equal(sim_res.Power.values[operating == 0], 0)
    npt.assert_array_equal(sim_res.Power.values[operating != 0] > 0, True)


def test_time_series_operating_wrong_shape():
    from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular
    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws, ws_std = [d[k][:6 * 24] for k in ['wd', 'ws', 'ws_std']]
    ws += 3
    t = np.arange(6 * 24)
    wt = V80()
    site = Hornsrev1Site()

    # replace powerCtFunction
    wt.powerCtFunction = PowerCtFunctionList(
        key='operating',
        powerCtFunction_lst=[PowerCtTabular(ws=[0, 100], power=[0, 0], power_unit='w', ct=[0, 0]),  # 0=No power and ct
                             wt.powerCtFunction],  # 1=Normal operation
        default_value=1)
    wfm = NOJ(site, wt)
    x, y = site.initial_position.T
    operating = (t < 48) | (t > 72)
    with pytest.raises(ValueError, match=r"Argument, operating\(shape=\(1, 144\)\), has unsupported shape."):
        wfm(x, y, ws=ws, wd=wd, time=t, operating=[operating])


def test_aep_wind_atlas_method():
    site = Hornsrev1Site()

    wt = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, wt)
    x, y = [0], [0]
    wd = np.arange(360)
    aep_lps = wfm(x, y, wd=wd, ws=np.arange(3, 27)).aep(linear_power_segments=True)
    aep = wfm(x, y, wd=wd, ws=np.r_[3, np.arange(3.5, 27)]).aep()
    if 0:
        plt.plot(aep_lps.ws, np.cumsum(aep_lps.sum(['wt', 'wd'])), '.-', label='Linear power segments')
        plt.plot(aep.ws, np.cumsum(aep.sum(['wt', 'wd'])), '.-', label='Constant power segments')
        plt.ylabel('Cumulated AEP [GWh]')
        plt.xlabel('Wind speed [m/s]')
        plt.legend()
        plt.show()
    npt.assert_almost_equal(aep_lps.sum(), 16.73490444)
    npt.assert_almost_equal(aep.sum(), 16.69320343)
