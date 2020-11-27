import numpy as np
import pytest
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake import NOJ, Fuga
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.examples.data.hornsrev1 import HornsrevV80, Hornsrev1Site, wt_x, wt_y, wt9_x, wt9_y
from py_wake.tests.test_files.fuga import LUT_path_2MW_z0_0_03
from py_wake.flow_map import HorizontalGrid
from py_wake.wind_farm_models.engineering_models import All2AllIterative, PropagateDownwind
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.superposition_models import SquaredSum, WeightedSum
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
from py_wake.turbulence_models.stf import STF2005TurbulenceModel
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian, IEA37SimpleBastankhahGaussianDeficit
from numpy import newaxis as na
import matplotlib.pyplot as plt
from py_wake.utils.gradients import autograd, cs, fd, plot_gradients
from py_wake.deficit_models.fuga import FugaDeficit
from py_wake.superposition_models import LinearSum
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.wind_turbines import WindTurbines
WindFarmModel.verbose = False


def test_wake_model():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(site, windTurbines)

    with pytest.raises(ValueError, match="Turbines 0 and 1 are at the same position"):
        wake_model([0, 0], [0, 0], wd=np.arange(0, 360, 22.5), ws=[9.8])


def test_wec():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wake_model = Fuga(LUT_path_2MW_z0_0_03, site, wts)
    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    flow_map_wec1 = wake_model(wt_x, wt_y, 70, wd=[30], ws=[10]).flow_map(HorizontalGrid(x_j, y_j))
    Z_wec1 = flow_map_wec1.WS_eff_xylk[:, :, 0, 0]
    wake_model.wec = 2
    flow_map_wec2 = wake_model(wt_x, wt_y, 70, wd=[30], ws=[10]).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map_wec1.XY
    Z_wec2 = flow_map_wec2.WS_eff_xylk[:, :, 0, 0]

    if 0:
        print(list(np.round(Z_wec1[140, 100:400:10].values, 4)))
        print(list(np.round(Z_wec2[140, 100:400:10].values, 4)))

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
        [10.0467, 10.0473, 10.0699, 10.0093, 9.6786, 7.8589, 6.8539, 9.2199, 9.9837, 10.036, 10.0796,
         10.0469, 10.0439, 9.1866, 7.2552, 9.1518, 10.0449, 10.0261, 10.0353, 9.9256, 9.319, 8.0062,
         6.789, 8.3578, 9.9393, 10.0332, 10.0183, 10.0186, 10.0191, 10.0139], 4)
    npt.assert_array_almost_equal(
        Z_wec2[140, 100:400:10],
        [10.0297, 9.9626, 9.7579, 9.2434, 8.2318, 7.008, 6.7039, 7.7303, 9.0101, 9.6877, 9.9068, 9.7497,
         9.1127, 7.9505, 7.26, 7.9551, 9.2104, 9.7458, 9.6637, 9.1425, 8.2403, 7.1034, 6.5109, 7.2764,
         8.7653, 9.7139, 9.9718, 10.01, 10.0252, 10.0357], 4)


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
    flow_map = wake_model(*site.initial_position[:n_wt].T, wd=[0, 90]).flow_map(HorizontalGrid(resolution=1000))
    # check that deficit matrix > 10MB (i.e. it enters the memory saving loop)
    assert (np.prod(flow_map.WS_eff_xylk.shape) * n_wt * 8 / 1024**2) > 10
    assert flow_map.WS_eff_xylk.shape == (1000, 1000, 2, 1)


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
    wt = WindTurbines.from_WindTurbines([IEA37_WindTurbines(), IEA37_WindTurbines()])

    wfm = NOJ(site, wt)

    sim_res = wfm([0, 500], [0, 0], type=[0, 1], wd=270)

    npt.assert_almost_equal(sim_res.aep(with_wake_loss=False).sum(),
                            2 * wfm([0], [0], wd=270).aep(with_wake_loss=False).sum())


def test_All2AllIterativeDeflection():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wf_model = All2AllIterative(site, windTurbines,
                                wake_deficitModel=NOJDeficit(),
                                superpositionModel=SquaredSum(),
                                deflectionModel=JimenezWakeDeflection())
    wf_model([0], [0])


def test_dAEP_2wt():
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wt = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, wt)
    x, y = iea37_site.initial_position[np.array([0, 2, 5, 8, 14])].T

    # plot 2 wt case
    x, y = np.array([[0, 130 * 4], [0, 0]], dtype=np.float)
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
    plt.close()


def test_dAEPdx():
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wt = IEA37_WindTurbines()
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
