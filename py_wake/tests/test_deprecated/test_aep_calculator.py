from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.site._site import UniformWeibullSite
from py_wake.tests import npt
from py_wake import NOJ, IEA37SimpleBastankhahGaussian
from py_wake.examples.data import hornsrev1
import numpy as np
from py_wake.aep_calculator import AEPCalculator
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
import pytest
import warnings


@pytest.fixture(autouse=True)
def my_fixture():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


def test_aep_no_wake():
    site = UniformWeibullSite([1], [10], [2], .75)
    V80 = hornsrev1.V80()
    aep = AEPCalculator(NOJ(site, V80))
    npt.assert_almost_equal(aep.calculate_AEP([0], [0], ws=np.arange(3, 25)).sum(), 8.260757098, 9)


def test_wake_map():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(site, windTurbines)
    aep = AEPCalculator(wake_model)
    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)
    X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[0], ws=[9])
    m = 49, slice(100, 133, 2)
    # print(np.round(Z[m], 2).tolist()) # ref
    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z)  # , np.arange(2, 10, .01))
        plt.colorbar(c)
        windTurbines.plot(x, y)
        plt.plot(X[m], Y[m], '.-r')
        plt.show()

    ref = [3.27, 3.27, 9.0, 7.46, 7.46, 7.46, 7.46, 7.31, 7.31, 7.31, 7.31, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3]
    npt.assert_array_almost_equal(Z[m], ref, 2)


def test_aep_map():
    site = IEA37Site(16)
    x = [0, 0]
    y = [0, 200]
    windTurbines = IEA37_WindTurbines()
    wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    aep = AEPCalculator(wake_model)
    # print(aep.calculate_AEP([0], [0]).sum())
    x_j = np.arange(-150, 150, 20)
    y_j = np.arange(-250, 250, 20)
    X, Y, Z = aep.aep_map(x_j, y_j, 0, x, y, wd=[0], ws=np.arange(3, 25))
    m = 17
    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z, 100)  # , np.arange(2, 10, .01))
        plt.colorbar(c)
        windTurbines.plot(x, y)
        plt.plot(X[m], Y[m], '.-r')
        plt.show()
    # print(np.round(Z[m], 2).tolist()) # ref
    ref = [21.5, 21.4, 21.02, 20.34, 18.95, 16.54, 13.17, 10.17, 10.17, 13.17, 16.54, 18.95, 20.34, 21.02, 21.4]
    npt.assert_array_almost_equal(Z[m].squeeze('h'), ref, 2)


def test_ti_map():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(site, windTurbines, turbulenceModel=STF2017TurbulenceModel())
    aep = AEPCalculator(wake_model)
    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)
    X, Y, Z = aep.ti_map(x_j, y_j, 110, x, y, wd=[0], ws=[9])
    m = 49, slice(100, 133, 2)

    if 0:
        print(np.round(Z[m], 2).tolist())  # ref
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z, np.arange(.075, .50, .001))
        plt.colorbar(c)
        windTurbines.plot(x, y)
        plt.plot(X[m], Y[m], '.-r')
        plt.show()

    ref = [0.48, 0.08, 0.08, 0.13, 0.16, 0.18, 0.19, 0.19, 0.2, 0.18, 0.17, 0.12, 0.13, 0.13, 0.13, 0.12, 0.12]
    npt.assert_array_almost_equal(Z[m], ref, 2)


def test_aep_map_no_turbines():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    aep = AEPCalculator(wake_model)
    x_j = np.arange(-150, 150, 20)
    y_j = np.arange(-250, 250, 20)
    _, _, Z = aep.aep_map(x_j, y_j, 0, [], [], wd=[0])
    expect = (3.35 * 1e6) * 24 * 365 * (1e-9)

    npt.assert_array_almost_equal(Z, expect, 2)


def test_aep_no_wake_loss_hornsrev():
    wt = hornsrev1.V80()
    x, y = hornsrev1.wt_x, hornsrev1.wt_y
    site = UniformWeibullSite([1], [10], [2], .75)
    site.default_ws = np.arange(3, 25)

    aep = AEPCalculator(NOJ(site, wt))
    aep_nowake = aep.calculate_AEP_no_wake_loss(x, y).sum()
    npt.assert_almost_equal(aep_nowake / 80, 8.260757098)
    cap_factor = aep.calculate_AEP(x, y).sum() / aep_nowake
    # print(cap_factor)
    npt.assert_almost_equal(cap_factor, 0.947175839142014)
