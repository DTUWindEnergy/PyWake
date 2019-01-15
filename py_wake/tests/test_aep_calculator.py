from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windrose,\
    read_iea37_windfarm
from py_wake.site._site import UniformSite, UniformWeibullSite
from py_wake.tests import npt
from py_wake.wake_models.noj import NOJ
from py_wake.examples.data import hornsrev1
from py_wake.wake_models.gaussian import IEA37SimpleBastankhahGaussian
import numpy as np
from py_wake.aep_calculator import AEPCalculator


def test_aep_no_wake():
    site = UniformWeibullSite([1], [10], [2], .75)
    V80 = hornsrev1.V80()
    aep = AEPCalculator(site, V80, NOJ(V80))
    npt.assert_almost_equal(aep.calculate_AEP([0], [0], ws=np.arange(3, 25)).sum(), 8.260757098, 9)


def test_wake_map():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(windTurbines)
    aep = AEPCalculator(site, windTurbines, wake_model)
    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)
    X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[0], ws=[9])

    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z)  # , np.arange(2, 10, .01))
        plt.colorbar(c)
        windTurbines.plot(x, y)
        plt.show()

    ref = [3.27, 3.27, 9.0, 7.46, 7.46, 7.46, 7.46, 7.31, 7.31, 7.31, 7.31, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3]
    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


def test_aep_map():
    site = IEA37Site(16)
    x = [0, 0]
    y = [0, 200]
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
    wake_model = IEA37SimpleBastankhahGaussian(windTurbines)
    aep = AEPCalculator(site, windTurbines, wake_model)
    # print(aep.calculate_AEP([0], [0]).sum())
    x_j = np.arange(-150, 150, 20)
    y_j = np.arange(-250, 250, 20)
    X, Y, Z = aep.aep_map(x_j, y_j, 0, x, y, wd=[0], ws=np.arange(3, 25))
    # print(y_j)
    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z, 100)  # , np.arange(2, 10, .01))
        plt.colorbar(c)
        windTurbines.plot(x, y)
        plt.show()
    # print(np.round(Z[17], 2).tolist())
    ref = [21.5, 21.4, 21.02, 20.34, 18.95, 16.54, 13.17, 10.17, 10.17, 13.17, 16.54, 18.95, 20.34, 21.02, 21.4]
    npt.assert_array_almost_equal(Z[17], ref, 2)


def test_aep_map_no_turbines():

    _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
    # n_wt = 16
    # x, y, _ = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)

    site = UniformSite(freq, ti=0.75)
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
    wake_model = IEA37SimpleBastankhahGaussian(windTurbines)
    aep = AEPCalculator(site, windTurbines, wake_model)
    x_j = np.arange(-150, 150, 20)
    y_j = np.arange(-250, 250, 20)
    X, Y, Z = aep.aep_map(x_j, y_j, 0, [], [], wd=[0])

    npt.assert_array_almost_equal(Z, 21.89, 2)


def test_aep_no_wake_loss_hornsrev():
    wt = hornsrev1.V80()
    x, y = hornsrev1.wt_x, hornsrev1.wt_y
    site = UniformWeibullSite([1], [10], [2], .75)
    site.default_ws = np.arange(3, 25)

    aep = AEPCalculator(site, wt, NOJ(wt))

    npt.assert_almost_equal(aep.calculate_AEP_no_wake_loss(x, y).sum() / 80, 8.260757098)
    cap_factor = aep.calculate_AEP(x, y).sum() / aep.calculate_AEP_no_wake_loss(x, y).sum()
    # print(cap_factor)
    npt.assert_almost_equal(cap_factor, 0.947175839142014)
