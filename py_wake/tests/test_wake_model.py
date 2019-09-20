import numpy as np
import pytest

from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.wake_models.noj import NOJ
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windrose
from py_wake.site._site import UniformSite
from py_wake.aep_calculator import AEPCalculator
from py_wake.tests import npt
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.wake_models.fuga import Fuga
from py_wake.tests.test_files.fuga import LUT_path_2MW_z0_0_03
from py_wake.wake_model import MaxSum


def test_wake_model():
    _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
    site = UniformSite(freq, ti=0.075)
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
    wake_model = NOJ(site, windTurbines)
    aep = AEPCalculator(wake_model)
    with pytest.raises(ValueError, match="Turbines 0 and 1 are at the same position"):
        aep.calculate_AEP([0, 0], [0, 0], wd=np.arange(0, 360, 22.5), ws=[9.8])


def test_wec():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wake_model = Fuga(LUT_path_2MW_z0_0_03, site, wts)
    aep = AEPCalculator(wake_model)
    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    _, _, Z_wec1 = aep.wake_map(x_j, y_j, 70, wt_x, wt_y, wt_height=70, wd=[30], ws=[10])
    aep.wake_model.wec = 2
    X, Y, Z_wec2 = aep.wake_map(x_j, y_j, 70, wt_x, wt_y, wt_height=70, wd=[30], ws=[10])

    if 0:
        import matplotlib.pyplot as plt

        c = plt.contourf(X, Y, Z_wec1, np.arange(6, 10.5, .1))
        plt.colorbar(c)
        plt.plot(X[0], Y[140])
        wts.plot(wt_x, wt_y)
        plt.figure()
        c = plt.contourf(X, Y, Z_wec2, np.arange(6, 10.5, .1))
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
        [10.0547, 10.0519, 10.0718, 10.0093, 9.6786, 7.8589, 6.8539, 9.2199,
         9.9837, 10.036, 10.0796, 10.0469, 10.0439, 9.1866, 7.2552, 9.1518,
         10.0449, 10.0261, 10.0353, 9.9256, 9.319, 8.0062, 6.789, 8.3578,
         9.9393, 10.0332, 10.0191, 10.0186, 10.0191, 10.0139], 4)
    npt.assert_array_almost_equal(
        Z_wec2[140, 100:400:10],
        [10.0297, 9.9626, 9.7579, 9.2434, 8.2318, 7.008, 6.7039, 7.7303, 9.0101,
         9.6877, 9.9068, 9.7497, 9.1127, 7.9505, 7.26, 7.9551, 9.2104, 9.7458,
         9.6637, 9.1425, 8.2403, 7.1034, 6.5109, 7.2764, 8.7653, 9.7139, 9.9718,
         10.01, 10.0252, 10.0357], 4)


def test_max_sum():
    ms = MaxSum()
    npt.assert_array_equal(ms.calc_effective_WS(WS_lk=[[10]], deficit_ilk=[[[1]], [[2]]]), 8)
