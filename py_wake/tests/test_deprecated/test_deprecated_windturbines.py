import matplotlib.pyplot as plt
import numpy as np
from py_wake import NOJ
from py_wake.examples.data import hornsrev1
from py_wake.examples.data.hornsrev1 import wt9_x, wt9_y, Hornsrev1Site
from py_wake.utils.gradients import use_autograd_in, autograd, plot_gradients, fd
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines, WindTurbine, OneTypeWindTurbines

import pytest
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.wind_turbines.wind_turbines_deprecated import DeprecatedWindTurbines
import warnings

WindFarmModel.verbose = False


@pytest.fixture(autouse=True)
def my_fixture():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


class V80Deprecated(OneTypeWindTurbines):
    def __init__(self):
        OneTypeWindTurbines.__init__(
            self, name='V80', diameter=80, hub_height=70,
            ct_func=lambda ws: np.interp(ws, hornsrev1.ct_curve[:, 0], hornsrev1.ct_curve[:, 1]),
            power_func=lambda ws: np.interp(ws, hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1]),
            power_unit='w')


def test_DeprecatedWindTurbines():

    for wts in [V80Deprecated(),
                WindTurbines(
                    names=['V80'], diameters=[80], hub_heights=[70],
                    ct_funcs=[lambda ws: np.interp(ws, hornsrev1.ct_curve[:, 0], hornsrev1.ct_curve[:, 1])],
                    power_funcs=[lambda ws: np.interp(
                        ws, hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1])],
                    power_unit='w')]:

        types0 = [0] * 9
        wfm = NOJ(Hornsrev1Site(), wts)
        npt.assert_array_equal(wts.types(), [0])
        npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types0), 81.2066072392765)


def test_deprecated_from_WindTurbines():
    v80 = V80Deprecated()

    v88 = WindTurbine('V88', 88, 77,
                      powerCtFunction=PowerCtTabular(
                          hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1] * 1.1, 'w',
                          hornsrev1.ct_curve[:, 1]))

    with pytest.raises(AssertionError, match='from_WindTurbines no longer supports DeprecatedWindTurbines'):
        WindTurbines.from_WindTurbines([v80, v88])


def test_twotype_windturbines():
    v80 = V80Deprecated()

    v88 = OneTypeWindTurbines('V88', 88, 77,
                              lambda ws: np.interp(ws, hornsrev1.ct_curve[:, 0], hornsrev1.ct_curve[:, 1]),
                              lambda ws: np.interp(ws, hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1] * 1.1),
                              'w')

    wts = DeprecatedWindTurbines.from_WindTurbines([v80, v88])
    ws = np.reshape([4, 6, 8, 10] * 2, (2, 4))
    types = np.asarray([0, 1])
    p = wts.power(ws, type=types)
    npt.assert_array_equal(p[0], p[1] / 1.1)


def test_get_defaults():
    v80 = V80Deprecated()
    npt.assert_array_equal(np.array(v80.get_defaults(1))[:, 0], [70, 80])
    npt.assert_array_equal(np.array(v80.get_defaults(1, h_i=100))[:, 0], [100, 80])
    npt.assert_array_equal(np.array(v80.get_defaults(1, d_i=100))[:, 0], [70, 100])


def test_yaw():
    v80 = V80Deprecated()
    yaw = np.deg2rad(np.arange(-30, 31))
    ws = np.zeros_like(yaw) + 8
    P0 = v80.power(ws[0])
    if 0:
        plt.plot(yaw, v80.power(ws, yaw=yaw) / P0)
        plt.plot(yaw, np.cos(yaw)**3)
        plt.grid()
        plt.figure()
        plt.plot(yaw, v80.ct(ws, yaw=yaw))
        plt.plot(yaw, v80.ct(ws) * np.cos(yaw)**2)
        plt.grid()
        plt.show()
    # Power in cube region
    npt.assert_array_almost_equal(v80.power(ws, yaw=yaw) / P0, np.cos(yaw)**3, 2)
    # ct in constant region
    npt.assert_array_almost_equal(v80.ct(ws, yaw=yaw), v80.ct(ws) * np.cos(yaw)**2, 3)
