import warnings

import pytest

import matplotlib.pyplot as plt
from py_wake import NOJ
from py_wake import np
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit
from py_wake.examples.data import hornsrev1
from py_wake.examples.data.hornsrev1 import wt9_x, wt9_y, Hornsrev1Site
from py_wake.tests import npt
from py_wake.utils.gradients import autograd, plot_gradients, fd, cs
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.wind_turbines import WindTurbines, WindTurbine, OneTypeWindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines.wind_turbines_deprecated import DeprecatedWindTurbines, DeprecatedOneTypeWindTurbines
from numpy import newaxis as na
from py_wake.deficit_models.utils import ct2a_mom1d


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
        wfm = NOJ(Hornsrev1Site(), wts, ct2a=ct2a_mom1d)
        npt.assert_array_equal(wts.types(), [0])
        npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types0, yaw=0), 81.2066072392765)


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


def test_OneTypeWindTurbines_from_tabular():
    wt_u = np.array([3.99, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    wt_p = np.array([0, 55., 185., 369., 619., 941., 1326., 1741., 2133., 2436., 2617., 2702., 2734.,
                     2744., 2747., 2748., 2748., 2750., 2750., 2750., 2750., 2750., 2750.])
    wt_ct = np.array([0, 0.871, 0.853, 0.841, 0.841, 0.833, 0.797, 0.743, 0.635, 0.543, 0.424,
                      0.324, 0.258, 0.21, 0.175, 0.147, 0.126, 0.109, 0.095, 0.083, 0.074, 0.065, 0.059])
    wt = OneTypeWindTurbines.from_tabular(name="NEG-Micon 2750/92 (2750 kW)", diameter=92, hub_height=70,
                                          ws=wt_u, ct=wt_ct, power=wt_p, power_unit='kw')
    u = np.linspace(5, 10, 7)
    npt.assert_array_equal(wt.power(u), np.interp(u, wt_u, wt_p) * 1000)
    npt.assert_array_equal(wt.ct(u), np.interp(u, wt_u, wt_ct))


def test_dAEP_2wt():
    site = Hornsrev1Site()

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
                        3 * power_rated * (wsp - wsp_cut_in)**2 /
                        (wsp_rated - wsp_cut_in)**3, 0)

    def dct(wsp):
        return wsp * 0  # constant ct

    wt = DeprecatedOneTypeWindTurbines(name='test', diameter=130, hub_height=110,
                                       ct_func=ct, power_func=power, power_unit='w')
    wt.set_gradient_funcs(dpower, dct)
    wfm = PropagateDownwind(site, wt, IEA37SimpleBastankhahGaussianDeficit())

    # plot 2 wt case
    x, y = np.array([[0, 130 * 4], [0, 0]], dtype=float)
    x_lst = np.array([0., 1.]) * np.arange(1, 600, 10)[:, na]
    kwargs = {'ws': [10], 'wd': [270], 'yaw': 0}

    _, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.plot(x_lst[:, 1], [wfm.aep(x_, y, **kwargs) for x_ in x_lst])
    ax1.set_xlabel('Downwind distance [m]')
    ax1.set_ylabel('AEP [GWh]')

    x_ = x_lst[20]
    ax1.set_title("Center line")
    for grad in [fd, cs, autograd]:
        dAEPdx = grad(wfm.aep, argnum=0)(x_, y, **kwargs)[1]
        npt.assert_almost_equal(dAEPdx / 360, 3.976975605364392e-06, (10, 5)[grad == fd])
        dAEPdxy = wfm.aep_gradients(grad, ['x', 'y'])(x_, y, **kwargs)
        npt.assert_almost_equal(dAEPdxy[0][1] / 360, 3.976975605364392e-06, (10, 5)[grad == fd])
        dAEPdxy = wfm.aep_gradients(grad, ['x', 'y'], wd_chunks=2, x=x_, y=y, **kwargs)
        npt.assert_almost_equal(dAEPdxy[0][1] / 360, 3.976975605364392e-06, (10, 5)[grad == fd])
        plot_gradients(wfm.aep(x_, y, **kwargs), dAEPdx, x_[1], grad.__name__, step=100, ax=ax1)
    y_lst = np.array([0, 1.]) * np.arange(-100, 100, 5)[:, na]
    ax2.plot(y_lst[:, 1], [wfm.aep(x, y_, **kwargs) for y_ in y_lst])
    ax2.set_xlabel('Crosswind distance [m]')
    ax2.set_ylabel('AEP [GWh]')
    y_ = y_lst[25]
    ax2.set_title("%d m downstream" % x[1])
    for grad in [fd, cs, autograd]:
        dAEPdy = grad(wfm.aep, argnum=1)(x, y_, **kwargs)[1]
        plot_gradients(wfm.aep(x, y_, **kwargs), dAEPdy, y_[1], grad.__name__, step=50, ax=ax2)
        npt.assert_almost_equal(dAEPdy / 360, 3.794435973860448e-05, (10, 5)[grad == fd])
        dAEPdxy = wfm.aep_gradients(grad, ['x', 'y'])(x, y_, **kwargs)
        npt.assert_almost_equal(dAEPdxy[1][1] / 360, 3.794435973860448e-05, (10, 5)[grad == fd])

    if 0:
        plt.legend()
        plt.show()
    plt.close('all')
