import os

import matplotlib.pyplot as plt
import numpy as np
from py_wake import NOJ
from py_wake.examples.data import wtg_path, hornsrev1
from py_wake.examples.data.hornsrev1 import V80, wt9_x, wt9_y, Hornsrev1Site
from py_wake.examples.data.iea37 import iea37_reader
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.utils.gradients import use_autograd_in, autograd, plot_gradients, fd
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines


def _test_wts_wtg(wts_wtg):
    assert(wts_wtg.name(types=0) == 'Vestas V80 (2MW, Offshore)')
    assert(wts_wtg.diameter(types=0) == 80)
    assert(wts_wtg.hub_height(types=0) == 67)
    npt.assert_array_equal(wts_wtg.power(np.array([0, 3, 5, 9, 18, 26]),
                                         type_i=0), np.array([0, 0, 154000, 996000, 2000000, 0]))
    npt.assert_array_equal(wts_wtg.ct(np.array([1, 4, 7, 9, 17, 27]), type_i=0),
                           np.array([0.052, 0.818, 0.805, 0.807, 0.167, 0.052]))

    assert(wts_wtg.name(types=1) == 'NEG-Micon 2750/92 (2750 kW)')
    assert(wts_wtg.diameter(types=1) == 92)
    assert(wts_wtg.hub_height(types=1) == 70)
    npt.assert_array_equal(wts_wtg.power(np.array([0, 3, 5, 9, 18, 26]),
                                         type_i=1), np.array([0, 0, 185000, 1326000, 2748000, 0]))
    npt.assert_array_equal(wts_wtg.ct(np.array([1, 4, 7, 9, 17, 27]), type_i=1),
                           np.array([.059, 0.871, 0.841, 0.797, 0.175, 0.059]))


def test_from_WAsP_wtg():
    vestas_v80_wtg = os.path.join(wtg_path, 'Vestas-V80.wtg')
    NEG_2750_wtg = os.path.join(wtg_path, 'NEG-Micon-2750.wtg')
    _test_wts_wtg(WindTurbines.from_WAsP_wtg([vestas_v80_wtg, NEG_2750_wtg]))


def test_from_WindTurbines():
    vestas_v80_wtg = WindTurbines.from_WAsP_wtg(os.path.join(wtg_path, 'Vestas-V80.wtg'))
    NEG_2750_wtg = WindTurbines.from_WAsP_wtg(os.path.join(wtg_path, 'NEG-Micon-2750.wtg'))
    _test_wts_wtg(WindTurbines.from_WindTurbines([vestas_v80_wtg, NEG_2750_wtg]))


def test_twotype_windturbines():
    v80 = V80()

    def power(ws, types):
        power = v80.power(ws)
        # add 10% type 1 turbines
        power[types == 1] *= 1.1
        return power

    power_curve = hornsrev1.power_curve
    wts = WindTurbines(names=['V80', 'V88'],
                       diameters=[80, 88],
                       hub_heights=[70, 77],
                       ct_funcs=[v80.ct_funcs[0],
                                 v80.ct_funcs[0]],
                       power_funcs=[lambda ws, yaw: np.interp(ws, power_curve[:, 0], power_curve[:, 1]),
                                    lambda ws, yaw: np.interp(ws, power_curve[:, 0], power_curve[:, 1]) * 1.1],
                       power_unit='w'
                       )

    import matplotlib.pyplot as plt
    types0 = [0] * 9
    types1 = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    types2 = [1] * 9
    wts.plot(wt9_x, wt9_y, types1)
    wfm = NOJ(Hornsrev1Site(), wts)
    npt.assert_array_equal(wts.types(), [0, 1])
    npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types0), 81.2066072392765)
    npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types1), 83.72420504573488)
    npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types2), 88.87227386796884)
    if 0:
        plt.show()


def test_get_defaults():
    v80 = V80()
    npt.assert_array_equal(np.array(v80.get_defaults(1))[:, 0], [0, 70, 80])
    npt.assert_array_equal(np.array(v80.get_defaults(1, h_i=100))[:, 0], [0, 100, 80])
    npt.assert_array_equal(np.array(v80.get_defaults(1, d_i=100))[:, 0], [0, 70, 100])


def test_yaw():
    v80 = V80()
    yaw = np.deg2rad(np.arange(-30, 31))
    ws = np.zeros_like(yaw) + 8
    P0 = v80.power(ws[0])
    if 0:
        plt.plot(yaw, v80.power(ws, 0, yaw) / P0)
        plt.plot(yaw, np.cos(yaw)**3)
        plt.grid()
        plt.show()
    npt.assert_array_almost_equal(v80.power(ws, 0, yaw) / P0, np.cos(yaw)**3, 2)


def test_gradients():
    wt = IEA37_WindTurbines()
    with use_autograd_in([WindTurbines, iea37_reader]):
        ws_lst = np.arange(3, 25, .1)
        plt.plot(ws_lst, wt.power(ws_lst))

        ws_pts = np.array([3., 6., 9., 12.])
        dpdu_lst = np.diag(autograd(wt.power)(ws_pts))
        if 0:
            for dpdu, ws in zip(dpdu_lst, ws_pts):
                plot_gradients(wt.power(ws), dpdu, ws, "", 1)

            plt.show()
        dpdu_ref = np.where((ws_pts > 4) & (ws_pts <= 9.8),
                            3 * 3350000 * (ws_pts - 4)**2 / (9.8 - 4)**3,
                            0)

        npt.assert_array_almost_equal(dpdu_lst, dpdu_ref)


def test_plot_yz():
    wt = IEA37_WindTurbines()
    yaw_lst = np.array([-30, 0, 30])
    for wd in 0, 45, 90:
        plt.figure()
        wt.plot_yz(yaw_lst * 20, wd=wd, yaw=yaw_lst)
        for i, yaw in enumerate(yaw_lst):
            plt.plot([], 'gray', label="WT %d yaw: %d deg" % (i, yaw))
        plt.legend()
        plt.title("WD: %s" % wd)
    if 0:
        plt.show()
    plt.close()


def test_plot_yz2_types():
    wt = WindTurbines.from_WindTurbines([IEA37_WindTurbines(), V80()])
    yaw_lst = np.array([-30, 0, 30])
    for wd in 0, 45, 90:
        plt.figure()
        wt.plot_yz(yaw_lst * 20, types=[0, 1, 0], wd=wd, yaw=yaw_lst)
        for i, yaw in enumerate(yaw_lst):
            plt.plot([], 'gray', label="WT %d yaw: %d deg" % (i, yaw))
        plt.legend()
        plt.title("WD: %s" % wd)
    if 0:
        plt.show()
    plt.close()


def test_set_gradients():
    wt = IEA37_WindTurbines()

    wt.set_gradient_funcs(lambda ws: np.where((ws > 4) & (ws <= 9.8),
                                              100000 * ws,  # not the right gradient, but similar to the reference
                                              0), lambda ws: 0)
    with use_autograd_in([WindTurbines, iea37_reader]):
        ws_lst = np.arange(3, 25, .1)
        plt.plot(ws_lst, wt.power(ws_lst))

        ws_pts = np.array([3., 6., 9., 12.])
        dpdu_lst = np.diag(autograd(wt.power)(ws_pts))
        if 0:
            for dpdu, ws in zip(dpdu_lst, ws_pts):
                plot_gradients(wt.power(ws), dpdu, ws, "", 1)

            plt.show()
        dpdu_ref = np.where((ws_pts > 4) & (ws_pts <= 9.8),
                            100000 * ws_pts,
                            0)

        npt.assert_array_almost_equal(dpdu_lst, dpdu_ref)


def test_spline():
    wt_tab = V80()
    wt_spline = V80()
    wt_spline.spline_ct_power(err_tol_factor=1e-2)
    ws_lst = np.arange(3, 25, .001)

    # mean and max error
    assert (wt_tab.power(ws_lst) - wt_spline.power(ws_lst)).mean() < 1
    assert ((wt_tab.power(ws_lst) - wt_spline.power(ws_lst)).max()) < 1400

    # max change of gradient 80 times lower
    assert np.diff(np.diff(wt_spline.power(ws_lst))).max() * 80 < np.diff(np.diff(wt_tab.power(ws_lst))).max()

    ws_pts = [6.99, 7.01]
    dpdu_tab_pts = np.diag(fd(wt_tab.power)(np.array(ws_pts)))
    with use_autograd_in():
        dpdu_spline_pts = np.diag(autograd(wt_spline.power)(np.array(ws_pts)))
    npt.assert_array_almost_equal(dpdu_spline_pts, [205555.17794162, 211859.45965873])

    if 0:
        plt.plot(ws_lst, wt_tab.power(ws_lst))
        plt.plot(ws_lst, wt_spline.power(ws_lst))

        for wt, dpdu_pts, label in [(wt_tab, dpdu_tab_pts, 'V80 tabular'),
                                    (wt_spline, dpdu_spline_pts, 'V80 spline')]:
            for ws, dpdu in zip(ws_pts, dpdu_pts):
                plot_gradients(wt.power(ws), dpdu, ws, label, 1)

        ax = plt.gca().twinx()
        ax.plot(ws_lst, wt.power(ws_lst) - wt_spline.power(ws_lst))
        plt.figure()
        plt.plot(np.diff(np.diff(wt_tab.power(ws_lst))))
        plt.plot(np.diff(np.diff(wt_spline.power(ws_lst))))
        plt.show()
