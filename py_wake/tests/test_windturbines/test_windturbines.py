import os
from numpy import newaxis as na
import pytest
import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.examples.data import wtg_path, hornsrev1
from py_wake.examples.data.hornsrev1 import V80, wt9_x, wt9_y, Hornsrev1Site
from py_wake.examples.data.iea37 import iea37_reader
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37WindTurbines, IEA37Site
from py_wake.superposition_models import SquaredSum
from py_wake.tests import npt
from py_wake.utils import gradients
from py_wake.utils.gradients import use_autograd_in, autograd, plot_gradients
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.wind_turbines import WindTurbines, WindTurbine, OneTypeWindTurbines, wind_turbines_deprecated,\
    power_ct_functions
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


def get_wfms(wt, site=Hornsrev1Site(), wake_model=NOJDeficit(), superpositionModel=SquaredSum()):
    wfm1 = PropagateDownwind(site, wt, wake_model, superpositionModel=superpositionModel)
    wfm2 = All2AllIterative(site, wt, wake_model, superpositionModel=superpositionModel)
    wfm2.verbose = False
    wfm1.verbose = False
    return wfm1, wfm2


class V80Deprecated(OneTypeWindTurbines):
    def __init__(self):

        OneTypeWindTurbines.__init__(self, name='V80', diameter=80, hub_height=70, ct_func=self.ct_func,
                                     power_func=self.power_func, power_unit='w')

    def ct_func(self, ws, **_):
        return np.interp(ws, hornsrev1.ct_curve[:, 0], hornsrev1.ct_curve[:, 1])

    def power_func(self, ws, **_):
        return np.interp(ws, hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1])


def test_DeprecatedWindTurbines():
    v80 = V80Deprecated()

    for wts in [v80,
                WindTurbines(names=['V80'], diameters=[80], hub_heights=[70],
                             ct_funcs=v80._ct_funcs, power_funcs=v80._power_funcs, power_unit='w')]:

        types0 = [0] * 9
        for wfm in get_wfms(wts):
            # wfm = NOJ(Hornsrev1Site(), wts)
            npt.assert_array_equal(wts.types(), [0])
            npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types0), 81.2066072392765)


def test_WindTurbines():
    u_p, p = np.asarray(hornsrev1.power_curve).T.copy()

    u_ct, ct = hornsrev1.ct_curve.T
    npt.assert_array_equal(u_p, u_ct)
    curve = PowerCtTabular(ws=u_p, power=p, power_unit='w', ct=ct)

    for wts in [WindTurbine(name='V80', diameter=80, hub_height=70, powerCtFunction=curve),
                WindTurbines(names=['V80'], diameters=[80], hub_heights=[70], powerCtFunctions=[curve])]:
        types0 = [0] * 9
        for wfm in get_wfms(wts):
            npt.assert_array_equal(wts.types(), [0])
            npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types0), 81.2066072392765)


def test_V80_windturbines():
    wts = V80()
    types0 = [0] * 9
    for wfm in get_wfms(wts):
        npt.assert_array_equal(wts.types(), [0])
        npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types0), 81.2066072392765)


def test_IEA37WindTurbines():
    wt = IEA37WindTurbines()
    site = IEA37Site(16)
    x, y = site.initial_position.T
    for wfm in get_wfms(wt, site, IEA37SimpleBastankhahGaussianDeficit(), SquaredSum()):
        sim_res = wfm(x, y, ws=9.8, wd=np.arange(0, 360, 22.5))
        npt.assert_almost_equal(sim_res.aep(normalize_probabilities=True).sum() * 1e3, 366941.57116, 5)


def test_twotype_windturbines():
    v80 = V80()

    v88 = WindTurbine('V88', 88, 77,
                      powerCtFunction=PowerCtTabular(
                          hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1] * 1.1, 'w',
                          hornsrev1.ct_curve[:, 1]))

    wts = WindTurbines.from_WindTurbines([v80, v88])

    types0 = [0] * 9
    types1 = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    types2 = [1] * 9

    for wfm in get_wfms(wts):
        npt.assert_array_equal(wts.types(), [0, 1])
        npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types0), 81.2066072392765)
        npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types1), 83.72420504573488)
        npt.assert_almost_equal(wfm.aep(wt9_x, wt9_y, type=types2), 88.87227386796884)


@pytest.mark.parametrize('wts_wtg', [
    lambda: WindTurbine.from_WAsP_wtg([os.path.join(wtg_path, 'Vestas-V80.wtg'),
                                       os.path.join(wtg_path, 'NEG-Micon-2750.wtg')]),
    lambda:WindTurbines.from_WindTurbine_lst([WindTurbine.from_WAsP_wtg(os.path.join(wtg_path, 'Vestas-V80.wtg')),
                                              WindTurbine.from_WAsP_wtg(os.path.join(wtg_path, 'NEG-Micon-2750.wtg'))])])
def test_wasp_wtg(wts_wtg):
    wts_wtg = wts_wtg()
    assert(wts_wtg.name(type=0) == 'Vestas V80 (2MW, Offshore)')
    assert(wts_wtg.diameter(type=0) == 80)
    assert(wts_wtg.hub_height(type=0) == 67)
    npt.assert_array_equal(wts_wtg.power(np.array([0, 3, 3.99, 4, 5, 9, 18, 25, 25.01, 100]),
                                         type=0), np.array([0, 0, 0, 66600, 154000, 996000, 2e6, 2e6, 0, 0]))
    npt.assert_array_equal(wts_wtg.ct(np.array([1, 3.99, 4, 7, 9, 17, 25, 25.01, 100]), type=0),
                           np.array([0.052, 0.052, 0.818, 0.805, 0.807, 0.167, 0.052, 0.052, 0.052]))

    assert(wts_wtg.name(type=1) == 'NEG-Micon 2750/92 (2750 kW)')
    assert(wts_wtg.diameter(type=1) == 92)
    assert(wts_wtg.hub_height(type=1) == 70)
    npt.assert_array_equal(wts_wtg.power(np.array([0, 3, 3.99, 4, 5, 9, 18, 25, 25.01, 100]), type=1),
                           np.array([0, 0, 0, 55000, 185000, 1326000, 2748000, 2750000, 0, 0]))
    npt.assert_array_equal(wts_wtg.ct(np.array([1, 3.99, 4, 7, 9, 17, 100]), type=1),
                           np.array([.059, .059, 0.871, 0.841, 0.797, 0.175, 0.059]))


def test_multimode_wasp_wtg():
    wt = WindTurbine.from_WAsP_wtg(os.path.join(wtg_path, 'Vestas V112-3.0 MW.wtg'))
    u = np.linspace(0, 30, 100)
    rho1225 = wt.power(ws=10, mode=0)

    for mode in range(14):
        rho = wt.wt_data[mode]['AirDensity']
        npt.assert_almost_equal(wt.power(ws=10, mode=mode) / rho1225 * 1.225, rho, 0.02)
    if 0:
        for mode in range(14):
            plt.plot(u, wt.power(ws=u, mode=mode), label='mode %d, rho: %f' % (mode, rho))
        plt.legend()
        plt.show()


def test_get_defaults():
    v80 = V80()
    npt.assert_array_equal(np.array(v80.get_defaults(1))[:, 0], [70, 80])
    npt.assert_array_equal(np.array(v80.get_defaults(1, h_i=100))[:, 0], [100, 80])
    npt.assert_array_equal(np.array(v80.get_defaults(1, d_i=100))[:, 0], [70, 100])

#
# def test_speed_V80():
#     v80 = V80()
#     v80d = V80Deprecated()
#     ws = np.broadcast_to(np.arange(3., 26)[na, na, :], (80, 360, 23))
#     _, t = timeit(v80.power_ct, min_runs=5)(ws)
#     _, td = timeit(v80d.power_ct, min_runs=5)(ws)
#
#     assert (np.mean(t) - np.mean(td)) / np.mean(td) < .2, (np.mean(t) - np.mean(td)) / np.mean(td)
#
#     v80 = V80(method='pchip')
#     v80.enable_autograd()
#     _, t = timeit(autograd(v80.power_ct))(ws)
#     _, tfd = timeit(fd(v80.power_ct))(ws)
#     assert np.abs(np.mean(t) - np.mean(tfd)) / np.mean(tfd) < .15, np.abs(np.mean(t) - np.mean(tfd)) / np.mean(tfd)
#
#
# def test_speed_IEA37():
#     wt = IEA37_WindTurbines()
#     wtd = IEA37WindTurbinesDeprecated(gradient_functions=True)
#     ws = np.broadcast_to(np.arange(3., 26)[na, na, :], (80, 360, 23))
#     _, t = timeit(wt.power_ct, min_runs=5, line_profile=0, verbose=0,
#                   profile_funcs=[CubePowerSimpleCt._power_ct]
#                   )(ws)
#     _, td = timeit(wtd.power_ct, min_runs=5, line_profile=0, verbose=0,
#                    profile_funcs=[T.power, T.ct, T.dpower, T.dct])(ws, yaw=0)
#     npt.assert_almost_equal(np.mean(t), np.mean(td), decimal=0.008)
#
#     _, tfd = timeit(fd(wt.power_ct))(ws)
#     wt.enable_autograd()
#     _, t = timeit(autograd(wt.power_ct))(ws)
#     npt.assert_almost_equal(np.mean(t), np.mean(td), decimal=0.008)


def test_yaw():
    v80 = V80()
    yaw = np.arange(-30, 31)
    ws = np.zeros_like(yaw) + 8
    P0 = v80.power(ws[0])
    if 0:
        plt.plot(yaw, v80.power(ws, yaw=yaw) / P0)
        plt.plot(yaw, np.cos(np.deg2rad(yaw))**3)
        plt.grid()
        plt.figure()
        plt.plot(yaw, v80.ct(ws, yaw=yaw))
        plt.plot(yaw, v80.ct(ws) * np.cos(np.deg2rad(yaw))**2)
        plt.grid()
        plt.show()
    # Power in cube region
    npt.assert_array_almost_equal(v80.power(ws, yaw=yaw) / P0, np.cos(np.deg2rad(yaw))**3, 2)
    # ct in constant region
    npt.assert_array_almost_equal(v80.ct(ws, yaw=yaw), v80.ct(ws) * np.cos(np.deg2rad(yaw))**2, 3)


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
    plt.close('all')


def test_plot_yz2_types():
    wt = WindTurbines.from_WindTurbine_lst([IEA37_WindTurbines(), V80()])
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
    plt.close('all')


def test_set_gradients():
    wt = IEA37_WindTurbines()

    def dpctdu(ws, run_only):
        if run_only == 0:
            return np.where((ws > 4) & (ws <= 9.8),
                            100000 * ws,  # not the right gradient, but similar to the reference
                            0)
        else:
            return np.full(ws.shape, 0)
    wt.powerCtFunction.set_gradient_funcs(dpctdu)
    with use_autograd_in([WindTurbines, iea37_reader, power_ct_functions, wind_turbines_deprecated]):
        ws_lst = np.arange(3, 25, .1)
        plt.plot(ws_lst, wt.power(ws_lst))

        ws_pts = np.array([3., 6., 9., 12.])
        dpdu_lst = autograd(wt.power)(ws_pts)
        if 0:
            for dpdu, ws in zip(dpdu_lst, ws_pts):
                plot_gradients(wt.power(ws), dpdu, ws, "", 1)

            plt.show()
        dpdu_ref = np.where((ws_pts > 4) & (ws_pts <= 9.8),
                            100000 * ws_pts,
                            0)
        npt.assert_array_almost_equal(dpdu_lst, dpdu_ref)


def test_method():
    wt_linear = V80()
    wt_pchip = V80(method='pchip')
    wt_spline = V80(method='spline')
    ws_lst = np.arange(3, 25, .001)
    for wt in [wt_linear, wt_pchip, wt_spline]:
        wt.enable_autograd()

    ws_pts = [6.99, 7.01]
    with use_autograd_in():
        dpdu_linear_pts = autograd(wt_linear.power)(np.array(ws_pts))
        dpdu_pchip_pts = autograd(wt_pchip.power)(np.array(ws_pts))
        dpdu_spline_pts = autograd(wt_spline.power)(np.array(ws_pts))

    if 0:
        wt_dp_label_lst = [(wt_linear, dpdu_linear_pts, 'linear'),
                           (wt_pchip, dpdu_pchip_pts, 'pchip'),
                           (wt_spline, dpdu_spline_pts, 'spline')]
        for wt, dpdu_pts, label in wt_dp_label_lst:
            c = plt.plot(ws_lst, wt.power(ws_lst), label=label)[0].get_color()
            gradients.color_dict[label] = c

            for ws, dpdu in zip(ws_pts, dpdu_pts):
                plot_gradients(wt.power(ws), dpdu, ws, label)

        plt.legend()
        plt.figure()
        for wt, dpdu_pts, label in wt_dp_label_lst:
            plt.plot(ws_lst, wt_linear.power(ws_lst) - wt.power(ws_lst), label=label)

        plt.legend()
        plt.ylabel('Power difference wrt. linear')

        plt.figure()
        for wt, dpdu_pts, label in wt_dp_label_lst:
            plt.plot(np.diff(np.diff(wt.power(ws_lst))), label=label)
        plt.ylabel('Change of gradient')
        plt.legend()

        plt.show()

    # mean and max error
    for wt, mean_tol, absmean_tol, max_tol in [(wt_pchip, 213, 2323, 15632),
                                               (wt_spline, 1, 1, 1380)]:
        assert np.abs((wt_linear.power(ws_lst) - wt.power(ws_lst)).mean()) < mean_tol
        assert np.abs((wt_linear.power(ws_lst) - wt.power(ws_lst)).mean()) < absmean_tol
        assert np.abs((wt_linear.power(ws_lst) - wt.power(ws_lst)).max()) < max_tol

    for wt, diff_grad_max, dpdu_pts, ref_dpdu_pts in [(wt_linear, 64, dpdu_linear_pts, [178000.00007264, 236000.00003353]),
                                                      (wt_pchip, 0.2, dpdu_pchip_pts, [
                                                       202520.16516056, 203694.66294614]),
                                                      (wt_spline, 0.8, dpdu_spline_pts, [205555.17794162, 211859.45965873])]:
        assert np.diff(np.diff(wt.power(ws_lst))).max() < diff_grad_max
        npt.assert_array_almost_equal(dpdu_pts, ref_dpdu_pts)
