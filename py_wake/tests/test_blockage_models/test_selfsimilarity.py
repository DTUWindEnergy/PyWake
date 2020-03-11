import pytest

import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models import SelfSimilarityDeficit
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.examples.data import hornsrev1
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.superposition_models import LinearSum
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import All2AllIterative


@pytest.fixture(scope='module')
def setup():
    site = Hornsrev1Site()
    windTurbines = hornsrev1.HornsrevV80()
    ss = SelfSimilarityDeficit()
    return site, windTurbines, ss


def test_selfsimilarity_reference_figures(setup):
    ss = setup[2]
    ws = 10
    D = 80
    R = D / 2
    WS_ilk = np.array([[[ws]]])
    D_src_il = np.array([[D]])
    ct_ilk = np.array([[[.8]]])

    x1, y1 = -np.arange(200), np.array([0])
    deficit_centerline = ss.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                         dw_ijlk=x1.reshape((1, len(x1), 1, 1)),
                                         cw_ijlk=y1.reshape((1, len(y1), 1, 1)), ct_ilk=ct_ilk)[0, :, 0, 0]

    x2, y2 = np.array([-2 * R]), np.arange(200)
    deficit_radial = ss.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                     dw_ijlk=x2.reshape((1, len(x2), 1, 1)),
                                     cw_ijlk=y2.reshape((1, len(y2), 1, 1)), ct_ilk=ct_ilk)[0, :, 0, 0]

    r12 = np.sqrt(ss.lambda_ * (ss.eta + (x2 / R) ** 2))   # Eq. (13) from [1]

    if 0:
        plt.title('Fig 11 from [1]')
        plt.xlabel('x/R')
        plt.ylabel('a')
        plt.plot(x1 / R, deficit_centerline / ws)
        print(list(np.round(deficit_centerline[::20], 6)))

        plt.figure()
        plt.title('Fig 10 from [1]')
        print(list(np.round(deficit_radial[::20] / deficit_radial[0], 6)))
        plt.xlabel('y/R12 (epsilon)')
        plt.ylabel('f')
        plt.plot((y2 / R) / r12, deficit_radial / deficit_radial[0])
        plt.show()

    fig11_ref = np.array([[-0.025, -1, -2, -3, -4, -5], [0.318, 0.096, 0.035, 0.017, 0.010, 0.0071]]).T
    npt.assert_array_almost_equal(np.interp(-fig11_ref[:, 0], -x1 / R, deficit_centerline / ws), fig11_ref[:, 1], 1)
    npt.assert_array_almost_equal(deficit_centerline[::20], [0, 1.806478, 0.95716, 0.548851, 0.345007,
                                                             0.233735, 0.1677, 0.125738, 0.097573, 0.077819])

    fig10_ref = np.array([[0, 1, 2, 3], [1, .5, .15, .045]]).T
    npt.assert_array_almost_equal(np.interp(fig10_ref[:, 0], (y2 / R) / r12, deficit_radial / deficit_radial[0]),
                                  fig10_ref[:, 1], 1)
    npt.assert_array_almost_equal(deficit_radial[::20] / deficit_radial[0],
                                  [1.0, 0.933011, 0.772123, 0.589765, 0.430823, 0.307779,
                                   0.217575, 0.153065, 0.107446, 0.075348])


def test_blockage_map(setup):
    site, windTurbines, ss = setup
    wm = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                          superpositionModel=LinearSum(), blockage_deficitModel=ss)

    flow_map = wm(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
    X_j, Y_j = flow_map.XY
    WS_eff = flow_map.WS_eff_xylk[:, :, 0, 0]

    if 0:
        plt.contourf(X_j, Y_j, WS_eff)
        plt.plot(X_j[200, ::50], Y_j[200, ::50], '.-')
        plt.plot(X_j[250, ::50], Y_j[250, ::50], '.-')
        print(list(np.round(WS_eff[200, ::50], 6)))
        print(list(np.round(WS_eff[250, ::50], 6)))
        ss.windTurbines.plot([0], [0], wd=[270])
        plt.show()

    npt.assert_array_almost_equal(WS_eff[200, ::50], [9.940967, 9.911659, 9.855934,
                                                      9.736016, 9.44199, 10.0, 10.0, 10.0, 10.0, 10.0])
    npt.assert_array_almost_equal(WS_eff[250, ::50], [9.937601, 9.90397, 9.834701,
                                                      9.659045, 9.049764, 10.0, 10.0, 10.0, 10.0, 10.0])


def test_wake_and_blockage(setup):
    site, windTurbines, ss = setup
    noj_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NOJDeficit(),
                              blockage_deficitModel=ss, superpositionModel=LinearSum())

    flow_map = noj_ss(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
    X_j, Y_j = flow_map.XY
    WS_eff = flow_map.WS_eff_xylk[:, :, 0, 0]

    npt.assert_array_almost_equal(WS_eff[200, ::50], [9.940967, 9.911659, 9.855934, 9.736016, 9.44199, 4.560631,
                                                      5.505472, 6.223921, 6.782925, 7.226399])
    npt.assert_array_almost_equal(WS_eff[250, ::50], [9.937601, 9.90397, 9.834701, 9.659045, 9.049764, 4.560631,
                                                      5.505472, 6.223921, 6.782925, 7.226399])

    if 0:
        plt.contourf(X_j, Y_j, WS_eff)
        plt.plot(X_j[200, ::50], Y_j[200, ::50], '.-')
        plt.plot(X_j[250, ::50], Y_j[250, ::50], '.-')
        print(list(np.round(WS_eff[200, ::50], 6)))
        print(list(np.round(WS_eff[250, ::50], 6)))
        ss.windTurbines.plot([0], [0], wd=[270])
        plt.show()


def test_aep_two_turbines(setup):
    site, windTurbines, ss = setup

    nwm_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                              blockage_deficitModel=ss, superpositionModel=LinearSum())

    sim_res = nwm_ss(x=[0, 80 * 3], y=[0, 0])
    aep_no_blockage = sim_res.aep_ilk(with_wake_loss=False).sum(2)
    aep = sim_res.aep_ilk().sum(2)

    # blockage reduce aep(wd=270) by .5%
    npt.assert_almost_equal((aep_no_blockage[0, 270] - aep[0, 270]) / aep_no_blockage[0, 270] * 100, 0.4896853)

    if 0:
        plt.plot(sim_res.WS_eff_ilk[:, :, 7].T)
        plt.show()
