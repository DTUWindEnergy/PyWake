import pytest

import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models import SelfSimilarityDeficit, SelfSimilarityDeficit2020
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.examples.data import hornsrev1
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.superposition_models import LinearSum
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import All2AllIterative

debug = False


@pytest.fixture(scope='module')
def setup():
    site = Hornsrev1Site()
    windTurbines = hornsrev1.HornsrevV80()
    ss = SelfSimilarityDeficit()
    ss20 = SelfSimilarityDeficit2020()
    return site, windTurbines, ss, ss20


def test_selfsimilarity_reference_figures(setup):
    ss = setup[2]
    ss20 = setup[3]
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
    deficit20_centerline = ss20.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                             dw_ijlk=x1.reshape((1, len(x1), 1, 1)),
                                             cw_ijlk=y1.reshape((1, len(y1), 1, 1)), ct_ilk=ct_ilk)[0, :, 0, 0]

    x2, y2 = np.array([-2 * R]), np.arange(200)
    deficit_radial = ss.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                     dw_ijlk=x2.reshape((1, len(x2), 1, 1)),
                                     cw_ijlk=y2.reshape((1, len(y2), 1, 1)), ct_ilk=ct_ilk)[0, :, 0, 0]
    deficit20_radial = ss20.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                         dw_ijlk=x2.reshape((1, len(x2), 1, 1)),
                                         cw_ijlk=y2.reshape((1, len(y2), 1, 1)), ct_ilk=ct_ilk)[0, :, 0, 0]

    # r12 = np.sqrt(ss.lambda_ * (ss.eta + (x2 / R) ** 2))   # Eq. (13) from [1]
    r12 = ss.r12(x2 / R)
    r12_20 = ss20.r12(x2 / R)

    if debug:
        plt.title('Fig 11 from [1]')
        plt.xlabel('x/R')
        plt.ylabel('a')
        plt.plot(x1 / R, deficit_centerline / ws)
        plt.plot(x1 / R, deficit20_centerline / ws, '--')
        print(list(np.round(deficit_centerline[::20], 6)))
        print(list(np.round(deficit20_centerline[::20], 6)))

        plt.figure()
        plt.title('Fig 10 from [1]')
        print(list(np.round(deficit_radial[::20] / deficit_radial[0], 6)))
        print(list(np.round(deficit20_radial[::20] / deficit20_radial[0], 6)))
        plt.xlabel('y/R12 (epsilon)')
        plt.ylabel('f')
        plt.plot((y2 / R) / r12, deficit_radial / deficit_radial[0])
        plt.plot((y2 / R) / r12_20, deficit20_radial / deficit20_radial[0], '--')
        plt.show()

    fig11_ref = np.array([[-0.025, -1, -2, -3, -4, -5], [0.318, 0.096, 0.035, 0.017, 0.010, 0.0071]]).T
    npt.assert_array_almost_equal(np.interp(-fig11_ref[:, 0], -x1 / R, deficit_centerline / ws), fig11_ref[:, 1], 1)
    npt.assert_array_almost_equal(deficit_centerline[::20], [0.0, 1.780159, 0.943215, 0.540855, 0.33998, 0.230329,
                                                             0.165257, 0.123906, 0.096151, 0.076686])
    npt.assert_array_almost_equal(deficit20_centerline[::20],
                                  [0.0,
                                   1.758202,
                                   0.931581,
                                   0.562763,
                                   0.362938,
                                   0.249322,
                                   0.180358,
                                   0.135934,
                                   0.105854,
                                   0.08463])

    fig10_ref = np.array([[0, 1, 2, 3], [1, .5, .15, .045]]).T
    npt.assert_array_almost_equal(np.interp(fig10_ref[:, 0], (y2 / R) / r12, deficit_radial / deficit_radial[0]),
                                  fig10_ref[:, 1], 1)
    npt.assert_array_almost_equal(deficit_radial[::20] / deficit_radial[0],
                                  [1.0, 0.933011, 0.772123, 0.589765, 0.430823, 0.307779,
                                   0.217575, 0.153065, 0.107446, 0.075348])
    npt.assert_array_almost_equal(deficit20_radial[::20] / deficit20_radial[0],
                                  [1.0, 0.937523, 0.78531, 0.608956, 0.451715, 0.32748, 0.23477, 0.167415, 0.119088, 0.084613])
