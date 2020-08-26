import matplotlib.pyplot as plt
from py_wake.rotor_avg_models.rotor_avg_model import gauss_quadrature, PolarGridRotorAvg, RotorCenter, \
    polar_gauss_quadrature, EqGridRotorAvg, GQGridRotorAvg, CGIRotorAvg
from py_wake.tests import npt
import numpy as np
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian, IEA37SimpleBastankhahGaussianDeficit
from py_wake.flow_map import HorizontalGrid
from py_wake.wind_farm_models.engineering_models import All2AllIterative, PropagateDownwind
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.tests.test_deficit_models.test_deficit_models import get_all_deficit_models
import pytest
from py_wake.turbulence_models.stf import STF2017TurbulenceModel


def test_RotorGridAvg():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site,
                                        windTurbines)
    flow_map = wfm([0, 500], [0, 0], wd=270, ws=10).flow_map(HorizontalGrid(x=[500], y=np.arange(-100, 100)))
    plt.plot(flow_map.Y[:, 0], flow_map.WS_eff_xylk[:, 0, 0, 0])
    R = windTurbines.diameter() / 2

    for name, rotorAvgModel, ref1 in [
            ('RotorCenter', RotorCenter(), 7.172723970425709),
            ('RotorGrid2', EqGridRotorAvg(2), 7.495889360682771),
            ('RotorGrid3', EqGridRotorAvg(3), 7.633415167369133),
            ('RotorGrid4', EqGridRotorAvg(4), 7.710215921858325),
            ('RotorGrid100', EqGridRotorAvg(100), 7.820762402628349),
            ('RotorGQGrid_4,3', GQGridRotorAvg(4, 3), 7.826105012683896)]:

        # test with PropagateDownwind
        wfm = IEA37SimpleBastankhahGaussian(site,
                                            windTurbines,
                                            rotorAvgModel=rotorAvgModel)
        sim_res = wfm([0, 500], [0, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.WS_eff_ilk[1, 0, 0], ref1)

        # test with All2AllIterative
        wfm = All2AllIterative(site, windTurbines,
                               IEA37SimpleBastankhahGaussianDeficit(),
                               rotorAvgModel=rotorAvgModel,
                               superpositionModel=SquaredSum())
        sim_res = wfm([0, 500], [0, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.WS_eff_ilk[1, 0, 0], ref1)

        plt.plot([-R, R], [sim_res.WS_eff_ilk[1, 0, 0]] * 2, label=name)
    if 1:
        plt.legend()
        plt.show()
    plt.close()


def test_gauss_quadrature():

    x, _, w = gauss_quadrature(4, 1)
    npt.assert_array_almost_equal(x, [-0.861136, -0.339981, 0.339981, 0.861136])
    npt.assert_array_almost_equal(w, np.array([0.347855, 0.652145, 0.652145, 0.347855]) / 2)


def test_RotorEqGridAvg():
    m = EqGridRotorAvg(3)
    npt.assert_array_almost_equal(m.nodes_x, [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5], 2)
    npt.assert_array_almost_equal(m.nodes_y, [-0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5], 2)
    if 0:
        for v in [m.nodes_x, m.nodes_y]:
            print(np.round(v, 2).tolist())
        plt.scatter(m.nodes_x, m.nodes_y, c=m.nodes_weight)
        plt.axis('equal')
        plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False))
        plt.ylim([-1, 1])
        plt.show()


def test_RotorGaussQuadratureGridAvgModel():
    m = GQGridRotorAvg(4, 3)
    npt.assert_array_almost_equal(m.nodes_x, [-0.34, 0.34, -0.86, -0.34, 0.34, 0.86, -0.34, 0.34], 2)
    npt.assert_array_almost_equal(m.nodes_y, [-0.77, -0.77, 0.0, 0.0, 0.0, 0.0, 0.77, 0.77], 2)
    npt.assert_array_almost_equal(m.nodes_weight, [0.11, 0.11, 0.1, 0.18, 0.18, 0.1, 0.11, 0.11], 2)
    if 0:
        for v in [m.nodes_x, m.nodes_y, m.nodes_weight]:
            print(np.round(v, 2).tolist())
        c = plt.scatter(m.nodes_x, m.nodes_y, c=m.nodes_weight)
        plt.colorbar(c)
        plt.axis('equal')
        plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False))
        plt.ylim([-1, 1])
        plt.show()


def test_polar_gauss_quadrature():
    m = PolarGridRotorAvg(*polar_gauss_quadrature(4, 3))

    if 0:
        for v in [m.nodes_x, m.nodes_y, m.nodes_weight]:
            print(np.round(v, 2).tolist())
        plt.scatter(m.nodes_x, m.nodes_y, c=m.nodes_weight)
        plt.axis('equal')
        plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False))
        plt.ylim([-1, 1])
        plt.show()

    npt.assert_array_almost_equal(m.nodes_x, [-0.05, -0.21, -0.44, -0.61, -0.0, -0.0,
                                              -0.0, -0.0, 0.05, 0.21, 0.44, 0.61], 2)
    npt.assert_array_almost_equal(m.nodes_y, [-0.05, -0.25, -0.51, -0.71, 0.07, 0.33,
                                              0.67, 0.93, -0.05, -0.25, -0.51, -0.71], 2)
    npt.assert_array_almost_equal(m.nodes_weight, [0.05, 0.09, 0.09, 0.05, 0.08, 0.14, 0.14,
                                                   0.08, 0.05, 0.09, 0.09, 0.05], 2)


@pytest.mark.parametrize('n,x,y,w', [
    (4, [-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, -0.5, 0.5], [0.25, 0.25, 0.25, 0.25]),
    (7, [0.0, -0.82, 0.82, -0.41, -0.41, 0.41, 0.41],
     [0.0, 0.0, 0.0, -0.71, 0.71, -0.71, 0.71],
     [0.25, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]),
    (9, [0.0, -1.0, 1.0, 0.0, 0.0, -0.5, -0.5, 0.5, 0.5],
     [0.0, 0.0, 0.0, -1.0, 1.0, -0.5, 0.5, -0.5, 0.5],
     [0.17, 0.04, 0.04, 0.04, 0.04, 0.17, 0.17, 0.17, 0.17]),
    (21, [0.0, 0.48, 0.18, -0.18, -0.48, -0.6, -0.48, -0.18, 0.18, 0.48, 0.6,
          0.74, 0.28, -0.28, -0.74, -0.92, -0.74, -0.28, 0.28, 0.74, 0.92],
        [0.0, 0.35, 0.57, 0.57, 0.35, 0.0, -0.35, -0.57, -0.57, -0.35,
         -0.0, 0.54, 0.87, 0.87, 0.54, 0.0, -0.54, -0.87, -0.87, -0.54, -0.0],
        [0.11, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
         0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])])
def test_CGIRotorAvg(n, x, y, w):
    m = CGIRotorAvg(n)

    if 0:
        for v in [m.nodes_x, m.nodes_y, m.nodes_weight]:
            print(np.round(v, 2).tolist())
        plt.scatter(m.nodes_x, m.nodes_y, c=m.nodes_weight)
        plt.axis('equal')
        plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False))
        plt.ylim([-1, 1])
        plt.show()

    assert (len(m.nodes_weight) == len(m.nodes_x) == len(m.nodes_y) == n)
    npt.assert_almost_equal(m.nodes_weight.sum(), 1)
    npt.assert_array_almost_equal(m.nodes_x, x, 2)
    npt.assert_array_almost_equal(m.nodes_y, y, 2)
    npt.assert_array_almost_equal(m.nodes_weight, w, 2)


@pytest.mark.parametrize('WFM', [All2AllIterative, PropagateDownwind])
@pytest.mark.parametrize('deficitModel', get_all_deficit_models())
def test_with_all_deficit_models(WFM, deficitModel):
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()

    wfm = WFM(site, windTurbines, wake_deficitModel=deficitModel,
              rotorAvgModel=RotorCenter(),
              superpositionModel=LinearSum(),
              deflectionModel=None, turbulenceModel=STF2017TurbulenceModel())

    wfm2 = WFM(site, windTurbines, wake_deficitModel=deficitModel,
               rotorAvgModel=EqGridRotorAvg(1),
               superpositionModel=LinearSum(),
               deflectionModel=None, turbulenceModel=STF2017TurbulenceModel())
    kwargs = {'x': [0, 0, 500, 500], 'y': [0, 500, 0, 500], 'wd': [0], 'ws': [8]}
    npt.assert_equal(wfm.aep(**kwargs), wfm2.aep(**kwargs))
