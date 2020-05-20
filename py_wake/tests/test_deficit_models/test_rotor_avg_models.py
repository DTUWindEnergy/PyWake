import matplotlib.pyplot as plt
from py_wake.rotor_avg_models.rotor_avg_model import gauss_quadrature, PolarGridRotorAvgModel, RotorCenter, polar_gauss_quadrature, EqGridRotorAvgModel, GQGridRotorAvgModel
from py_wake.tests import npt
import numpy as np
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian, IEA37SimpleBastankhahGaussianDeficit
from py_wake.flow_map import HorizontalGrid
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.superposition_models import SquaredSum


def test_RotorGridAvgModel():
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
            ('RotorGrid2', EqGridRotorAvgModel(2), 7.495889360682771),
            ('RotorGrid3', EqGridRotorAvgModel(3), 7.633415167369133),
            ('RotorGrid4', EqGridRotorAvgModel(4), 7.710215921858325),
            ('RotorGrid100', EqGridRotorAvgModel(100), 7.820762402628349),
            ('RotorGQGrid_4,3', GQGridRotorAvgModel(4, 3), 7.826105012683896)]:

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
    if 0:
        plt.legend()
        plt.show()
    plt.close()


def test_gauss_quadrature():

    x, _, w = gauss_quadrature(4, 1)
    npt.assert_array_almost_equal(x, [-0.861136, -0.339981, 0.339981, 0.861136])
    npt.assert_array_almost_equal(w, np.array([0.347855, 0.652145, 0.652145, 0.347855]) / 2)


def test_RotorEqGridAvgModel():
    m = EqGridRotorAvgModel(3)
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
    m = GQGridRotorAvgModel(4, 3)
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
    m = PolarGridRotorAvgModel(*polar_gauss_quadrature(4, 3))

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
    npt.assert_array_almost_equal(m.nodes_y, [0.05, 0.25, 0.51, 0.71, -0.07, -0.33, -0.67,
                                              -0.93, 0.05, 0.25, 0.51, 0.71], 2)
    npt.assert_array_almost_equal(m.nodes_weight, [0.05, 0.09, 0.09, 0.05, 0.08, 0.14, 0.14,
                                                   0.08, 0.05, 0.09, 0.09, 0.05], 2)
