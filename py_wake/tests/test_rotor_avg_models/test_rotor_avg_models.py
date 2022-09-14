import pytest

import matplotlib.pyplot as plt
from py_wake import np
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel, WakeRadiusTopHat
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit,\
    BastankhahGaussianDeficit, BastankhahGaussian
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
from py_wake.examples.data.hornsrev1 import V80
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.flow_map import HorizontalGrid, XYGrid
from py_wake.rotor_avg_models import gauss_quadrature, PolarGridRotorAvg, \
    polar_gauss_quadrature, EqGridRotorAvg, GQGridRotorAvg, CGIRotorAvg, GridRotorAvg, WSPowerRotorAvg
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel, RotorCenter
from py_wake.site._site import UniformSite
from py_wake.superposition_models import SquaredSum, LinearSum, WeightedSum
from py_wake.tests import npt
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.utils.model_utils import get_models
from py_wake.wind_farm_models.engineering_models import All2AllIterative, PropagateDownwind, EngineeringWindFarmModel
from py_wake.turbulence_models.turbulence_model import TurbulenceModel


EngineeringWindFarmModel.verbose = False


def get_wfm(wfm=PropagateDownwind, wake_deficitModel=None,
            rotorAvgModel=None, turbulenceModel=None, site=UniformSite()):
    windTurbines = IEA37_WindTurbines()
    kwargs = dict(superpositionModel=SquaredSum(),
                  wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(rotorAvgModel=rotorAvgModel))
    if wake_deficitModel:
        kwargs['wake_deficitModel'] = wake_deficitModel
    if turbulenceModel:
        kwargs['turbulenceModel'] = turbulenceModel

    return wfm(site, windTurbines, **kwargs)


def test_RotorGridAvg_deficit():
    wfm = get_wfm()
    flow_map = wfm([0], [0], wd=270, ws=10).flow_map(HorizontalGrid(x=[500], y=np.arange(-100, 100)))
    flow_map.WS_eff.squeeze().plot()

    # fit with gaussian
    sigma = wfm.wake_deficitModel.sigma_ijlk(WS_ilk=[[[10]]], dw_ijlk=np.array([[[[500]]]]), D_src_il=[[130]]).squeeze()
    max_deficit = 10 - flow_map.WS_eff.min().item()
    x = np.linspace(-100, 100)
    plt.plot(x, 10 - (max_deficit * np.exp(-1 / 2 * x**2 / sigma**2)))

    R = wfm.windTurbines.diameter() / 2
    plt.axvline(-R, color='k')
    plt.axvline(R, color='k')

    for name, rotorAvgModel, ref1 in [
            ('RotorCenter', None, 7.172723970425709),
            ('RotorGrid2', EqGridRotorAvg(2), 7.495889360682771),
            ('RotorGrid4', EqGridRotorAvg(4), 7.710215921858325),
            ('RotorGrid100', EqGridRotorAvg(100), 7.820762402628349),
            ('RotorGQGrid_4,3', GQGridRotorAvg(4, 3), 7.826105012683896),
            ('RotorCGI4', CGIRotorAvg(4), 7.848406907726826),
            ('RotorCGI7', CGIRotorAvg(7), 7.819900693605533),
            ('RotorCGI9', CGIRotorAvg(9), 7.82149363932618),
            ('RotorCGI21', CGIRotorAvg(21), 7.821558905416136),
            ('GaussianOverlap', GaussianOverlapAvgModel(), 7.821566060575336)]:

        # test with PropagateDownwind
        wfm = get_wfm(rotorAvgModel=rotorAvgModel)
        sim_res = wfm([0, 500], [0, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.WS_eff_ilk[1, 0, 0], ref1)

        # test with All2AllIterative
        wfm = get_wfm(All2AllIterative, rotorAvgModel=rotorAvgModel)
        sim_res = wfm([0, 500], [0, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.WS_eff_ilk[1, 0, 0], ref1)

        plt.plot([-R, R], [sim_res.WS_eff_ilk[1, 0, 0]] * 2, label=name)

    if 0:
        plt.legend()
        plt.show()
    plt.close('all')


def test_RotorGridAvg_deficit_with_offset():
    site = UniformSite()
    windTurbines = IEA37_WindTurbines()
    wfm = get_wfm()
    y_lst = np.linspace(-100, 100)
    flow_map = wfm([0], [100], wd=270, ws=10).flow_map(HorizontalGrid(x=[500], y=y_lst))
    flow_map.WS_eff.squeeze().plot()

    R = windTurbines.diameter() / 2
    plt.axvline(-R, color='k')
    plt.axvline(R, color='k')

    for name, rotorAvgModel, ref1 in [
            ('RotorCenter', None, 9.22391392271725),
            ('RotorGrid2', EqGridRotorAvg(2), 9.201908170589759),
            ('RotorGrid4', EqGridRotorAvg(4), 9.193330679480756),
            ('RotorGrid100', EqGridRotorAvg(100), 9.187280275700319),
            ('RotorGQGrid_4,3', GQGridRotorAvg(4, 3), 9.196482675497514),
            ('RotorCGI4', CGIRotorAvg(4), 9.188296907553514),
            ('RotorCGI7', CGIRotorAvg(7), 9.187441520918014),
            ('RotorCGI9', CGIRotorAvg(9), 9.188024478200141),
            ('RotorCGI21', CGIRotorAvg(21), 9.187206836007253),
            ('GaussianOverlap', GaussianOverlapAvgModel(), 9.187203332450924)]:

        # test with PropagateDownwind
        wfm = get_wfm(rotorAvgModel=rotorAvgModel)
        sim_res = wfm([0, 500], [100, 0], wd=270, ws=10)

        npt.assert_almost_equal(sim_res.WS_eff_ilk[1, 0, 0], ref1)
        # print(name, sim_res.WS_eff_ilk[1, 0, 0])

        # test with All2AllIterative
        wfm = get_wfm(All2AllIterative, rotorAvgModel=rotorAvgModel)
        sim_res = wfm([0, 500], [100, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.WS_eff_ilk[1, 0, 0], ref1)

        plt.plot([-R, R], [sim_res.WS_eff_ilk[1, 0, 0]] * 2, label=name)
    if 0:
        plt.legend()
        plt.show()
    plt.close('all')


def test_RotorGridAvg_blockage_with_offset():
    site = UniformSite()
    windTurbines = IEA37_WindTurbines()
    wfm = All2AllIterative(site, windTurbines, NoWakeDeficit(), blockage_deficitModel=SelfSimilarityDeficit())
    y_lst = np.linspace(-100, 100)
    wfm([0], [100], wd=270, ws=10).flow_map(HorizontalGrid(x=[-500], y=y_lst)).WS_eff.squeeze().plot()

    R = windTurbines.diameter() / 2
    plt.axvline(-R, color='k')
    plt.axvline(R, color='k')

    for name, rotorAvgModel, ref1 in [
            ('RotorCenter', None, 9.970148274564286),
            ('RotorGrid2', EqGridRotorAvg(2), 9.97029112851441),
            ('RotorGrid4', EqGridRotorAvg(4), 9.970404117231945),
            ('RotorGrid100', EqGridRotorAvg(100), 9.970466874319124),
            ('RotorGQGrid_4,3', GQGridRotorAvg(4, 3), 9.9704726952638),
            ('RotorCGI4', CGIRotorAvg(4), 9.970468231487335),
            ('RotorCGI7', CGIRotorAvg(7), 9.970467335153515),
            ('RotorCGI9', CGIRotorAvg(9), 9.970467339738361),
            ('RotorCGI21', CGIRotorAvg(21), 9.970467339195823),
    ]:

        # test with All2AllIterative
        wfm = All2AllIterative(site, windTurbines,
                               NoWakeDeficit(),
                               blockage_deficitModel=SelfSimilarityDeficit(rotorAvgModel=rotorAvgModel))
        sim_res = wfm([0, -500], [100, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.WS_eff_ilk[1, 0, 0], ref1)
        # print(name, sim_res.WS_eff_ilk[1, 0, 0])

        plt.plot([-R, R], [sim_res.WS_eff_ilk[1, 0, 0]] * 2, label=name)
    if 0:
        plt.legend()
        plt.show()
    plt.close('all')


def test_RotorGridAvg_turbulence_with_offset():
    wfm = get_wfm(wake_deficitModel=NoWakeDeficit(), turbulenceModel=STF2017TurbulenceModel())
    y_lst = np.linspace(-100, 100)
    wfm([0], [50], wd=270, ws=10).flow_map(HorizontalGrid(x=[500], y=y_lst)).TI_eff.squeeze().plot()

    R = wfm.windTurbines.diameter() / 2
    plt.axvline(-R, color='k')
    plt.axvline(R, color='k')

    for name, rotorAvgModel, ref1 in [
            ('RotorCenter', None, .20626197737221919),
            ('RotorGrid2', EqGridRotorAvg(2), 0.19974031201677134),
            ('RotorGrid4', EqGridRotorAvg(4), 0.19499960283707082),
            ('RotorGrid100', EqGridRotorAvg(100), 0.1913413153414768),
            ('RotorGQGrid_4,3', GQGridRotorAvg(4, 3), 0.1921628716508336),
            ('RotorCGI4', CGIRotorAvg(4), 0.1921273970149179),
            ('RotorCGI7', CGIRotorAvg(7), 0.1924602171955057),
            ('RotorCGI9', CGIRotorAvg(9), 0.19067739131942635),
            ('RotorCGI21', CGIRotorAvg(21), 0.1906851729848792),
    ]:

        # test with PropagateDownwind
        wfm = get_wfm(wake_deficitModel=NoWakeDeficit(),
                      turbulenceModel=STF2017TurbulenceModel(rotorAvgModel=rotorAvgModel))
        sim_res = wfm([0, 500], [50, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.TI_eff_ilk[1, 0, 0], ref1)
        # print(name, sim_res.TI_eff_ilk[1, 0, 0])

        # test with All2AllIterative
        wfm = get_wfm(All2AllIterative, wake_deficitModel=NoWakeDeficit(),
                      turbulenceModel=STF2017TurbulenceModel(rotorAvgModel=rotorAvgModel))
        sim_res = wfm([0, 500], [50, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.TI_eff_ilk[1, 0, 0], ref1)
        # print(name, sim_res.TI_eff_ilk[1, 0, 0])

        plt.plot([-R, R], [sim_res.TI_eff_ilk[1, 0, 0]] * 2, label=name)
    if 0:
        plt.legend()
        plt.show()
    plt.close('all')


def test_RotorGridAvg_ti():
    site = IEA37Site(16)
    wfm = get_wfm(turbulenceModel=STF2017TurbulenceModel())
    flow_map = wfm([0, 500], [0, 0], wd=270, ws=10).flow_map(HorizontalGrid(x=[500], y=np.arange(-100, 100)))
    plt.plot(flow_map.Y[:, 0], flow_map.TI_eff_xylk[:, 0, 0, 0])
    R = wfm.windTurbines.diameter() / 2

    for name, rotorAvgModel, ref1 in [
            ('RotorCenter', None, 0.22292190804089568),
            ('RotorGrid2', EqGridRotorAvg(2), 0.2111162769995657),
            ('RotorGrid3', EqGridRotorAvg(3), 0.2058616982653193),
            ('RotorGrid4', EqGridRotorAvg(4), 0.2028701990648858),
            ('RotorGrid100', EqGridRotorAvg(100), 0.1985255601976247),
            ('RotorGQGrid_4,3', GQGridRotorAvg(4, 3), 0.1982984399750206)]:

        # test with PropagateDownwind
        wfm = get_wfm(turbulenceModel=STF2017TurbulenceModel(rotorAvgModel=rotorAvgModel), site=site)
        sim_res = wfm([0, 500], [0, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.TI_eff_ilk[1, 0, 0], ref1)

        # test with All2AllIterative
        wfm = get_wfm(All2AllIterative, turbulenceModel=STF2017TurbulenceModel(rotorAvgModel=rotorAvgModel), site=site)
        sim_res = wfm([0, 500], [0, 0], wd=270, ws=10)
        npt.assert_almost_equal(sim_res.TI_eff_ilk[1, 0, 0], ref1)

        plt.plot([-R, R], [sim_res.TI_eff_ilk[1, 0, 0]] * 2, label=name)
    if 0:
        plt.legend()
        plt.show()
    plt.close('all')


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
def test_with_all_deficit_models(WFM):
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    for deficitModel in get_models(WakeDeficitModel):
        wfm = WFM(site, windTurbines, wake_deficitModel=deficitModel(),
                  superpositionModel=LinearSum(),
                  deflectionModel=None, turbulenceModel=STF2017TurbulenceModel())

        wfm2 = WFM(site, windTurbines, wake_deficitModel=deficitModel(rotorAvgModel=EqGridRotorAvg(1)),
                   superpositionModel=LinearSum(),
                   deflectionModel=None, turbulenceModel=STF2017TurbulenceModel())
        kwargs = {'x': [0, 0, 500, 500], 'y': [0, 500, 0, 500], 'wd': [0], 'ws': [8]}
        npt.assert_equal(wfm.aep(**kwargs), wfm2.aep(**kwargs))

        wfm3 = WFM(site, windTurbines, wake_deficitModel=deficitModel(rotorAvgModel=CGIRotorAvg(7)),
                   superpositionModel=LinearSum(),
                   deflectionModel=None, turbulenceModel=STF2017TurbulenceModel())
        if isinstance(deficitModel(), WakeRadiusTopHat) or deficitModel == NoWakeDeficit:
            npt.assert_equal(wfm.aep(**kwargs), wfm3.aep(**kwargs))
        else:
            assert wfm.aep(**kwargs) < wfm3.aep(**kwargs)


@pytest.mark.parametrize('blockage_deficitModel', get_models(BlockageDeficitModel))
def test_with_all_blockage_models(blockage_deficitModel):
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    if blockage_deficitModel is not None:

        wfm_wo = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  blockage_deficitModel=blockage_deficitModel(),
                                  turbulenceModel=STF2017TurbulenceModel())
        wfm_w = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                 blockage_deficitModel=blockage_deficitModel(rotorAvgModel=CGIRotorAvg(7)),
                                 turbulenceModel=STF2017TurbulenceModel())
        kwargs = {'x': [0, 500], 'y': [0, 0], 'wd': [270], 'ws': [10]}
        assert wfm_w(**kwargs).WS_eff.sel(wt=0).item() > wfm_wo(**kwargs).WS_eff.sel(wt=0).item()


@pytest.mark.parametrize('turbulenceModel', get_models(TurbulenceModel))
def test_with_all_ti_models(turbulenceModel):
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    if turbulenceModel is not None:

        wfm_wo = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  turbulenceModel=STF2017TurbulenceModel())
        wfm_w = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                 turbulenceModel=STF2017TurbulenceModel(rotorAvgModel=CGIRotorAvg(7)))
        kwargs = {'x': [0, 500], 'y': [0, 0], 'wd': [270], 'ws': [10]}
        assert wfm_w(**kwargs).TI_eff.sel(wt=1).item() < wfm_wo(**kwargs).TI_eff.sel(wt=1).item()


@pytest.mark.parametrize('model', get_models(RotorAvgModel))
def test_with_weighted_sum(model):
    if model is None:
        return
    if model is RotorCenter:
        wfm = All2AllIterative(UniformSite(), V80(), BastankhahGaussianDeficit(rotorAvgModel=model()),
                               superpositionModel=WeightedSum())
        wfm([0, 500], [0, 0])
    else:
        with pytest.raises(AssertionError, match='WeightedSum only works with RotorCenter'):
            wfm = All2AllIterative(UniformSite(), V80(), BastankhahGaussianDeficit(),
                                   superpositionModel=WeightedSum(), rotorAvgModel=model())


def test_WSPowerRotorAvgModel():
    wfm = BastankhahGaussian(UniformSite(), V80())
    x, y = [0, 200], [0, 0]
    y_g = [-40, 0, 40]
    if 0:
        wfm(x, y, wd=270).flow_map().plot_wake_map()
        plt.show()

    ws_eff_p = wfm(x, y, wd=270).flow_map(XYGrid(x=200 - 1e-8, y=y_g)).WS_eff
    ws_eff_ref = np.mean(ws_eff_p**2)**(1 / 2)

    rotorAvgModel = WSPowerRotorAvg(GridRotorAvg(nodes_x=[-1, 0, 1], nodes_y=[0, 0, 0]), alpha=2)
    wfm = BastankhahGaussian(UniformSite(), V80(), rotorAvgModel=rotorAvgModel)
    npt.assert_almost_equal(wfm(x, y, wd=270).WS_eff.sel(wt=1).squeeze(), ws_eff_ref)
