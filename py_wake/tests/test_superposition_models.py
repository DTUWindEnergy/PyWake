import pytest
from py_wake import np
from py_wake import NOJ
from py_wake.site._site import UniformSite
from py_wake.superposition_models import LinearSum, SquaredSum, MaxSum, SqrMaxSum, WeightedSum
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.turbulence_models import TurbulenceModel
from py_wake.flow_map import HorizontalGrid
from py_wake.tests.test_deficit_models.test_noj import NibeA0
from py_wake.examples.data.hornsrev1 import V80
from py_wake.deficit_models.deficit_model import BlockageDeficitModel, WakeDeficitModel
from py_wake.deficit_models import NoWakeDeficit
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit

d02 = 8.1 - 5.7
d12 = 8.1 - 4.90473373


@pytest.mark.parametrize('superpositionModel,res', [(LinearSum(), 8.1 - (d02 + d12)),
                                                    (SquaredSum(), 8.1 - np.hypot(d02, d12)),
                                                    (MaxSum(), 8.1 - d12)])
def test_superposition_models(superpositionModel, res):
    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, NibeA0(), superpositionModel=superpositionModel)
    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]
    WS_eff = wake_model(x_i, y_i, h_i, type=[0, 0, 1], wd=0.0, ws=8.1).WS_eff
    npt.assert_array_almost_equal(WS_eff.squeeze()[-1], res)


@pytest.mark.parametrize('superpositionModel,res', [(LinearSum(), 0.1 + (.6 + .2)),
                                                    (SquaredSum(), .1 + np.hypot(.6, .2)),
                                                    (MaxSum(), .1 + .6),
                                                    (SqrMaxSum(), np.hypot(.1, .6))])
def test_superposition_models_TI(superpositionModel, res):
    site = UniformSite([1], 0.1)

    class MyTurbulenceModel(TurbulenceModel):

        def calc_added_turbulence(self, dw_ijlk, **_):
            return 1.2 - dw_ijlk / 100

    wake_model = PropagateDownwind(
        site,
        NibeA0(),
        NoWakeDeficit(),
        turbulenceModel=MyTurbulenceModel(superpositionModel))
    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]
    TI_eff = wake_model(x_i, y_i, h_i, type=[0, 0, 1], wd=0.0, ws=8.1).TI_eff
    npt.assert_array_almost_equal(TI_eff[-1, 0, 0], res)


@pytest.mark.parametrize('superpositionModel,sum_func', [(LinearSum(), np.sum),
                                                         (SquaredSum(), lambda x: np.hypot(*x)),
                                                         (MaxSum(), np.max)])
def test_superposition_model_indices(superpositionModel, sum_func):
    class WTSite(UniformSite):
        def local_wind(self, x=None, y=None, h=None,
                       wd=None, ws=None, time=None, wd_bin_size=None, ws_bins=None, **_):
            lw = UniformSite.local_wind(self, x=x, y=y, h=h, wd=wd, ws=ws,
                                        wd_bin_size=wd_bin_size, ws_bins=ws_bins)
            lw['WS_ilk'] = lw.WS_ilk + np.arange(len(x))[:, np.newaxis, np.newaxis]
            return lw

    site = WTSite([1], 0.1)

    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]

    # WS_ilk different at each wt position
    WS_ilk = site.local_wind(x_i, y_i, h_i, wd=0, ws=8.1).WS_ilk
    npt.assert_array_equal(WS_ilk, np.reshape([8.1, 9.1, 10.1], (3, 1, 1)))

    for wake_model in [PropagateDownwind(site, NibeA0(), wake_deficitModel=NOJDeficit(), superpositionModel=superpositionModel),
                       All2AllIterative(site, NibeA0(), wake_deficitModel=NOJDeficit(), superpositionModel=superpositionModel)]:

        # No wake (ct = 0), i.e. WS_eff == WS
        WS_eff = wake_model(x_i, y_i, h_i, type=[1, 1, 1], wd=0.0, ws=8.1).WS_eff
        npt.assert_array_equal(WS_eff, WS_ilk)

        ref = WS_ilk - np.reshape([0, 3.75, sum_func([2.4, 3.58974359])], (3, 1, 1))

        # full wake (CT=8/9)
        WS_eff = wake_model(x_i, y_i, h_i, wd=0.0, ws=8.1).WS_eff
        npt.assert_array_almost_equal(WS_eff, ref)

        sim_res = wake_model(x_i, y_i, h_i, [0, 0, 0], 0.0, 8.1)
        WS_eff_ilk = sim_res.flow_map(HorizontalGrid(x=[0], y=y_i, h=50)).WS_eff_xylk[:, 0]

        npt.assert_array_almost_equal(WS_eff_ilk, ref)


def test_diff_wake_blockage_superposition():
    site = UniformSite([1], 0.1)

    class MyWakeDeficit(WakeDeficitModel):
        def calc_deficit(self, dw_ijlk, **_):
            return (dw_ijlk > 0) * 2

    class MyBlockageDeficit(BlockageDeficitModel):

        def __init__(self, superpositionModel=None):
            BlockageDeficitModel.__init__(self, upstream_only=True, superpositionModel=superpositionModel)

        def calc_deficit(self, dw_ijlk, **_):
            return (dw_ijlk < 0) * .3

    wfm = All2AllIterative(site, V80(), wake_deficitModel=MyWakeDeficit(), superpositionModel=SquaredSum(),
                           blockage_deficitModel=MyBlockageDeficit(superpositionModel=LinearSum()))
    x = np.arange(5) * 160
    y = x * 0
    sim_res = wfm(x, y, ws=10, wd=270)
    npt.assert_array_almost_equal(sim_res.WS_eff.squeeze(), [10 - (4 - i) * .3 - np.sqrt(i * 2**2) for i in range(5)])


def test_WeightedSum_blockage():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wfm = All2AllIterative(site, windTurbines, wake_deficitModel=BastankhahGaussianDeficit(),
                           # blockage_deficitModel=SelfSimilarityDeficit(),
                           superpositionModel=WeightedSum())
    wfm(x, y, wd=270)  # .flow_map().plot_wake_map()
    # import matplotlib.pyplot as plt
    # plt.show()
