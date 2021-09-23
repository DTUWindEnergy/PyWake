import pytest
import numpy as np
from py_wake import NOJ
from py_wake.site._site import UniformSite
from py_wake.superposition_models import LinearSum, SquaredSum, MaxSum
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.flow_map import HorizontalGrid
from py_wake.tests.test_deficit_models.test_noj import NibeA0
import xarray as xr
from py_wake.examples.data.hornsrev1 import V80
from py_wake.deficit_models.deficit_model import BlockageDeficitModel, WakeDeficitModel
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit

d02 = 8.1 - 5.7
d12 = 8.1 - 4.90473373


@pytest.mark.parametrize('superpositionModel,res', [(LinearSum(), 8.1 - (d02 + d12)),
                                                    (SquaredSum(), 8.1 - np.hypot(d02, d12)),
                                                    (MaxSum(), 8.1 - d12)])
def test_superposition_models(superpositionModel, res):
    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, NibeA0, superpositionModel=superpositionModel)
    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]
    WS_eff_ilk = wake_model.calc_wt_interaction(x_i, y_i, h_i, [0, 0, 1], 0.0, 8.1)[0]
    npt.assert_array_almost_equal(WS_eff_ilk[-1, 0, 0], res)


@pytest.mark.parametrize('superpositionModel,sum_func', [(LinearSum(), np.sum),
                                                         (SquaredSum(), lambda x: np.hypot(*x)),
                                                         (MaxSum(), np.max)])
def test_superposition_model_indices(superpositionModel, sum_func):
    class WTSite(UniformSite):
        def local_wind(self, x_i=None, y_i=None, h_i=None, wd=None, ws=None, time=None, wd_bin_size=None, ws_bins=None):
            lw = UniformSite.local_wind(self, x_i=x_i, y_i=y_i, h_i=h_i, wd=wd, ws=ws,
                                        wd_bin_size=wd_bin_size, ws_bins=ws_bins)
            lw['WS'] = xr.DataArray(lw.WS_ilk + np.arange(len(x_i))[:, np.newaxis, np.newaxis],
                                    [('wt', [0, 1, 2]), ('wd', np.atleast_1d(wd)), ('ws', np.atleast_1d(ws))])

            return lw

    site = WTSite([1], 0.1)

    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]

    # WS_ilk different at each wt position
    WS_ilk = site.local_wind(x_i, y_i, h_i, wd=0, ws=8.1).WS_ilk
    npt.assert_array_equal(WS_ilk, np.reshape([8.1, 9.1, 10.1], (3, 1, 1)))

    for wake_model in [PropagateDownwind(site, NibeA0, wake_deficitModel=NOJDeficit(), superpositionModel=superpositionModel),
                       All2AllIterative(site, NibeA0, wake_deficitModel=NOJDeficit(), superpositionModel=superpositionModel)]:

        # No wake (ct = 0), i.e. WS_eff == WS
        WS_eff_ilk = wake_model.calc_wt_interaction(x_i, y_i, h_i, [1, 1, 1], 0.0, 8.1)[0]
        npt.assert_array_equal(WS_eff_ilk, WS_ilk)

        ref = WS_ilk - np.reshape([0, 3.75, sum_func([2.4, 3.58974359])], (3, 1, 1))

        # full wake (CT=8/9)
        WS_eff_ilk = wake_model.calc_wt_interaction(x_i, y_i, h_i, [0, 0, 0], 0.0, 8.1)[0]
        npt.assert_array_almost_equal(WS_eff_ilk, ref
                                      )

        sim_res = wake_model(x_i, y_i, h_i, [0, 0, 0], 0.0, 8.1)
        WS_eff_ilk = sim_res.flow_map(HorizontalGrid(x=[0], y=y_i, h=50)).WS_eff_xylk[:, 0]

        npt.assert_array_almost_equal(WS_eff_ilk, ref)


def test_diff_wake_blockage_superposition():
    site = UniformSite([1], 0.1)

    class MyWakeDeficit(WakeDeficitModel):
        args4deficit = ['dw_ijlk']

        def calc_deficit(self, dw_ijlk, **_):
            return (dw_ijlk > 0) * 2

    class MyBlockageDeficit(BlockageDeficitModel):
        args4deficit = ['dw_ijlk']

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
