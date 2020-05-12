import pytest
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.tests import npt
from py_wake.turbulence_models.stf import STF2005TurbulenceModel, STF2017TurbulenceModel
from py_wake import NOJ
from py_wake.wind_farm_models.engineering_models import All2AllIterative, PropagateDownwind
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.site._site import UniformSite
import numpy as np
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.tests.test_wind_farm_models.test_noj import NibeA0
from py_wake.flow_map import HorizontalGrid


@pytest.mark.parametrize('turbulence_model,ref_ti',
                         [(STF2005TurbulenceModel(), [0.075, 0.075, 0.075, 0.104, 0.167, 0.124, 0.075, 0.075,
                                                      0.075, 0.075, 0.075, 0.075, 0.104, 0.136, 0.098, 0.104]),
                          (STF2017TurbulenceModel(), [0.075, 0.075, 0.075, 0.114, 0.197, 0.142, 0.075, 0.075,
                                                      0.075, 0.075, 0.075, 0.075, 0.115, 0.158, 0.108, 0.115])])
def test_stf(turbulence_model, ref_ti):
    # setup site, turbines and wind farm model
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    for wake_model in [NOJ(site, windTurbines, turbulenceModel=turbulence_model),
                       All2AllIterative(site, windTurbines,
                                        wake_deficitModel=NOJDeficit(),
                                        superpositionModel=SquaredSum(),
                                        turbulenceModel=turbulence_model)]:

        res = wake_model(x, y)
        # print(np.round(res.TI_eff_ilk[:, 0, 0], 3).tolist())
        npt.assert_array_almost_equal(res.TI_eff_ilk[:, 0, 0], ref_ti, 3)


def test_superposition_model_indices():
    class WTSite(UniformSite):
        def local_wind(self, x_i=None, y_i=None, h_i=None, wd=None, ws=None, wd_bin_size=None, ws_bins=None):
            lw = UniformSite.local_wind(self, x_i=x_i, y_i=y_i, h_i=h_i, wd=wd, ws=ws,
                                        wd_bin_size=wd_bin_size, ws_bins=ws_bins)
            lw.TI_ilk += np.arange(len(x_i))[:, np.newaxis, np.newaxis] * .1
            return lw

    site = WTSite([1], 0.1)

    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]

    # WS_ilk different at each wt position
    TI_ilk = site.local_wind(x_i, y_i, h_i, wd=0, ws=8.1).TI_ilk
    npt.assert_array_almost_equal(TI_ilk, np.reshape([0.1, 0.2, 0.3], (3, 1, 1)), 10)

    def get_wf_model(cls):
        return cls(site, NibeA0, wake_deficitModel=NoWakeDeficit(),
                   superpositionModel=LinearSum(),
                   turbulenceModel=STF2017TurbulenceModel())
    for wake_model in [get_wf_model(PropagateDownwind),
                       get_wf_model(All2AllIterative)]:

        # No wake (ct = 0), i.e. WS_eff == WS
        TI_eff_ilk = wake_model.calc_wt_interaction(x_i, y_i, h_i, [1, 1, 1], 0.0, 8.1)[1]
        npt.assert_array_equal(TI_eff_ilk, TI_ilk)

        # full wake (CT=8/9)
        ref_TI_eff_ilk = TI_ilk + np.reshape([0, 0.33738364, np.sum([0.19369135, 0.21239116])], (3, 1, 1))

        TI_eff_ilk = wake_model.calc_wt_interaction(x_i, y_i, h_i, [0, 0, 0], 0.0, 8.1)[1]
        npt.assert_array_almost_equal(TI_eff_ilk, ref_TI_eff_ilk)

        sim_res = wake_model(x_i, y_i, h_i, [0, 0, 0], 0.0, 8.1)
        TI_eff_ilk = sim_res.flow_map(HorizontalGrid(x=[0], y=y_i, h=50)).TI_eff_xylk[:, 0]

        npt.assert_array_almost_equal(TI_eff_ilk, ref_TI_eff_ilk)
