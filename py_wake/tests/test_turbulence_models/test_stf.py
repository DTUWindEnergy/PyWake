import pytest
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.tests import npt
from py_wake.turbulence_models.stf import STF2005TurbulenceModel, STF2017TurbulenceModel
from py_wake import NOJ
import numpy as np
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.superposition_models import SquaredSum


@pytest.mark.parametrize('turbulence_model,ref_ti',
                         [(STF2005TurbulenceModel(), [0.075, 0.075, 0.075, 0.104, 0.149, 0.114, 0.075, 0.075,
                                                      0.075, 0.075, 0.075, 0.075, 0.104, 0.104, 0.088, 0.104]),
                          (STF2017TurbulenceModel(), [0.075, 0.075, 0.075, 0.114, 0.171, 0.129, 0.075, 0.075,
                                                      0.075, 0.075, 0.075, 0.075, 0.115, 0.114, 0.094, 0.115])])
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
