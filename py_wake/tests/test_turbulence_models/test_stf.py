from py_wake.aep_calculator import AEPCalculator
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.turbulence_models.stf import NOJ_STF2005, NOJ_STF2017
import numpy as np
import pytest
from py_wake.tests import npt


@pytest.mark.parametrize('WakeModel,ref_ti',
                         [(NOJ_STF2005, [0.075, 0.075, 0.075, 0.104, 0.149, 0.114, 0.075, 0.075,
                                         0.075, 0.075, 0.075, 0.075, 0.104, 0.104, 0.088, 0.104]),
                          (NOJ_STF2017, [0.075, 0.075, 0.075, 0.114, 0.171, 0.129, 0.075, 0.075,
                                         0.075, 0.075, 0.075, 0.075, 0.115, 0.114, 0.094, 0.115])])
def test_stf(WakeModel, ref_ti):
    # setup site, turbines and wakemodel
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wake_model = WakeModel(site, windTurbines)
    aep_calculator = AEPCalculator(wake_model)
    aep_calculator.calculate_AEP(x, y)
    # print(np.round(aep_calculator.TI_eff_ilk[:, 0, 0], 3).tolist())
    npt.assert_array_almost_equal(aep_calculator.TI_eff_ilk[:, 0, 0], ref_ti, 3)
