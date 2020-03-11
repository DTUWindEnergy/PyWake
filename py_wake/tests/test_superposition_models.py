import pytest
import numpy as np
from py_wake import NOJ
from py_wake.site._site import UniformSite
from py_wake.superposition_models import LinearSum, SquaredSum, MaxSum
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines


# Two turbines, 0: Nibe-A, 1:Ct=0
NibeA0 = WindTurbines(names=['Nibe-A'] * 2, diameters=[40] * 2,
                      hub_heights=[50] * 2,
                      ct_funcs=[lambda _: 8 / 9, lambda _: 0],
                      power_funcs=[lambda _: 0] * 2, power_unit='w')

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
