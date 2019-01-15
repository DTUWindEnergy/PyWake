from py_wake.examples.data.hornsrev1 import V80, wt_x, wt_y, wt9_x, wt9_y,\
    Hornsrev1Site
from py_wake.aep_calculator import AEPCalculator
from py_wake.wake_models.noj import NOJ
import numpy as np
from py_wake.tests import npt


def test_twotype_windturbines():
    from py_wake.wind_turbines import WindTurbines
    v80 = V80()

    def power(ws, types):
        power = v80.power(ws)
        # add 10% type 1 turbines
        power[types == 1] *= 1.1
        return power

    wts = WindTurbines(names=['V80', 'V88'],
                       diameters=[80, 88],
                       hub_heights=[70, 77],
                       ct_funcs=[v80.ct_funcs[0],
                                 v80.ct_funcs[0]],
                       power_funcs=[v80.power,
                                    lambda ws:v80.power(ws) * 1.1],
                       power_unit='w'
                       )

    import matplotlib.pyplot as plt
    types0 = [0] * 9
    types1 = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    types2 = [1] * 9
    wts.plot(wt9_x, wt9_y, types1)

    aep_calculator = AEPCalculator(Hornsrev1Site(), wts, NOJ(wts))
    npt.assert_almost_equal(aep_calculator.calculate_AEP(wt9_x, wt9_y, type_i=types0).sum(), 81.2066072392765)
    npt.assert_almost_equal(aep_calculator.calculate_AEP(wt9_x, wt9_y, type_i=types1).sum(), 83.72420504573488)
    npt.assert_almost_equal(aep_calculator.calculate_AEP(wt9_x, wt9_y, type_i=types2).sum(), 88.87227386796884

                            )
    if 0:
        plt.show()
