from py_wake.examples.data.hornsrev1 import V80, wt_x, wt_y, wt9_x, wt9_y,\
    Hornsrev1Site
from py_wake.aep_calculator import AEPCalculator
from py_wake.wake_models.noj import NOJ
import numpy as np
from py_wake.tests import npt


def test_from_WAsP_wtg():
    import os
    from py_wake.examples.data import wtg_path
    from py_wake.wind_turbines import WindTurbines
    vestas_v80_wtg = os.path.join(wtg_path, 'Vestas-V80.wtg')
    NEG_2750_wtg = os.path.join(wtg_path, 'NEG-Micon-2750.wtg')
    wts_wtg = WindTurbines.from_WAsP_wtg([vestas_v80_wtg, NEG_2750_wtg])

    assert(wts_wtg.name(types=0) == 'Vestas V80 (2MW, Offshore)')
    assert(wts_wtg.diameter(types=0) == 80)
    assert(wts_wtg.hub_height(types=0) == 67)
    npt.assert_array_equal(wts_wtg.power(np.array([0, 3, 5, 9, 18, 26]), type_i=0), np.array([0, 0, 154000, 996000, 2000000, 0]))
    npt.assert_array_equal(wts_wtg.ct(np.array([1, 4, 7, 9, 17, 27]), type_i=0), np.array([0, 0.818, 0.805, 0.807, 0.167, 0]))

    assert(wts_wtg.name(types=1) == 'NEG-Micon 2750/92 (2750 kW)')
    assert(wts_wtg.diameter(types=1) == 92)
    assert(wts_wtg.hub_height(types=1) == 70)
    npt.assert_array_equal(wts_wtg.power(np.array([0, 3, 5, 9, 18, 26]), type_i=1), np.array([0, 0, 185000, 1326000, 2748000, 0]))
    npt.assert_array_equal(wts_wtg.ct(np.array([1, 4, 7, 9, 17, 27]), type_i=1), np.array([0, 0.871, 0.841, 0.797, 0.175, 0]))


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

    aep_calculator = AEPCalculator(NOJ(Hornsrev1Site(), wts))
    npt.assert_almost_equal(aep_calculator.calculate_AEP(wt9_x, wt9_y, type_i=types0).sum(), 81.2066072392765)
    npt.assert_almost_equal(aep_calculator.calculate_AEP(wt9_x, wt9_y, type_i=types1).sum(), 83.72420504573488)
    npt.assert_almost_equal(aep_calculator.calculate_AEP(wt9_x, wt9_y, type_i=types2).sum(), 88.87227386796884

                            )
    if 0:
        plt.show()


def test_get_defaults():
    v80 = V80()
    npt.assert_array_equal(np.array(v80.get_defaults(1))[:, 0], [0, 70, 80])
    npt.assert_array_equal(np.array(v80.get_defaults(1, h_i=100))[:, 0], [0, 100, 80])
    npt.assert_array_equal(np.array(v80.get_defaults(1, d_i=100))[:, 0], [0, 70, 100])
