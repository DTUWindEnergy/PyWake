import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models.gcl import GCLDeficit, get_dU, get_Rw
from py_wake.superposition_models import LinearSum
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.tests.check_speed import timeit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80


def test_GCL_ex80():
    site = Hornsrev1Site()

    x, y = site.initial_position.T
    windTurbines = V80()
    wfm = PropagateDownwind(
        site,
        windTurbines,
        wake_deficitModel=GCLDeficit(),
        superpositionModel=LinearSum())
    if 0:
        windTurbines.plot(x, y)
        plt.show()

    sim_res = timeit(wfm.__call__, line_profile=0, profile_funcs=[get_dU], verbose=0)(x, y, ws=np.arange(10, 15))[0]

    # test that the result is equal to previuos runs (no evidens that  these number are correct)
    aep_ref = 1055.956615887197
    npt.assert_almost_equal(sim_res.aep_ilk(normalize_probabilities=True).sum(), aep_ref, 5)

    sim_res = wfm(x, y, ws=np.arange(3, 10))
    npt.assert_array_almost_equal(sim_res.aep_ilk(normalize_probabilities=True).sum(), 261.6143039016946, 5)
