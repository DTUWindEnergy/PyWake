from py_wake.turbulence_models.turbulence_model import MaxSum
from py_wake.tests import npt


def test_max_sum():
    maxSum = MaxSum()
    npt.assert_array_equal(maxSum.calc_effective_TI(TI_xxx=[.2, .4], add_turb_jxxx=[[0.1, .4], [.3, .2]]), [.5, .8])
