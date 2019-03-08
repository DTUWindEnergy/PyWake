"""Verify the IEA 37 models have correct AEP values in file

Comparison of pw_wake and IEA 37 code is in test_wake_models
"""
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37AEPCalc
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm
from py_wake.tests import npt


def test_iea37_aep_file():
    """Compare AEP values in file to those calculated by task 37 code"""
    n_wts = [9, 16, 36, 64]  # diff wind farm sizes
    for n_wt in n_wts:
        iea37_aep = IEA37AEPCalc(n_wt).get_aep()  # task 37's code
        _, _, file_aep = read_iea37_windfarm(iea37_path +
                                             'iea37-ex%d.yaml' % n_wt)
        npt.assert_almost_equal(file_aep[0], iea37_aep.sum(), 5)
        npt.assert_almost_equal(file_aep[1], iea37_aep, 5)
