import numpy as np
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm
from py_wake.tests import npt
from py_wake import IEA37SimpleBastankhahGaussian
import matplotlib.pyplot as plt


def test_IEA37SimpleBastankhahGaussian_all_ex():
    """check SBG matches IEA37 value in file"""
    for n_wt in [9, 16, 36, 64]:
        x, y, aep_ref = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)
        if 0:
            plt.plot(x, y, '2k')
            for i, (x_, y_) in enumerate(zip(x, y)):
                plt.annotate(i, (x_, y_))
            plt.axis('equal')
            plt.show()
        site = IEA37Site(n_wt)
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
        wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)

        aep_ilk = wake_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
        aep_MW_l = aep_ilk.sum((0, 2)) * 1000
        # test that the result is equal to results provided for IEA task 37
        npt.assert_almost_equal(aep_ref[0], aep_MW_l.sum(), 5)
        npt.assert_array_almost_equal(aep_ref[1], aep_MW_l, 5)
