import numpy as np
from py_wake.aep._aep import AEP
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windrose,\
    read_iea37_windfarm
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.wake_models.gaussian import IEA37SimpleBastankhahGaussian


def test_IEA37SimpleBastankhahGaussian_ex16():
    _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
    n_wt = 16
    x, y, aep_ref = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(x, y, '2k')
        for i, (x_, y_) in enumerate(zip(x, y)):
            plt.annotate(i, (x_, y_))
        plt.axis('equal')
        plt.show()
    site = UniformSite(freq, ti=0.75)
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
    wake_model = IEA37SimpleBastankhahGaussian(windTurbines)

    aep = AEP(site, windTurbines, wake_model, np.arange(0, 360, 22.5), [9.8])
    aep_ilk = aep.calculate_AEP(x, y)
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000
    npt.assert_almost_equal(aep_ref[0], aep_MW_l.sum(), 5)
    npt.assert_array_almost_equal(aep_ref[1], aep_MW_l, 5)


def test_wake_map():
    _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
    n_wt = 16
    x, y, _ = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)
    site = UniformSite(freq, ti=0.75)
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
    wake_model = IEA37SimpleBastankhahGaussian(windTurbines)
    aep = AEP(site, windTurbines, wake_model, np.arange(0, 360, 22.5), [9.8])
    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)
    X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[0], ws=[9])

    ref = [3.32, 4.86, 7.0, 8.1, 7.8, 7.23, 6.86, 6.9, 7.3, 7.82, 8.11, 8.04, 7.87, 7.79, 7.85, 8.04, 8.28]
    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)
    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z, np.arange(2, 9.1, .01))
        plt.colorbar(c)
        plt.plot(x, y, '2k')
        for i, (x_, y_) in enumerate(zip(x, y)):
            plt.annotate(i, (x_, y_))
        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    test_IEA37SimpleBastankhahGaussian_ex8()
