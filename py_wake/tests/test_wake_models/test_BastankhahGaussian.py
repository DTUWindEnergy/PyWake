import numpy as np
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windrose,\
    read_iea37_windfarm
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.wake_models.gaussian import IEA37SimpleBastankhahGaussian,\
    BastankhahGaussian
from py_wake.aep_calculator import AEPCalculator


def test_BastankhahGaussian_ex16():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
    wake_model = BastankhahGaussian(site, windTurbines)
    if 0:
        import matplotlib.pyplot as plt
        windTurbines.plot(x, y)
        plt.show()

    aep = AEPCalculator(wake_model)
    aep_ilk = aep.calculate_AEP(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8])
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000
    # test that the result is equal to last run (no evidens that  these number are correct)
    aep_ref = (355971.9717035484,
               [9143.74048, 8156.71681, 11311.92915, 13955.06316, 19807.65346,
                25196.64182, 39006.65223, 41463.31044, 23042.22602, 12978.30551,
                14899.26913, 32320.21637, 67039.04091, 17912.40907, 12225.04134,
                7513.75582])
    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)

    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


def test_BastankhahGaussian_wake_map():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

    wake_model = BastankhahGaussian(site, windTurbines)
    aep = AEPCalculator(wake_model)
    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)
    X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[0], ws=[9])

    # test that the result is equal to last run (no evidens that  these number are correct)
    ref = [0.18, 3.6, 7.27, 8.32, 7.61, 6.64, 5.96, 6.04, 6.8, 7.69, 8.08, 7.87, 7.59, 7.46, 7.55, 7.84, 8.19]
    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z, np.arange(-.25, 9.1, .01))
        plt.colorbar(c)
        windTurbines.plot(x, y)
        plt.show()
    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


def test_IEA37SimpleBastankhahGaussian_ex16():
    """check SBG matches IEA37 value in file"""
    _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
    for n_wt in [9, 16, 36, 64]:
        x, y, aep_ref = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(x, y, '2k')
            for i, (x_, y_) in enumerate(zip(x, y)):
                plt.annotate(i, (x_, y_))
            plt.axis('equal')
            plt.show()
        site = IEA37Site(n_wt)
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
        wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)

        aep = AEPCalculator(wake_model)
        aep_ilk = aep.calculate_AEP(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8])
        aep_MW_l = aep_ilk.sum((0, 2)) * 1000
        # test that the result is equal to results provided for IEA task 37
        npt.assert_almost_equal(aep_ref[0], aep_MW_l.sum(), 5)
        npt.assert_array_almost_equal(aep_ref[1], aep_MW_l, 5)


def test_IEA37SimpleBastankhahGaussian_wake_map():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
    wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
    aep = AEPCalculator(wake_model)
    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)
    X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[0], ws=[9])

    # test that the result is equal to last run (no evidens that  these number are correct)
    ref = [3.32, 4.86, 7.0, 8.1, 7.8, 7.23, 6.86, 6.9, 7.3, 7.82, 8.11, 8.04, 7.87, 7.79, 7.85, 8.04, 8.28]
    if 0:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z, np.arange(2, 9.1, .01))
        plt.colorbar(c)
        windTurbines.plot(x, y)
        plt.plot(X[49, 100:133:2], Y[49, 100:133:2], '-.')
        plt.show()
    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)
