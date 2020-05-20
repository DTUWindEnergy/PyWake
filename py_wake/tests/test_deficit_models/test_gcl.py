import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models.gcl import GCLDeficitModel, get_dU, get_Rw, GCL
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.flow_map import HorizontalGrid
from py_wake.superposition_models import LinearSum
from py_wake.tests import npt
from py_wake.turbulence_models.gcl import GCLTurbulenceModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.tests.check_speed import timeit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80


def test_GCL_ex16():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wfm = GCL(site, windTurbines)
    if 0:
        windTurbines.plot(x, y)
        plt.show()
    aep_ilk = wfm(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)

    aep_MW_l = aep_ilk.sum((0, 2)) * 1000
    # test that the result is equal to last run (no evidens that  these number are correct)
    aep_ref = (360126.40388393076,
               [9105.55989, 8528.06167, 11216.59308, 13878.00696, 20174.90212,
                25057.51256, 38677.90718, 43350.98015, 22946.01093, 13572.39417,
                14919.85229, 31933.39927, 68967.56313, 17698.02851, 12241.93008,
                7857.70189])
    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)

    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


def test_GCL_ex80():
    site = Hornsrev1Site()

    x, y = site.initial_position.T
    windTurbines = V80()
    wfm = PropagateDownwind(
        site,
        windTurbines,
        wake_deficitModel=GCLDeficitModel(),
        superpositionModel=LinearSum())
    if 0:
        windTurbines.plot(x, y)
        plt.show()

    if 0:
        sim_res = timeit(wfm)(x, y, ws=np.arange(10, 15))[0]

        def run():
            wfm(x, y, ws=np.arange(10, 15))

        from line_profiler import LineProfiler
        lp = LineProfiler()
        lp.timer_unit = 1e-6
        lp.add_function(GCLDeficitModel.calc_deficit)
        lp.add_function(get_dU)
        lp.add_function(get_Rw)
        lp_wrapper = lp(run)
        res = lp_wrapper()
        lp.print_stats()
    else:
        sim_res = wfm(x, y, ws=np.arange(10, 15))

    # test that the result is equal to previuos runs (no evidens that  these number are correct)
    aep_ref = 1055.956615887197
    npt.assert_almost_equal(sim_res.aep_ilk(normalize_probabilities=True).sum(), aep_ref, 5)

    sim_res = wfm(x, y, ws=np.arange(3, 10))
    npt.assert_array_almost_equal(sim_res.aep_ilk(normalize_probabilities=True).sum(), 261.6143039016946, 5)


def test_GCLSimple_wake_map():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=GCLDeficitModel(),
                            superpositionModel=LinearSum())
    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)

    flow_map = wfm(x, y, wd=0, ws=9).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map.X, flow_map.Y
    Z = flow_map.WS_eff_xylk[:, :, 0, 0]

    # reference result from previuos run (no evidens that  these number are correct)
    ref = [2.39, 4.93, 7.46, 8.34, 7.95, 7.58, 7.26, 7.2, 7.37, 7.56, 7.74, 7.86, 7.89, 7.83, 7.92, 8.1, 8.3]
    if 0:

        print(np.round(Z[49, 100:133:2], 2).tolist())
        flow_map.plot_wake_map()
        plt.plot(X[49, 100:133:2], Y[49, 100:133:2], '.-')
        plt.figure()
        plt.plot(Z[49, 100:133:2])
        plt.plot(ref)
        plt.show()

    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


def test_wake_radius():
    n = 5
    npt.assert_array_almost_equal(GCLDeficitModel().wake_radius(
        D_src_il=np.reshape([100, 50, 100, 100, 100], (n, 1)),
        dw_ijlk=np.reshape([500, 500, 1000, 500, 500], (n, 1, 1, 1)),
        ct_ilk=np.reshape([.8, .8, .8, .4, .8], (n, 1, 1)),
        TI_ilk=np.reshape([.1, .1, .1, .1, .2], (n, 1, 1)))[:, 0, 0, 0],
        [156.949964, 97.763333, 195.526667, 113.225695, 250.604162])

    # Check that it works when called from WindFarmModel
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=GCLDeficitModel(),
                            superpositionModel=LinearSum(), turbulenceModel=GCLTurbulenceModel())
    wfm(x=[0, 500], y=[0, 0], wd=[30], ws=[10])
