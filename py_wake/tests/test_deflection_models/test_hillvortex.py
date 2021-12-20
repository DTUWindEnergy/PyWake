import numpy as np
import matplotlib.pyplot as plt
from py_wake.deficit_models.gaussian import BastankhahGaussian, ZongGaussian, ZongGaussianDeficit
from py_wake.deflection_models import GCLHillDeflection
from py_wake.examples.data.hornsrev1 import V80
from py_wake.flow_map import XYGrid
from py_wake.site.xrsite import UniformSite
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.tests import npt
from py_wake.tests.check_speed import timeit


def test_torque_result():
    """Reproduce case from

    Larsen, G. C., Ott, S., Liew, J., van der Laan, M. P., Simon, E., R.Thorsen, G., & Jacobs, P. (2020).
    Yaw induced wake deflection - a full-scale validation study.
    Journal of Physics - Conference Series, 1618, [062047].
    https://doi.org/10.1088/1742-6596/1618/6/062047

    Note that the implementation used in the paper differs from the actual implementation by:
    - using rotor average deficit instead of peak deficit
    - using ainslie deficit model + static "meander distribution"
    - dismiss Ua (downwind distance reduction) term in integration formula
    Hence the result is not expected to match
    """
    site = UniformSite(p_wd=[1], ti=0.06)
    x, y = [0], [0]

    def power_ct_function(ws, yaw, run_only):
        return (np.zeros_like(ws), np.where(yaw == 17.5, .86, 0.83))[run_only]
    D = 52
    v52 = WindTurbine(name="V52", diameter=D, hub_height=44,
                      powerCtFunction=PowerCtFunction(input_keys=['ws', 'yaw'],
                                                      power_ct_func=power_ct_function,
                                                      power_unit='w', additional_models=[]))

    wfm = ZongGaussian(site, v52,
                       deflectionModel=GCLHillDeflection(),
                       turbulenceModel=CrespoHernandez())
    x_ref = np.arange(2, 12, 2)
    positive_ref = [-0.2, -.35, -.45, -.55, -.65]
    negative_ref = [.15, .3, .4, .45, .5]
    for ws, yaw, ref in [(9.48, 17.5, positive_ref), (9.73, -14.5, negative_ref)]:
        plt.figure(figsize=(12, 3))

        grid = XYGrid(x=np.linspace(-2 * D, D * 10, 100), y=np.linspace(-1.5 * D, 1.5 * D, 100))
        fm = wfm(x, y, yaw=yaw, wd=270, ws=ws).flow_map(grid)
        fm.plot_wake_map(normalize_with=D)
        center_line = fm.min_WS_eff()
        plt.title(f'Yaw {yaw}deg')
        plt.plot(center_line.x / D, center_line / D, label='Centerline')
        plt.plot(x_ref, ref)
        npt.assert_allclose(center_line.interp(x=x_ref * D) / D, ref, atol=.1)

        plt.grid()
        plt.legend()
    if 0:
        plt.show()


def test_N():

    site = UniformSite(p_wd=[1], ti=0.06)
    x, y = [0], [0]
    wt = V80()
    D = wt.diameter()

    plt.figure(figsize=(12, 3))
    grid = XYGrid(x=[10 * D, 20 * D], y=np.linspace(-1.5 * D, 1.5 * D, 100))

    def deflection_20d(N):
        wfm = ZongGaussian(site, wt, deflectionModel=GCLHillDeflection(N=N),
                           turbulenceModel=CrespoHernandez())
        fm = wfm(x, y, yaw=30, wd=270, ws=10).flow_map(grid)
        return fm.min_WS_eff().values[-1]

    N_lst = [10, 20, 100]
    res = [timeit(deflection_20d, min_runs=10)(N) for N in N_lst]
    plt.plot(N_lst, [d for d, _ in res])
    plt.ylabel('Deflection 20D downstream [m]')
    plt.xlabel('N')
    ax = plt.twinx()
    ax.plot(N_lst, [np.mean(t) for _, t in res], '--')
    ax.set_ylabel('Time [s]')

    assert np.abs(res[-1][0] - res[1][0]) < .3  # Mismatch between N=20 and 100 is less than 30cm 20D downstream
    if 0:
        plt.show()


def test_wake_deficitModel_input():
    site = UniformSite(p_wd=[1], ti=0.06)
    x, y = [0], [0]

    wt = V80()
    D = wt.diameter()
    wfm1 = ZongGaussian(site, wt, deflectionModel=GCLHillDeflection(),
                        turbulenceModel=CrespoHernandez())
    wfm2 = BastankhahGaussian(site, wt,
                              deflectionModel=GCLHillDeflection(wake_deficitModel=ZongGaussianDeficit()),
                              turbulenceModel=CrespoHernandez())

    grid = XYGrid(x=np.linspace(10, D * 10, 100), y=np.linspace(-1.5 * D, 1.5 * D, 100))
    cl1, cl2 = [wfm(x, y, yaw=30, wd=270, ws=10).flow_map(grid).min_WS_eff() for wfm in [wfm1, wfm2]]
    npt.assert_array_almost_equal(cl1, cl2, 3)
