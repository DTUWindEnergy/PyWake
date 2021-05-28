from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models.fuga import FugaYawDeficit
from py_wake.flow_map import XYGrid
from py_wake.site.xrsite import UniformSite
from py_wake.tests.test_files import tfp
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.tests import npt
from py_wake.deflection_models.fuga_deflection import FugaDeflection
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site


def test_fuga_deflection_vs_notebook():

    p = Path(tfp) / "fuga/v80_deflection_x.csv"
    x, notebook_deflection = np.array([v.split(",") for v in p.read_text().strip().split("\n")], dtype=float).T

    path = tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00'
    fuga_deflection = -FugaDeflection(path).calc_deflection(dw_ijl=np.reshape(x, (1, 41, 1)),
                                                            hcw_ijl=np.reshape(x * 0, (1, 41, 1)),
                                                            dh_ijl=np.reshape(x * 0, (1, 41, 1)),
                                                            WS_ilk=np.array([[[9.5]]]),
                                                            WS_eff_ilk=np.array([[[9.5]]]),
                                                            yaw_ilk=np.array([[[17.4493]]]),
                                                            ct_ilk=np.array([[[0.850877]]]) *
                                                            np.cos(0.30454774)**2,
                                                            D_src_il=np.array([[80]]))[1].squeeze()

    if 0:
        plt.plot(x, notebook_deflection, label='Notebook deflection')
        plt.plot(x, fuga_deflection)
        plt.show()
    plt.close('all')
    npt.assert_allclose(fuga_deflection, notebook_deflection, atol=1e-5)


def test_fuga_wake_center_vs_notebook():

    p = Path(tfp) / "fuga/v80_wake_center_x.csv"
    x, notebook_wake_center = np.array([v.split(",") for v in p.read_text().strip().split("\n")], dtype=float).T

    powerCtFunction = PowerCtTabular([0, 100], [0, 0], 'w', [0.850877, 0.850877])
    wt = WindTurbine(name='', diameter=80, hub_height=70, powerCtFunction=powerCtFunction)

    path = tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00'
    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wfm = PropagateDownwind(
        site,
        wt,
        wake_deficitModel=FugaYawDeficit(path),
        deflectionModel=FugaDeflection(path, 'input_par')
    )

    WS = 10
    sim_res = wfm([0], [0], yaw=[17.4493], wd=270, ws=[WS])
    y = wfm.wake_deficitModel.mirror(wfm.wake_deficitModel.y, anti_symmetric=True)
    fm = sim_res.flow_map(XYGrid(x=x[1:], y=y[240:271]))
    fuga_wake_center = [np.interp(0, InterpolatedUnivariateSpline(ws.y, ws.values).derivative()(ws.y), ws.y)
                        for ws in fm.WS_eff.squeeze().T]

    if 0:
        plt.plot(x, notebook_wake_center, label='Notebook deflection')
        plt.plot(x[1:], fuga_wake_center)
        plt.show()
    plt.close('all')
    npt.assert_allclose(fuga_wake_center, notebook_wake_center[1:], atol=.14)


def test_fuga_deflection_time_series_gradient_evaluation():

    p = Path(tfp) / "fuga/v80_wake_center_x.csv"
    x, notebook_wake_center = np.array([v.split(",") for v in p.read_text().strip().split("\n")], dtype=float).T

    powerCtFunction = PowerCtTabular([0, 100], [0, 0], 'w', [0.850877, 0.850877])
    wt = WindTurbine(name='', diameter=80, hub_height=70, powerCtFunction=powerCtFunction)

    path = tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00'
    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wfm = PropagateDownwind(
        site,
        wt,
        wake_deficitModel=FugaYawDeficit(path),
        deflectionModel=FugaDeflection(path, 'input_par')
    )

    WS = 10

    yaw_ref = np.full((10, 1), 17)
    yaw_step = np.eye(10, 10) * 1e-6 + yaw_ref
    yaw = np.concatenate([yaw_step, yaw_ref], axis=1)
    sim_res = wfm(np.arange(10) * wt.diameter() * 4, [0] * 10, yaw=yaw, wd=[270] * 11, ws=[WS] * 11, time=True)
    print(sim_res)
