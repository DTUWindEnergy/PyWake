from py_wake import IEA37SimpleBastankhahGaussian
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.examples.data.iea37._iea37 import IEA37Site
import numpy as np
import matplotlib.pyplot as plt
import pytest
from py_wake.flow_map import XYGrid
from py_wake.deflection_models.fuga_deflection import FugaDeflection
from py_wake.tests import npt
from py_wake.examples.data.hornsrev1 import V80
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.utils.model_utils import get_models
from py_wake.tests.test_files import tfp
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit


@pytest.mark.parametrize('deflectionModel,dy10d', [
    (JimenezWakeDeflection, 0.5672964),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00/')), 0.4625591892703828),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/')), 0.37719329354768527),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/')), 0.32787746772608933),
])
def test_deflection_model_dy10d(deflectionModel, dy10d):
    # center line deflection 10d downstream
    site = IEA37Site(16)
    x, y = [0], [0]
    windTurbines = V80()
    D = windTurbines.diameter()
    wfm = IEA37SimpleBastankhahGaussian(site, windTurbines, deflectionModel=deflectionModel())

    yaw_ilk = np.reshape([-30], (1, 1, 1))

    sim_res = wfm(x, y, yaw=yaw_ilk, wd=270, ws=10)
    fm = sim_res.flow_map(XYGrid(x=np.arange(-D, 10 * D + 10, 10)))
    min_WS_line = fm.min_WS_eff()
    if 0:
        plt.figure(figsize=(14, 3))
        fm.plot_wake_map()
        min_WS_line.plot()
        plt.plot(10 * D, dy10d * D, '.', label="Ref, 10D")
        plt.legend()
        plt.show()

    npt.assert_almost_equal(min_WS_line.interp(x=10 * D).item() / D, dy10d)


@pytest.mark.parametrize('deflectionModel,dy', [
    (JimenezWakeDeflection,
     [2.0, 12.0, 20.0, 26.0, 2.0, -5.0, -11.0, -16.0, -0.0, 8.0, 15.0, 20.0]),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00/')),
     [1.0, 6.0, 12.0, 18.0, 2.0, -0.0, -4.0, -7.0, -1.0, 2.0, 4.0, 7.0]),
    # ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/')), 0.37719329354768527),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/')),
     [1.0, 6.0, 11.0, 15.0, 2.0, -0.0, -3.0, -5.0, -1.0, 2.0, 4.0, 6.0]),
])
def test_deflection_model(deflectionModel, dy):
    # center line deflection 10d downstream
    site = IEA37Site(16)
    x, y = [0, 400, 800], [0, 0, 0]
    windTurbines = V80()
    D = windTurbines.diameter()
    wfm = PropagateDownwind(site, windTurbines, IEA37SimpleBastankhahGaussianDeficit(),
                            deflectionModel=deflectionModel())

    yaw = [-30, 30, -30]

    sim_res = wfm(x, y, yaw=yaw, wd=270, ws=10)
    fm = sim_res.flow_map(XYGrid(x=np.arange(-D, 15 * D + 10, 10)))
    min_WS_line = fm.min_WS_eff()
    if 0:
        plt.figure(figsize=(14, 3))
        fm.plot_wake_map()
        min_WS_line[::10].plot(ls='-', marker='.')
        print(np.round(min_WS_line.values[::10][1:]).tolist())
        plt.legend()
        plt.show()

    npt.assert_almost_equal(min_WS_line.values[::10][1:], dy, 0)


@pytest.mark.parametrize('deflectionModel,dy', [
    (JimenezWakeDeflection,
     [2.0, 12.0, 20.0, 26.0, 2.0, -5.0, -11.0, -16.0, -0.0, 8.0, 15.0, 20.0]),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00/')),
     [1.0, 6.0, 12.0, 18.0, 2.0, -0.0, -4.0, -7.0, -1.0, 2.0, 4.0, 7.0]),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/')),
     [1.0, 6.0, 11.0, 15.0, 2.0, -0.0, -3.0, -5.0, -1.0, 2.0, 4.0, 6.0]),
])
def test_deflection_model_All2AllIterative(deflectionModel, dy):
    # center line deflection 10d downstream
    site = IEA37Site(16)
    x, y = [0, 400, 800], [0, 0, 0]
    windTurbines = V80()
    D = windTurbines.diameter()
    wfm = All2AllIterative(site, windTurbines, IEA37SimpleBastankhahGaussianDeficit(),
                           deflectionModel=deflectionModel())

    yaw = [-30, 30, -30]

    sim_res = wfm(x, y, yaw=yaw, wd=270, ws=10)
    fm = sim_res.flow_map(XYGrid(x=np.arange(-D, 15 * D + 10, 10)))
    min_WS_line = fm.min_WS_eff()
    if 0:
        plt.figure(figsize=(14, 3))
        fm.plot_wake_map()
        min_WS_line[::10].plot(ls='-', marker='.')
        print(np.round(min_WS_line.values[::10][1:]).tolist())
        plt.legend()
        plt.show()

    npt.assert_almost_equal(min_WS_line.values[::10][1:], dy, 0)


@pytest.mark.parametrize('deflectionModel', [m for m in get_models(DeflectionModel) if m is not None])
def test_plot_deflection_grid(deflectionModel):
    site = IEA37Site(16)
    x, y = [0], [0]
    windTurbines = V80()
    D = windTurbines.diameter()
    wfm = IEA37SimpleBastankhahGaussian(site, windTurbines, deflectionModel=deflectionModel())

    yaw_ilk = np.reshape([-30], (1, 1, 1))

    sim_res = wfm(x, y, yaw=yaw_ilk, wd=270, ws=10)
    fm = sim_res.flow_map(XYGrid(x=np.arange(-D, 10 * D + 10, 10)))

    plt.figure(figsize=(14, 3))
    fm.plot_wake_map()
    fm.plot_deflection_grid()
    min_WS_line = fm.min_WS_eff()
    min_WS_line.plot()
    plt.legend()
    plt.title(wfm.deflectionModel)
    if 0:
        plt.show()
    plt.close('all')
