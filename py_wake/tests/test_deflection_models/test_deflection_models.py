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


@pytest.mark.parametrize('deflectionModel,dy10d', [
    (JimenezWakeDeflection, 0.5672964),
    ((lambda: FugaDeflection(tfp + 'fuga/D080.0000_zH070.0000/Z0=0.03000000Zi=00401Zeta0=0.00E+00/')), 0.2567786526168626),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00014617Zi=00399Zeta0=0.00E+0/')), 0.33216633496287334),
])
def test_deflection_model(deflectionModel, dy10d):
    site = IEA37Site(16)
    x, y = [0], [0]
    windTurbines = V80()
    D = windTurbines.diameter()
    wfm = IEA37SimpleBastankhahGaussian(site, windTurbines, deflectionModel=deflectionModel())

    yaw_ilk = np.reshape([-30], (1, 1, 1))

    sim_res = wfm(x, y, yaw_ilk=yaw_ilk, wd=270, ws=10)
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


@pytest.mark.parametrize('deflectionModel', [m for m in get_models(DeflectionModel) if m is not None])
def test_plot_deflection_grid(deflectionModel):
    site = IEA37Site(16)
    x, y = [0], [0]
    windTurbines = V80()
    D = windTurbines.diameter()
    wfm = IEA37SimpleBastankhahGaussian(site, windTurbines, deflectionModel=deflectionModel())

    yaw_ilk = np.reshape([-30], (1, 1, 1))

    sim_res = wfm(x, y, yaw_ilk=yaw_ilk, wd=270, ws=10)
    fm = sim_res.flow_map(XYGrid(x=np.arange(-D, 10 * D + 10, 10)))

    plt.figure(figsize=(14, 3))
    fm.plot_wake_map()
    fm.plot_deflection_grid()
    min_WS_line = fm.min_WS_eff()
    min_WS_line.plot()
    plt.legend()
    if 0:
        plt.show()
    plt.close()
