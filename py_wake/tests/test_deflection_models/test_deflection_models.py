from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake import np
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
from py_wake.superposition_models import SquaredSum
from py_wake.deficit_models.fuga import FugaBlockage
from py_wake.site._site import UniformSite


def simple_wfm(deflectionModel):
    site = IEA37Site(16)
    windTurbines = V80()
    return PropagateDownwind(site, windTurbines, wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                             superpositionModel=SquaredSum(), deflectionModel=deflectionModel())


@pytest.mark.parametrize('deflectionModel,dy10d', [
    (JimenezWakeDeflection, 0.5687187382677713),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00.nc')), 0.4625591892703828),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc')), 0.37719329354768527),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc')), 0.32787746772608933),
])
def test_deflection_model_dy10d(deflectionModel, dy10d):
    # center line deflection 10d downstream
    site = IEA37Site(16)
    x, y = [0], [0]
    windTurbines = V80()
    D = windTurbines.diameter()

    wfm = simple_wfm(deflectionModel)
    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                            superpositionModel=SquaredSum(), deflectionModel=deflectionModel())

    yaw_ilk = np.reshape([-30], (1, 1, 1))

    sim_res = wfm(x, y, yaw=yaw_ilk, tilt=0, wd=270, ws=10)
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
     [2., 12., 20., 26., -1., -10., -18., -24., 1., 11., 18., 25.]),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00.nc')),
     [1., 6., 12., 18., 1., -3., -7., -11., -0., 3., 7., 10.]),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc')),
     [1., 6., 11., 15., 1., -2., -6., -8., -0., 3., 6., 8.]),
])
def test_deflection_model(deflectionModel, dy):
    # center line deflection 10d downstream
    x, y = [0, 400, 800], [0, 0, 0]
    wfm = simple_wfm(deflectionModel)
    D = wfm.windTurbines.diameter()
    yaw = [-30, 30, -30]

    sim_res = wfm(x, y, yaw=yaw, tilt=0, wd=270, ws=10)
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
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00.nc')),
     [1.0, 6.0, 12.0, 18.0, 2.0, -0.0, -4.0, -7.0, -1.0, 2.0, 4.0, 7.0]),
    ((lambda: FugaDeflection(tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc')),
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

    sim_res = wfm(x, y, yaw=yaw, tilt=0, wd=270, ws=10)
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
    x, y = [0], [0]
    yaw_ilk = np.reshape([-30], (1, 1, 1))
    wfm = simple_wfm(deflectionModel)
    D = wfm.windTurbines.diameter()
    sim_res = wfm(x, y, yaw=yaw_ilk, tilt=0, wd=270, ws=10)
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


def test_combined_tilt_and_yaw_deflection():
    wfm = simple_wfm(JimenezWakeDeflection)
    deflectionModel = wfm.deflectionModel

    wfm([0, 500], [0, 0], yaw=30, tilt=0, wd=270, ws=10)
    hcw = deflectionModel.hcw_ijlk
    wfm([0, 500], [0, 0], yaw=0, tilt=30, wd=270, ws=10)
    dh = deflectionModel.dh_ijlk
    npt.assert_array_almost_equal(hcw, -dh)

    # rotate 30deg yaw, 20 deg around u direction
    theta, gamma = np.deg2rad(30), np.deg2rad(20)
    tilt = np.rad2deg(np.arcsin(np.sin(theta) * np.sin(gamma)))

    yaw = np.rad2deg(np.arcsin(np.cos(gamma) * np.sin(theta) / np.sqrt(1 - (np.sin(gamma) * np.sin(theta))**2)))
    # same as
    yaw = np.rad2deg(np.arcsin(np.cos(gamma) * np.sin(theta) / np.cos(np.arcsin(np.sin(theta) * np.sin(gamma)))))
    wfm([0, 500], [0, 0], yaw=yaw, tilt=tilt, wd=270, ws=10)
    npt.assert_array_almost_equal(hcw, np.hypot(deflectionModel.hcw_ijlk, deflectionModel.dh_ijlk))


def test_upstream_deflection():
    plot = 0
    if plot:
        ax = plt.gca()
    for m in get_models(DeflectionModel):
        wfm = FugaBlockage(tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc',
                           UniformSite(), V80(), deflectionModel=m if m is None else m())

        fm = wfm([0], [0], wd=270, ws=10, yaw=30, tilt=0).flow_map(XYGrid(x=-160, y=np.linspace(-100, 100, 501)))
        l = "None" if m is None else m.__name__
        blockage_center = fm.y[np.argmin(fm.WS_eff.squeeze().values)].item()
        ref = {'None': 0.0,
               'FugaDeflection': 1.6,
               'GCLHillDeflection': -1.6,
               'JimenezWakeDeflection': -18}[l]

        # npt.assert_almost_equal(ref, blockage_center, err_msg=l)
        if plot:
            fm.WS_eff.squeeze().plot(ax=ax, label=l)
            plt.figure()
            yaw = 30
            fm = wfm([0], [0], ws=10, wd=270, yaw=[[[yaw]]], tilt=0).flow_map(
                XYGrid(x=np.arange(-100, 800, 10), y=np.arange(-250, 250, 10)))
            X, Y = np.meshgrid(fm.x, fm.y)
            c1 = plt.contourf(X, Y, fm.WS_eff.squeeze(), np.arange(6.5, 10.1, 0.1), cmap='Blues_r')
            c2 = plt.contourf(X, Y, fm.WS_eff.squeeze(), np.arange(10, 10.2, .005), cmap='Reds')
            plt.colorbar(c2, label='Wind speed (deficit regions) [m/s]')
            plt.colorbar(c1, label='Wind speed (speed-up regions) [m/s]')
            V80().plot([0], [0], wd=270, yaw=yaw)
            max_deficit_line = fm.min_WS_eff(x=np.arange(-100, 800, 10))
            max_deficit_line.plot(color='k', label='Max deficit line')
            plt.axhline(0, label='Center line')
            plt.title(l)
            plt.xlim([-100, 300])
            plt.legend()

    if plot:
        ax.legend()
        plt.show()
