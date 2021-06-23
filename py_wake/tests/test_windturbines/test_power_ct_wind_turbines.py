from numpy import newaxis as na
import pytest
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from py_wake.examples.data import hornsrev1
from py_wake.tests import npt
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt, DensityScale, \
    PowerCtTabular, PowerCtFunction, PowerCtFunctionList, PowerCtNDTabular, PowerCtXr
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.wind_turbines import WindTurbine, WindTurbines, power_ct_functions, wind_turbines_deprecated
from py_wake.deficit_models.noj import NOJ
from py_wake.site.xrsite import XRSite
from py_wake.tests.test_windturbines.test_power_ct_curves import get_continuous_curve
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.utils.gradients import use_autograd_in, autograd, plot_gradients, cs, fd
from py_wake.examples.data.iea37 import iea37_reader
from types import FunctionType
from py_wake.utils import gradients

v80_upct = np.array([hornsrev1.power_curve[:, 0], hornsrev1.power_curve[:, 1], hornsrev1.ct_curve[:, 1]])


def get_wt(power_ct_func):
    if isinstance(power_ct_func, list):
        N = len(power_ct_func)
        return WindTurbines(names=['Test'] * N, diameters=[80] * N, hub_heights=[70] * N, power_ct_funcs=power_ct_func)
    else:
        return WindTurbine(name='Test', diameter=80, hub_height=70, powerCtFunction=power_ct_func)


def get_wfm(power_ct_func, **kwargs):
    if 'site' not in kwargs:
        kwargs['site'] = Hornsrev1Site()
    wt = get_wt(power_ct_func)
    return NOJ(windTurbines=wt, **kwargs)


@pytest.mark.parametrize('method,unit,p_scale,p_ref,ct_ref', [
    ('linear', 'w', 1,
     [0.0, 0.0, 371000.0, 1168500.0, 1912000.0, 1998000.0, 2000000.0, 2000000.0, 2000000.0, 0.0],
     [0.0, 0.0, 0.804, 0.8, 0.559, 0.226, 0.13, 0.082, 0.056, 0.0]),
    ('linear', 'kw', 1e-3,
     [0.0, 0.0, 371000.0, 1168500.0, 1912000.0, 1998000.0, 2000000.0, 2000000.0, 2000000.0, 0.0],
     [0.0, 0.0, 0.804, 0.8, 0.559, 0.226, 0.13, 0.082, 0.056, 0.0]),
    ('linear', 'mw', 1e-6,
     [0.0, 0.0, 371000.0, 1168500.0, 1912000.0, 1998000.0, 2000000.0, 2000000.0, 2000000.0, 0.0],
     [0.0, 0.0, 0.804, 0.8, 0.559, 0.226, 0.13, 0.082, 0.056, 0.0]),
    ('pchip', 'w', 1,
     [0.0, 0.0, 364247.229, 1167112.52, 1922219.683, 1998242.424, 2000000.0, 2000000.0, 2000000.0, 0.0],
     [0.0, 0.0, 0.804, 0.803, 0.57, 0.224, 0.129, 0.082, 0.058, 0.0])][:])
def test_PowerCtTabular(method, unit, p_scale, p_ref, ct_ref):
    u_p, p, ct = v80_upct.copy()
    p *= p_scale
    curve = PowerCtTabular(ws=u_p, power=p, power_unit=unit, ct=ct, ws_cutin=4, ws_cutout=25, method=method)
    u = np.arange(0, 30, .1)

    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, [])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])
    sim_res = wfm([0], [0], ws=u, wd=[0]).squeeze()
    p = sim_res.Power.values
    ct = sim_res.CT.values

    s = slice(5, None, 30)
    if 0:
        plt.plot(u, p)
        plt.plot(u[s], p[s], '.')
        ax2 = plt.gca().twinx()
        ax2.plot(u, ct)
        ax2.plot(u[s], ct[s], '.')
        print(np.round(p[s], 3).tolist())
        print(np.round(ct[s], 3).tolist())
        plt.show()

    npt.assert_array_almost_equal(p[s], p_ref, 3)
    npt.assert_array_almost_equal(ct[s], ct_ref, 3)


def test_MultiPowerCtCurve():
    u_p, p, ct = v80_upct.copy()

    curve = PowerCtFunctionList('mode', [PowerCtTabular(ws=u_p, power=p, power_unit='w', ct=ct),
                                         PowerCtTabular(ws=u_p, power=p * 1.1, power_unit='w', ct=ct + .1)])

    u = np.arange(0, 30, .1)

    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, ['mode'])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])

    sim_res = wfm([0], [0], ws=u, wd=0, mode=0)
    p0, ct0 = sim_res.Power.squeeze().values, sim_res.CT.squeeze().values,
    sim_res = wfm([0], [0], ws=u, wd=0, mode=1)
    p1, ct1 = sim_res.Power.squeeze().values, sim_res.CT.squeeze().values,

    npt.assert_array_almost_equal(p0, p1 / 1.1)
    npt.assert_array_almost_equal(ct0, ct1 - .1)

    mode_16 = (np.arange(16) > 7)
    mode_16_360 = np.broadcast_to(mode_16[:, na], (16, 360))
    mode_16_360_23 = np.broadcast_to(mode_16[:, na, na], (16, 360, 23))

    ref_p = np.array(np.broadcast_to(hornsrev1.power_curve[:, 1][na, na], (16, 360, 23)))
    ref_p[8:] *= 1.1

    for m in [mode_16, mode_16_360, mode_16_360_23]:
        sim_res = wfm(np.arange(16) * 1e3, [0] * 16, wd=np.arange(360) % 5, mode=m)  # no wake effects
        p = sim_res.Power.values
        npt.assert_array_almost_equal(p, ref_p)


def test_MultiMultiPowerCtCurve_subset():
    u_p, p, ct = v80_upct.copy()

    curves = PowerCtFunctionList('mytype', [
        PowerCtFunctionList('mode', [PowerCtTabular(ws=u_p, power=p + 1, power_unit='w', ct=ct),
                                     PowerCtTabular(ws=u_p, power=p + 2, power_unit='w', ct=ct),
                                     PowerCtTabular(ws=u_p, power=p + 3, power_unit='w', ct=ct)]),
        PowerCtFunctionList('mode', [PowerCtTabular(ws=u_p, power=p + 4, power_unit='w', ct=ct),
                                     PowerCtTabular(ws=u_p, power=p + 5, power_unit='w', ct=ct),
                                     PowerCtTabular(ws=u_p, power=p + 6, power_unit='w', ct=ct)]),
    ])
    wfm = get_wfm(curves)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, ['mode', 'mytype'])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])

    u = np.zeros((2, 3, 4)) + np.arange(3, 7)[na, na, :]
    type_2 = np.array([0, 1])
    type_2_3 = np.broadcast_to(type_2[:, na], (2, 3))
    type_2_3_4 = np.broadcast_to(type_2[:, na, na], (2, 3, 4))

    mode_2_3 = np.broadcast_to(np.array([0, 1, 2])[na, :], (2, 3))
    mode_2_3_4 = np.broadcast_to(mode_2_3[:, :, na], (2, 3, 4))

    ref_p = np.array(np.broadcast_to(hornsrev1.power_curve[:4, 1][na, na], (2, 3, 4)))
    ref_p[0, :] += np.array([1, 2, 3])[:, na]
    ref_p[1, :] += np.array([4, 5, 6])[:, na]

    for t in [type_2, type_2_3, type_2_3_4]:
        for m in [mode_2_3, mode_2_3_4]:
            sim_res = wfm([0, 1000], [0, 0], wd=np.arange(3), ws=np.arange(3, 7), mode=m, mytype=t)  # no wake effects
            p = sim_res.Power.values
            npt.assert_array_almost_equal(p, ref_p)


def test_2d_tabular():
    u_p, p_c, ct_c = v80_upct.copy()

    curve = PowerCtNDTabular(['ws', 'boost'], [u_p, [0, 1]],
                             np.array([p_c, 2 * p_c]).T, 'w',
                             np.array([ct_c, ct_c]).T)
    u = np.linspace(3, 25, 10)
    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, ['boost'])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])

    sim_res = wfm([0], [0], wd=0, ws=u, boost=0)
    p = sim_res.Power.values
    ct = sim_res.CT.values
    npt.assert_array_almost_equal_nulp(p, np.interp(u, u_p, p_c))
    npt.assert_array_almost_equal_nulp(ct, np.interp(u, u_p, ct_c))

    sim_res = wfm([0], [0], wd=0, ws=u, boost=.5)
    p = sim_res.Power.values
    ct = sim_res.CT.values

    npt.assert_array_almost_equal_nulp(p, np.interp(u, u_p, p_c * 1.5))
    npt.assert_array_almost_equal_nulp(ct, np.interp(u, u_p, ct_c))

    boost_16 = (np.arange(16) > 7) * .1
    boost_16_360 = np.broadcast_to(boost_16[:, na], (16, 360))
    boost_16_360_23 = np.broadcast_to(boost_16[:, na, na], (16, 360, 23))

    ref_p = np.array(np.broadcast_to(hornsrev1.power_curve[:, 1][na, na], (16, 360, 23)))
    ref_p[8:] *= 1.1

    wfm = get_wfm(curve)
    for b in [boost_16, boost_16_360, boost_16_360_23]:
        sim_res = wfm(np.arange(16) * 1e3, [0] * 16, wd=np.arange(360) % 5, boost=b)  # no wake effects
        p = sim_res.Power.values
        npt.assert_array_almost_equal(p, ref_p)


def test_PowerCtXr():
    u_p, p_c, ct_c = v80_upct.copy()

    ds = xr.Dataset(
        data_vars={'ct': (['ws', 'boost'], np.array([ct_c, ct_c]).T),
                   'power': (['ws', 'boost'], np.array([p_c, p_c * 2]).T)},
        coords={'boost': [0, 1], 'ws': u_p, }).transpose('boost', 'ws')
    curve = PowerCtXr(ds, 'w')
    u = np.linspace(3, 25, 10)
    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, ['boost'])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])

    sim_res = wfm([0], [0], wd=0, ws=u, boost=0)
    p = sim_res.Power.values
    ct = sim_res.CT.values
    npt.assert_array_almost_equal_nulp(p, np.interp(u, u_p, p_c))
    npt.assert_array_almost_equal_nulp(ct, np.interp(u, u_p, ct_c))

    sim_res = wfm([0], [0], wd=0, ws=u, boost=.5)
    p = sim_res.Power.values
    ct = sim_res.CT.values

    npt.assert_array_almost_equal_nulp(p[0, 0], np.interp(u, u_p, p_c * 1.5))
    npt.assert_array_almost_equal_nulp(ct, np.interp(u, u_p, ct_c))

    boost_16 = (np.arange(16) > 7) * .1
    boost_16_360 = np.broadcast_to(boost_16[:, na], (16, 360))
    boost_16_360_23 = np.broadcast_to(boost_16[:, na, na], (16, 360, 23))

    ref_p = np.array(np.broadcast_to(hornsrev1.power_curve[:, 1][na, na], (16, 360, 23)))
    ref_p[8:] *= 1.1

    wfm = get_wfm(curve)
    for b in [boost_16, boost_16_360, boost_16_360_23]:
        sim_res = wfm(np.arange(16) * 1e3, [0] * 16, wd=np.arange(360) % 5, boost=b)  # no wake effects
        p = sim_res.Power.values
        npt.assert_array_almost_equal(p, ref_p)


def test_FunctionalPowerCtCurve():
    curve = CubePowerSimpleCt()
    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, [])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])

    u = np.arange(0, 30, .1)
    sim_res = wfm([0], [0], wd=0, ws=u)
    p = sim_res.Power.values.squeeze()
    ct = sim_res.CT.values.squeeze()

    s = slice(5, None, 30)
    if 0:
        c = plt.plot(u, p)[0].get_color()
        plt.plot(u[s], p[s], '.', color=c)
        c = plt.plot([])[0].get_color()
        ax2 = plt.gca().twinx()
        ax2.plot(u, ct, color=c)
        ax2.plot(u[s], ct[s], '.', color=c)
        print(np.round(p[s], 3).tolist())
        print(np.round(ct[s], 3).tolist())
        plt.show()
    npt.assert_array_almost_equal(p[s], [0.0, 857.339, 294067.215, 1883573.388, 5000000.0, 5000000.0,
                                         5000000.0, 5000000.0, 5000000.0, 0.0], 3)
    npt.assert_array_almost_equal(ct[s], [0.03, 0.889, 0.889, 0.889, 0.824, 0.489, 0.245, 0.092, 0.031, 0.03], 3)


def test_continuous():
    curve = get_continuous_curve('boost', True)

    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, [])
    npt.assert_array_equal(oi, ['Air_density', 'boost', 'tilt', 'yaw'])

    u = np.arange(0, 30, .1)
    sim_res = wfm([0], [0], wd=0, ws=u)
    p0 = sim_res.Power.values
    ct0 = sim_res.CT.values

    sim_res = wfm([0], [0], wd=0, ws=u, boost=1.1)
    p1 = sim_res.Power.values
    ct1 = sim_res.CT.values

    npt.assert_array_almost_equal(p0, p1 / 1.1)
    npt.assert_array_almost_equal(ct0, ct1)

    # subset
    boost_16 = (np.arange(16) > 7) * .1 + 1
    boost_16_360 = np.broadcast_to(boost_16[:, na], (16, 360))
    boost_16_360_23 = np.broadcast_to(boost_16[:, na, na], (16, 360, 23))

    ref_p = np.array(np.broadcast_to(hornsrev1.power_curve[:, 1][na, na], (16, 360, 23)))
    ref_p[8:] *= 1.1

    for b in [boost_16, boost_16_360, boost_16_360_23]:
        sim_res = wfm(np.arange(16) * 1e3, [0] * 16, wd=np.arange(360) % 5, boost=b)  # no wake effects
        p = sim_res.Power.values
        npt.assert_array_almost_equal(p, ref_p)


def test_unused_input():
    curve = get_continuous_curve('boost', True)
    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, [])
    npt.assert_array_equal(oi, ['Air_density', 'boost', 'tilt', 'yaw'])
    with pytest.raises(TypeError, match=r"got unexpected keyword argument\(s\): 'mode'"):
        wfm([0], [0], boost=1, mode=1)


def test_missing_input():
    curve = get_continuous_curve('boost', False)
    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, ['boost'])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])

    with pytest.raises(KeyError, match="Argument, boost, required to calculate power and ct not found"):
        wfm([0], [0])


def test_missing_input_PowerCtFunctionList():
    u_p, p, ct = v80_upct.copy()
    curve = PowerCtFunctionList('mode', [PowerCtTabular(ws=u_p, power=p + 1, power_unit='w', ct=ct),
                                         PowerCtTabular(ws=u_p, power=p + 2, power_unit='w', ct=ct),
                                         PowerCtTabular(ws=u_p, power=p + 3, power_unit='w', ct=ct)])

    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, ['mode'])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])

    with pytest.raises(KeyError, match="Argument, mode, required to calculate power and ct not found"):
        wfm([0], [0])


def test_DensityScaleAndSimpleYawModel():
    u_p, p_c, ct_c = v80_upct.copy()

    curve = PowerCtTabular(ws=u_p, power=p_c, power_unit='w', ct=ct_c, method='linear')
    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, [])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])

    u = np.arange(4, 25, 1.1)
    yaw = 30
    theta = np.deg2rad(yaw)
    yaw_ilk = (u * 0 + yaw).reshape(1, 1, len(u))

    co = np.cos(theta)

    sim_res = wfm([0], [0], wd=0, ws=u, yaw=yaw_ilk, Air_density=1.3)
    p = sim_res.Power.values.squeeze()
    ct = sim_res.CT.values.squeeze()

    npt.assert_array_almost_equal(p, np.interp(u * co, u_p, p_c) * 1.3 / 1.225)
    npt.assert_array_almost_equal(ct, np.interp(u * co, u_p, ct_c) * co**2 * 1.3 / 1.225)


def test_DensityScaleFromSite():
    ds = Hornsrev1Site().ds
    ds['Air_density'] = 1.3

    u_p, p_c, ct_c = v80_upct.copy()

    for rho_ref in [1.225, 1.2]:
        curve = PowerCtTabular(ws=u_p, power=p_c, power_unit='w', ct=ct_c, ws_cutin=4, ws_cutout=25,
                               method='linear', additional_models=[DensityScale(rho_ref)])
        wfm = get_wfm(curve, site=XRSite(ds))
        ri, oi = wfm.windTurbines.function_inputs
        npt.assert_array_equal(ri, [])
        npt.assert_array_equal(oi, ['Air_density'])
        u = np.arange(4, 25, .1)
        sim_res = wfm([0], [0], wd=0, ws=u,)
        p = sim_res.Power.values.squeeze()
        ct = sim_res.CT.values.squeeze()
        npt.assert_array_almost_equal(p, np.interp(u, u_p, p_c) * 1.3 / rho_ref)
        npt.assert_array_almost_equal(ct, np.interp(u, u_p, ct_c) * 1.3 / rho_ref)


def test_WSFromLocalWind():

    u_p, p_c, _ = v80_upct.copy()

    curve = get_continuous_curve('WD', False)
    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, ['WD'])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])
    u = np.arange(4, 25, .1)
    sim_res = wfm([0], [0], wd=[2], ws=u)
    p = sim_res.Power.values.squeeze()
    npt.assert_array_almost_equal(p, np.interp(u, u_p, p_c) * 2)


def test_TIFromWFM():
    u_p, p_c, ct_c = v80_upct.copy()

    tab_powerct_curve = PowerCtTabular(ws=u_p, power=p_c, power_unit='w', ct=ct_c)

    def _power_ct(ws, run_only, TI_eff):
        return tab_powerct_curve(ws, run_only) * [TI_eff, np.ones_like(TI_eff)][run_only]

    curve = PowerCtFunction(['ws', 'TI_eff'], _power_ct, 'w')

    wfm = get_wfm(curve)
    ri, oi = wfm.windTurbines.function_inputs
    npt.assert_array_equal(ri, ['TI_eff'])
    npt.assert_array_equal(oi, ['Air_density', 'tilt', 'yaw'])
    u = np.arange(4, 25, .1)
    with pytest.raises(KeyError, match='Argument, TI_eff, needed to calculate power and ct requires a TurbulenceModel'):
        wfm([0], [0], wd=[2], ws=u)

    wfm = get_wfm(curve, turbulenceModel=STF2017TurbulenceModel())
    u = np.arange(4, 25, .1)
    sim_res = wfm([0], [0], wd=[2], ws=u)

    p = sim_res.Power.values.squeeze()
    npt.assert_array_almost_equal(p, np.interp(u, u_p, p_c) * sim_res.TI_eff.squeeze())


@pytest.mark.parametrize('case,wt,dpdu_ref,dctdu_ref', [
    ('CubePowerSimpleCt', WindTurbine('Test', 130, 110, CubePowerSimpleCt(4, 25, 9.8, 3.35e6, 'W', 8 / 9, 0, [])),
     lambda ws_pts: np.where((ws_pts > 4) & (ws_pts <= 9.8), 3 * 3350000 * (ws_pts - 4)**2 / (9.8 - 4)**3, 0),
     lambda ws_pts: [0, 0, 0, 2 * 12 * 0.0038473376423514938 + -0.1923668821175747]),
    ('CubePowerSimpleCt_MW', WindTurbine('Test', 130, 110, CubePowerSimpleCt(4, 25, 9.8, 3.35, 'MW', 8 / 9, 0, [])),
     lambda ws_pts: np.where((ws_pts > 4) & (ws_pts <= 9.8), 3 * 3350000 * (ws_pts - 4)**2 / (9.8 - 4)**3, 0),
     lambda ws_pts: [0, 0, 0, 2 * 12 * 0.0038473376423514938 + -0.1923668821175747]),
    ('PowerCtTabular_linear', V80(), [66600.00000931, 178000.00001444, 344999.99973923, 91999.99994598],
     [0.818, 0.001, -0.014, -0.3]),
    ('PowerCtTabular_pchip', V80(method='pchip'), [58518.7948, 148915.03267974, 320930.23255814, 127003.36700337],
     [1.21851, 0., 0., -0.05454545]),
    ('PowerCtTabular_spline', V80(method='spline'), [69723.7929, 158087.18190692, 324012.9604669, 156598.55856862],
     [8.176490e-01, -3.31076624e-05, -7.19353708e-03, -1.66006862e-01]),
    ('PowerCtTabular_kW', get_wt(PowerCtTabular(v80_upct[0], v80_upct[1] / 1000, 'kW', v80_upct[2])),
     [66600.00000931, 178000.00001444, 344999.99973923, 91999.99994598], [0.818, 0.001, -0.014, -0.3]),
    ('PowerCtFunctionList',
     get_wt(PowerCtFunctionList('mode', [
         PowerCtTabular(ws=v80_upct[0], power=v80_upct[1], power_unit='w', ct=v80_upct[2]),
         PowerCtTabular(ws=v80_upct[0], power=v80_upct[1] * 1.1, power_unit='w', ct=v80_upct[2] + .1)])),
     [73260., 195800., 379499.9998, 101199.9999], [0.818, 0.001, -0.014, -0.3])
])
@pytest.mark.parametrize('grad_method', [autograd, cs, fd])
def test_gradients(case, wt, dpdu_ref, dctdu_ref, grad_method):

    ws_pts = np.array([3.1, 6., 9., 12.])
    if isinstance(dpdu_ref, FunctionType):
        dpdu_ref, dctdu_ref = dpdu_ref(ws_pts), dctdu_ref(ws_pts)

    with use_autograd_in([WindTurbines, iea37_reader, power_ct_functions, wind_turbines_deprecated]):
        if grad_method == autograd:
            wt.enable_autograd()
        ws_lst = np.arange(2, 25, .1)
        kwargs = {k: 1 for k in wt.function_inputs[0]}
        dpdu_lst = grad_method(wt.power)(ws_pts, **kwargs)
        dctdu_lst = grad_method(wt.ct)(ws_pts, **kwargs)

    if 0:
        gradients.color_dict = {'power': 'b', 'ct': 'r'}
        plt.plot(ws_lst, wt.power(ws_lst, **kwargs), label='power')
        c = plt.plot([], label='ct')[0].get_color()
        for dpdu, ws in zip(dpdu_lst, ws_pts):
            plot_gradients(wt.power(ws, **kwargs), dpdu, ws, "power", 1)
        ax = plt.twinx()
        ax.plot(ws_lst, wt.ct(ws_lst, **kwargs), color=c)
        for dctdu, ws in zip(dctdu_lst, ws_pts):
            plot_gradients(wt.ct(ws, **kwargs), dctdu, ws, "ct", 1, ax=ax)
        plt.title(case)
        plt.show()

    npt.assert_array_almost_equal(dpdu_lst, dpdu_ref, (0, 4)[grad_method == autograd])
    npt.assert_array_almost_equal(dctdu_lst, dctdu_ref, (4, 6)[grad_method == autograd])


if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 100)
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(x, np.sin(x) / x)
    plt.plot(x, np.sin(x))
    plt.show()
