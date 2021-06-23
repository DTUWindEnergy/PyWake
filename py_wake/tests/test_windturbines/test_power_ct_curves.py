from numpy import newaxis as na
import pytest
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from py_wake.examples.data import hornsrev1
from py_wake.tests import npt
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt, PowerCtNDTabular, DensityScale, \
    PowerCtTabular, PowerCtFunction, PowerCtFunctionList, PowerCtXr
from py_wake.wind_turbines.wind_turbine_functions import WindTurbineFunction
from py_wake.utils.model_utils import fix_shape


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
     [0.0, 0.0, 0.804, 0.803, 0.57, 0.224, 0.129, 0.082, 0.058, 0.0])])
def test_TabularPowerCtCurve(method, unit, p_scale, p_ref, ct_ref):
    u_p, p = np.asarray(hornsrev1.power_curve).T.copy()
    p *= p_scale
    u_ct, ct = hornsrev1.ct_curve.T
    npt.assert_array_equal(u_p, u_ct)
    curve = PowerCtTabular(ws=u_p, power=p, power_unit=unit, ct=ct, ws_cutin=4, ws_cutout=25, method=method)
    npt.assert_array_equal(curve.optional_inputs, ['Air_density', 'tilt', 'yaw'])
    npt.assert_array_equal(curve.required_inputs, [])

    u = np.arange(0, 30, .1)
    p, ct = curve(u)
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

    p, ct = curve(u, run_only=0), curve(u, run_only=1)
    npt.assert_array_almost_equal(p[s], p_ref, 3)
    npt.assert_array_almost_equal(ct[s], ct_ref, 3)


def test_MultiPowerCtCurve():
    u_p, p = np.asarray(hornsrev1.power_curve).T.copy()
    ct = hornsrev1.ct_curve[:, 1]

    curve = PowerCtFunctionList('mode', [PowerCtTabular(ws=u_p, power=p, power_unit='w', ct=ct),
                                         PowerCtTabular(ws=u_p, power=p * 1.1, power_unit='w', ct=ct + .1)])

    npt.assert_array_equal(sorted(curve.optional_inputs), ['Air_density', 'tilt', 'yaw'])
    npt.assert_array_equal(list(curve.required_inputs), ['mode'])

    u = np.arange(0, 30, .1)
    p0, ct0 = curve(u, mode=0)
    p1, ct1 = curve(u, mode=1)

    npt.assert_array_almost_equal(p0, p1 / 1.1)
    npt.assert_array_almost_equal(ct0, ct1 - .1)

    # subset
    u = np.zeros((16, 360, 23)) + np.arange(3, 26)[na, na, :]
    mode_16 = (np.arange(16) > 7)
    mode_16_360 = np.broadcast_to(mode_16[:, na], (16, 360))
    mode_16_360_23 = np.broadcast_to(mode_16[:, na, na], (16, 360, 23))

    ref_p = np.array(np.broadcast_to(hornsrev1.power_curve[:, 1][na, na], (16, 360, 23)))
    ref_p[8:] *= 1.1

    for m in [mode_16, mode_16_360, mode_16_360_23]:
        p, ct = curve(u, mode=m)
        npt.assert_array_almost_equal(p, ref_p)


def test_MultiMultiPowerCtCurve_subset():
    u_p, p = np.asarray(hornsrev1.power_curve).T.copy()
    ct = hornsrev1.ct_curve[:, 1]

    curve = PowerCtFunctionList('type', [
        PowerCtFunctionList('mode', [PowerCtTabular(ws=u_p, power=p + 1, power_unit='w', ct=ct),
                                     PowerCtTabular(ws=u_p, power=p + 2, power_unit='w', ct=ct),
                                     PowerCtTabular(ws=u_p, power=p + 3, power_unit='w', ct=ct)]),
        PowerCtFunctionList('mode', [PowerCtTabular(ws=u_p, power=p + 4, power_unit='w', ct=ct),
                                     PowerCtTabular(ws=u_p, power=p + 5, power_unit='w', ct=ct),
                                     PowerCtTabular(ws=u_p, power=p + 6, power_unit='w', ct=ct)]),
    ])

    npt.assert_array_equal(sorted(curve.optional_inputs)[::-1], ['yaw', 'tilt', 'Air_density'])
    npt.assert_array_equal(sorted(curve.required_inputs)[::-1], ['type', 'mode'])

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
            p, ct = curve(u, mode=m, type=t)
            npt.assert_array_almost_equal(p, ref_p)


def test_2d_tabular():
    u_p, p_c = np.asarray(hornsrev1.power_curve).T.copy()
    ct_c = hornsrev1.ct_curve[:, 1]
    curve = PowerCtNDTabular(['ws', 'boost'], [u_p, [0, 1]],
                             np.array([p_c, 2 * p_c]).T, 'w',
                             np.array([ct_c, ct_c]).T)
    npt.assert_array_equal(sorted(curve.optional_inputs)[::-1], ['yaw', 'tilt', 'Air_density'])
    npt.assert_array_equal(sorted(curve.required_inputs)[::-1], ['boost'])

    u = np.linspace(3, 25, 10)
    p, ct = curve(ws=u, boost=0)
    npt.assert_array_almost_equal_nulp(p, np.interp(u, u_p, p_c))
    npt.assert_array_almost_equal_nulp(ct, np.interp(u, u_p, ct_c))

    # check out of bounds
    with pytest.raises(ValueError, match='Input, boost, with value, 2.0 outside range 0-1'):
        curve(ws=u, boost=2)

    # no default value > KeyError
    with pytest.raises(KeyError, match="boost"):
        curve(ws=u)

    p, ct = curve(ws=u, boost=.5)
    npt.assert_array_almost_equal_nulp(p, np.interp(u, u_p, p_c * 1.5))
    npt.assert_array_almost_equal_nulp(ct, np.interp(u, u_p, ct_c))

    # subset
    u = np.zeros((16, 360, 23)) + np.arange(3, 26)[na, na, :]
    boost_16 = (np.arange(16) > 7) * .1
    boost_16_360 = np.broadcast_to(boost_16[:, na], (16, 360))
    boost_16_360_23 = np.broadcast_to(boost_16[:, na, na], (16, 360, 23))

    ref_p = np.array(np.broadcast_to(hornsrev1.power_curve[:, 1][na, na], (16, 360, 23)))
    ref_p[8:] *= 1.1

    for b in [boost_16, boost_16_360, boost_16_360_23]:
        p, ct = curve(u, boost=b)
        npt.assert_array_almost_equal(p, ref_p)


def test_2d_tabular_default_value():
    u_p, p_c = np.asarray(hornsrev1.power_curve).T.copy()
    ct_c = hornsrev1.ct_curve[:, 1]
    curve = PowerCtNDTabular(['ws', 'boost'], [u_p, [0, 1]],
                             np.array([p_c, 2 * p_c]).T, 'w',
                             np.array([ct_c, ct_c]).T,
                             {'boost': .1})

    u = np.linspace(3, 25, 10)

    # check default value
    npt.assert_array_equal(curve(ws=u, boost=.1), curve(ws=u))


def test_FunctionalPowerCtCurve():
    curve = CubePowerSimpleCt()
    npt.assert_array_equal(sorted(curve.optional_inputs)[::-1], ['yaw', 'tilt', 'Air_density'])
    npt.assert_array_equal(curve.required_inputs, [])
    u = np.arange(0, 30, .1)
    p, ct = curve(u, Air_density=1.3)
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
    npt.assert_array_almost_equal(p[s], np.array([0.0, 857.339, 294067.215, 1883573.388, 5000000.0, 5000000.0,
                                                  5000000.0, 5000000.0, 5000000.0, 0.0]) * 1.3 / 1.225, 3)
    npt.assert_array_almost_equal(ct[s], np.array([0.03, 0.889, 0.889, 0.889, 0.824, 0.489,
                                                   0.245, 0.092, 0.031, 0.03]) * 1.3 / 1.225, 3)


def get_continuous_curve(key, optional):
    u_p, p = np.asarray(hornsrev1.power_curve).T.copy()
    ct = hornsrev1.ct_curve[:, 1]
    tab_powerct_curve = PowerCtTabular(ws=u_p, power=p, power_unit='w', ct=ct)

    def _power_ct(ws, run_only, **kwargs):
        try:
            v = fix_shape(kwargs.pop(key), ws, True)
        except KeyError:
            if optional:
                v = 1
            else:
                raise
        return tab_powerct_curve(ws, run_only) * (v, 1)[run_only]

    oi = [key] if optional else []
    return PowerCtFunction(['ws', key], _power_ct, 'w', optional_inputs=oi)


def test_continuous():
    curve = get_continuous_curve('boost', True)

    u = np.arange(0, 30, .1)
    p0, ct0 = curve(u)
    p1, ct1 = curve(u, boost=1.1)
    p2, ct2 = curve(u, boost=1.1, Air_density=1.3)

    npt.assert_array_almost_equal(p0, p1 / 1.1)
    npt.assert_array_almost_equal(ct0, ct1)

    npt.assert_array_almost_equal(p0 * 1.1 * 1.3 / 1.225, p2)
    npt.assert_array_almost_equal(ct0 * 1.3 / 1.225, ct2)

    # subset
    u = np.zeros((16, 360, 23)) + np.arange(3, 26)[na, na, :]
    boost_16 = (np.arange(16) > 7) * .1 + 1
    boost_16_360 = np.broadcast_to(boost_16[:, na], (16, 360))
    boost_16_360_23 = np.broadcast_to(boost_16[:, na, na], (16, 360, 23))

    ref_p = np.array(np.broadcast_to(hornsrev1.power_curve[:, 1][na, na], (16, 360, 23)))
    ref_p[8:] *= 1.1

    for b in [boost_16, boost_16_360, boost_16_360_23]:
        p, ct = curve(u, boost=b)
        npt.assert_array_almost_equal(p, ref_p)


def test_PowerCtXr():
    u_p, p_c = np.asarray(hornsrev1.power_curve).T.copy()
    ct_c = hornsrev1.ct_curve[:, 1]
    ds = xr.Dataset(
        data_vars={'power': (['ws', 'boost'], np.array([p_c, p_c * 1.1]).T),
                   'ct': (['ws', 'boost'], np.array([ct_c, ct_c * 1.1]).T)},
        coords={'boost': [0, 10], 'ws': u_p})
    curve = PowerCtXr(ds, 'w')
    npt.assert_array_almost_equal(curve(u_p, boost=0), curve(u_p, boost=10) / 1.1)


def test_missing_input_PowerCtFunctionList():
    u_p, p = np.asarray(hornsrev1.power_curve).T.copy()
    ct = hornsrev1.ct_curve[:, 1]

    curve = PowerCtFunctionList('mode', [PowerCtTabular(ws=u_p, power=p + 1, power_unit='w', ct=ct),
                                         PowerCtTabular(ws=u_p, power=p + 2, power_unit='w', ct=ct),
                                         PowerCtTabular(ws=u_p, power=p + 3, power_unit='w', ct=ct)])
    u = np.zeros((16, 360, 23)) + np.arange(3, 26)[na, na, :]
    with pytest.raises(KeyError, match="Argument, mode, required to calculate power and ct not found"):
        curve(u)


def test_missing_input():
    curve = get_continuous_curve('boost', False)

    u = np.zeros((16, 360, 23)) + np.arange(3, 26)[na, na, :]
    with pytest.raises(KeyError, match="'boost'"):
        curve(u)


def test_density_scale():
    u_p, p_c = np.asarray(hornsrev1.power_curve).T.copy()
    _, ct_c = hornsrev1.ct_curve.T
    for rho_ref in [1.225, 1.2]:
        curve = PowerCtTabular(ws=u_p, power=p_c, power_unit='w', ct=ct_c, ws_cutin=4, ws_cutout=25,
                               method='linear', additional_models=[DensityScale(rho_ref)])
        u = np.arange(4, 25, .1)
        p, ct = curve(u, Air_density=1.3)
        npt.assert_array_almost_equal(p, np.interp(u, u_p, p_c) * 1.3 / rho_ref)
        npt.assert_array_almost_equal(ct, np.interp(u, u_p, ct_c) * 1.3 / rho_ref)


def test_SimpleYawModel():
    u_p, p_c = np.asarray(hornsrev1.power_curve).T.copy()
    _, ct_c = hornsrev1.ct_curve.T
    curve = PowerCtTabular(ws=u_p, power=p_c, power_unit='w', ct=ct_c, method='linear')
    u = np.arange(4, 25, 1.1)
    yaw = 30
    theta = np.deg2rad(yaw)
    co = np.cos(theta)
    p, ct = curve(u, yaw=yaw)
    npt.assert_array_almost_equal(p, np.interp(u * co, u_p, p_c))
    npt.assert_array_almost_equal(ct, np.interp(u * co, u_p, ct_c) * co**2)


def test_DensityScaleAndSimpleYawModel():
    u_p, p_c = np.asarray(hornsrev1.power_curve).T.copy()
    _, ct_c = hornsrev1.ct_curve.T
    curve = PowerCtTabular(ws=u_p, power=p_c, power_unit='w', ct=ct_c, method='linear')
    u = np.arange(4, 25, 1.1)
    yaw = 30
    theta = np.deg2rad(yaw)
    co = np.cos(theta)
    p, ct = curve(u, yaw=yaw, Air_density=1.3)
    npt.assert_array_almost_equal(p, np.interp(u * co, u_p, p_c) * 1.3 / 1.225)
    npt.assert_array_almost_equal(ct, np.interp(u * co, u_p, ct_c) * co**2 * 1.3 / 1.225)


if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 100)
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(x, np.sin(x) / x)
    plt.plot(x, np.sin(x))
    plt.show()
