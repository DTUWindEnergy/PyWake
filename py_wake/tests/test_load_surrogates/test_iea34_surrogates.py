import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from py_wake.examples.data.iea34_130rwt._iea34_130rwt import IEA34_130_1WT_Surrogate, IEA34_130_2WT_Surrogate
from py_wake.tests import npt
from py_wake.deficit_models.noj import NOJ
from py_wake.site.xrsite import UniformSite
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from pathlib import Path
from py_wake.wind_turbines.wind_turbine_functions import FunctionSurrogates
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
from py_wake.examples.data import example_data_path
from py_wake.utils.gradients import plot_gradients, fd, autograd, cs
import pytest


@pytest.fixture(scope='module')
def iea34_130_1WT_Surrogate():
    return IEA34_130_1WT_Surrogate()


@pytest.fixture(scope='module')
def iea34_130_2WT_Surrogate():
    return IEA34_130_2WT_Surrogate()


def test_one_turbine_case0(iea34_130_1WT_Surrogate):
    ws, ti, shear = 9.2984459862, 0.0597870198, 0.2

    if 0:
        f = r'C:\mmpe\programming\python\Topfarm\iea-3_4-130-rwt\turbine_model\res/'
        print(pd.concat([pd.read_csv(f + 'stats_one_turbine_mean.csv').iloc[0, [10, 14, 322]],
                         pd.read_csv(f + 'stats_one_turbine_std.csv').iloc[0, [10, 14, 322]]],
                        axis=1))
        # Free wind speed Vy, gl. coo, of gl. pos    0.00...  9.309756e+00       0.401308
        # Aero rotor thrust                                   5.408776e+02      11.489005
        # generator_servo inpvec   2  2: pelec [w]            2.931442e+06  116491.192548

        print(pd.read_csv(f + 'stats_one_turbine_del.csv').iloc[0, [28, 29, 1, 2, 9]])
        # MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1      1822.247387
        # MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1      5795.166623
        # MomentMx Mbdy:tower nodenr:   1 coo: tower  tower bottom moment             4385.405881
        # MomentMy Mbdy:tower nodenr:   1 coo: tower  tower bottom moment             2468.357017
        # MomentMz Mbdy:tower nodenr:  11 coo: tower  tower top/yaw bearing moment    1183.884786

    ws_ref = 9.309756e+00
    ws_std_ref = 0.401308
    power_ref = 2.931442e+06
    thrust_ref = 5.408776e+02
    ref_dels = [1822, 5795, 4385, 2468, 1183]

    wt = iea34_130_1WT_Surrogate
    assert wt.loadFunction.output_keys[1] == 'del_blade_edge'
    assert wt.loadFunction.wohler_exponents == [10, 10, 4, 4, 7]
    site = UniformSite(p_wd=[1], ti=ti, ws=ws)
    sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())([0], [0], wd=0, Alpha=shear)

    npt.assert_allclose(ws, ws_ref, rtol=.0013)
    npt.assert_allclose(ti, ws_std_ref / ws_ref, atol=.02)
    npt.assert_allclose(sim_res.Power, power_ref, rtol=0.003)
    npt.assert_allclose(sim_res.CT, thrust_ref * 1e3 / (1 / 2 * 1.225 * (65**2 * np.pi) * ws_ref**2), rtol=0.011)

    loads = sim_res.loads(method='OneWT')
    npt.assert_allclose(loads.DEL.squeeze(), ref_dels, rtol=.11)
    f = 20 * 365 * 24 * 3600 / 1e7
    m = np.array([10, 10, 4, 4, 7])
    npt.assert_array_almost_equal(loads.LDEL.squeeze(), (loads.DEL.squeeze()**m * f)**(1 / m))

    loads = sim_res.loads(method='OneWT_WDAvg')
    npt.assert_allclose(loads.DEL.squeeze(), ref_dels, rtol=.11)
    npt.assert_array_almost_equal(loads.LDEL.squeeze(), (loads.DEL.squeeze()**m * f)**(1 / m))


def test_two_turbine_case0(iea34_130_2WT_Surrogate):
    if 0:
        i = 0
        f = r'C:\mmpe\programming\python\Topfarm\iea-3_4-130-rwt\turbine_model\res/'
        print(list(pd.DataFrame(eval(Path(f + 'input_two_turbines_dist.json').read_text())).iloc[i]))
        # [10.9785338191, 0.2623204277, 0.4092031776, -38.4114616871, 5.123719529]

        print(pd.concat([pd.read_csv(f + 'stats_two_turbines_mean.csv').iloc[i, [12, 14, 322]],
                         pd.read_csv(f + 'stats_two_turbines_std.csv').iloc[i, [12, 14, 322]]],
                        axis=1))
        # Free wind speed Abs_vhor, gl. coo, of gl. pos  ...  1.103937e+01     0.914252
        # Aero rotor thrust                                   4.211741e+02    41.015962
        # generator_servo inpvec   2  2: pelec [w]            3.399746e+06  3430.717100

        print(pd.read_csv(f + 'stats_two_turbines_del.csv').iloc[i, [28, 29, 1, 2, 9]])
        # MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1       4546.998501
        # MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1       5931.157693
        # MomentMx Mbdy:tower nodenr:   1 coo: tower  tower bottom moment             11902.153031
        # MomentMy Mbdy:tower nodenr:   1 coo: tower  tower bottom moment              7599.336676
        # MomentMz Mbdy:tower nodenr:  11 coo: tower  tower top/yaw bearing moment     2407.279074

    ws, ti, shear, wdir, dist = [10.9785338191, 0.2623204277, 0.4092031776, -38.4114616871 % 360, 5.123719529]

    # ref from simulation statistic (not updated yet)
    ws_ref = 1.103937e+01
    ws_std_ref = 0.914252
    thrust_ref = 4.211741e+02
    power_ref = 3.399746e+06
    ref_dels = [4546, 5931, 11902, 7599, 2407]

    wt = iea34_130_2WT_Surrogate
    site = UniformSite(p_wd=[1], ti=ti, ws=ws)
    sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())([0, 0], [0, dist * 130], wd=wdir, Alpha=shear)

    npt.assert_allclose(ws, ws_ref, rtol=.006)
    # npt.assert_allclose(ti, ws_std_ref / ws_ref, atol=.19)
    npt.assert_allclose(sim_res.Power.sel(wt=0), power_ref, rtol=0.002)
    npt.assert_allclose(sim_res.CT.sel(wt=0), thrust_ref * 1e3 / (1 / 2 * 1.225 * (65**2 * np.pi) * ws_ref**2),
                        rtol=0.03)

    loads = sim_res.loads(method='TwoWT')
    npt.assert_allclose(loads.DEL.sel(wt=0).squeeze(), ref_dels, rtol=.05)

    f = 20 * 365 * 24 * 3600 / 1e7
    m = loads.m.values
    npt.assert_array_almost_equal(loads.LDEL.sel(wt=0).squeeze(), (loads.DEL.sel(wt=0).squeeze()**m * f)**(1 / m))

    loads = sim_res.loads(method='TwoWT', softmax_base=100)
    npt.assert_allclose(loads.DEL.sel(wt=0).squeeze(), ref_dels, rtol=.05)
    npt.assert_array_almost_equal(loads.LDEL.sel(wt=0).squeeze(), (loads.DEL.sel(wt=0).squeeze()**m * f)**(1 / m))


def test_two_turbine_case0_time_series(iea34_130_2WT_Surrogate):
    # same as test_two_turbine_case0
    ws, ti, shear, wdir, dist = [10.9785338191, 0.2623204277, 0.4092031776, -38.4114616871 % 360, 5.123719529]

    # ref from simulation statistic (not updated yet)
    ws_ref = 1.103937e+01
    thrust_ref = 4.211741e+02
    power_ref = 3.399746e+06
    ref_dels = [4546, 5931, 11902, 7599, 2407]

    wt = iea34_130_2WT_Surrogate
    site = UniformSite(p_wd=[1], ti=ti, ws=ws)
    wfm = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())
    sim_res = wfm([0, 0], [0, dist * 130], wd=wdir, time=True, Alpha=shear)
    assert sim_res.dw_ijl.dims == ('wt', 'wt', 'time')

    npt.assert_allclose(ws, ws_ref, rtol=.006)
    # npt.assert_allclose(ti, ws_std_ref / ws_ref, atol=.19)
    npt.assert_allclose(sim_res.Power.sel(wt=0), power_ref, rtol=0.002)
    npt.assert_allclose(sim_res.CT.sel(wt=0), thrust_ref * 1e3 / (1 / 2 * 1.225 * (65**2 * np.pi) * ws_ref**2),
                        rtol=0.03)
    sim_res['duration'] = ('time', [3600 * 24 * 365 * 20])
    loads = sim_res.loads(method='TwoWT')
    npt.assert_allclose(loads.DEL.sel(wt=0).squeeze(), ref_dels, rtol=.05)

    f = 20 * 365 * 24 * 3600 / 1e7
    m = loads.m.values
    npt.assert_array_almost_equal(loads.LDEL.sel(wt=0).squeeze(), (loads.DEL.sel(wt=0).squeeze()**m * f)**(1 / m))

    loads = sim_res.loads(method='TwoWT', softmax_base=100)
    npt.assert_allclose(loads.DEL.sel(wt=0).squeeze(), ref_dels, rtol=.05)
    npt.assert_array_almost_equal(loads.LDEL.sel(wt=0).squeeze(), (loads.DEL.sel(wt=0).squeeze()**m * f)**(1 / m))


def test_functionSurrogate():
    surrogate_path = Path(example_data_path) / 'iea34_130rwt' / 'one_turbine'
    load_sensors = ['del_blade_flap', 'del_blade_edge']

    loadFunction = FunctionSurrogates(
        [TensorflowSurrogate(surrogate_path / s, 'operating') for s in load_sensors],
        input_parser=lambda ws, TI_eff=.1, Alpha=0: [ws, TI_eff, Alpha])

    assert loadFunction.output_keys == [
        'MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1',
        'MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1']
    npt.assert_array_almost_equal(loadFunction(np.array([10, 11])), [[2077.9673, 2116.636], [5710.0894, 5653.4956]], 3)


def test_ws_gradients(iea34_130_1WT_Surrogate):
    wt = iea34_130_1WT_Surrogate
    ws = np.linspace(3, 20, 1000)
    x = 7.
    plt.plot(ws, wt.power(ws, TI_eff=0.1))

    method_lst = [lambda *args, **kwargs: fd(*args, step=.001, **kwargs), cs, autograd]
    dpdx_lst = [method(wt.power)(x, TI_eff=0.1) for method in method_lst]
    npt.assert_allclose(dpdx_lst[0], dpdx_lst[1], rtol=.0005)
    npt.assert_array_almost_equal(dpdx_lst[1], dpdx_lst[2], 10)
    if 0:
        y = wt.power(x, TI_eff=0.1)
        for method, dpdx in zip(method_lst, dpdx_lst):
            plot_gradients(y, dpdx, x, label=method.__name__)
        plt.show()


def test_ti_gradients(iea34_130_1WT_Surrogate):
    wt = iea34_130_1WT_Surrogate
    ti_low, ti_high = wt.powerCtFunction.function_surrogate_lst[0].input_space['ti']
    ws_lst = np.linspace(3, 20, 1000)
    ti_lst = np.linspace(ti_low, ti_high, 100)

    def power(ws, ti):
        return wt.power(ws, TI_eff=ti)
    if 0:
        for ti in np.linspace(ti_low, ti_high, 5):
            plt.plot(ws_lst, wt.power(ws_lst, TI_eff=ti), label=ti)
        plt.show()
    ws = 10
    x = .1

    method_lst = [lambda *args, **kwargs: fd(*args, step=.001, **kwargs), cs, autograd]
    dpdx_lst = [method(power, argnum=1)(ws, x) for method in method_lst]
    npt.assert_allclose(dpdx_lst[0], dpdx_lst[1], rtol=.005)
    npt.assert_array_almost_equal(dpdx_lst[1], dpdx_lst[2], 10)
    if 0:
        plt.plot(ti_lst, wt.power(ti_lst * 0 + ws, TI_eff=ti))
        y = power(ws, x)
        for method, dpdx in zip(method_lst, dpdx_lst):
            plot_gradients(y, dpdx, x, label=method.__name__, step=.1)
        plt.show()


def test_ws_and_ti_gradients(iea34_130_1WT_Surrogate):
    wt = iea34_130_1WT_Surrogate
    ws_lst = np.linspace(9, 11, 100)

    def power(ws):
        return wt.power(ws, TI_eff=ws / 50)

    ws = 10

    method_lst = [lambda *args, **kwargs: fd(*args, step=.001, **kwargs), cs, autograd]
    dpdx_lst = [method(power)(ws) for method in method_lst]
    npt.assert_allclose(dpdx_lst[0], dpdx_lst[1], rtol=.005)
    npt.assert_array_almost_equal(dpdx_lst[1], dpdx_lst[2], 10)
    if 0:
        plt.plot(ws_lst, wt.power(ws_lst, TI_eff=0.001))
        plt.plot(ws_lst, power(ws_lst))
        y = power(ws)
        for method, dpdx in zip(method_lst, dpdx_lst):
            plot_gradients(y, dpdx, ws, label=method.__name__, step=.1)
        plt.show()
