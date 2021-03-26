import numpy as np
from py_wake.examples.data.iea34_130rwt._iea34_130rwt import IEA34_130_1WT_Surrogate, IEA34_130_2WT_Surrogate
from py_wake.tests import npt
from py_wake.deficit_models.noj import NOJ
from py_wake.site.xrsite import UniformSite
from py_wake.turbulence_models.stf import STF2017TurbulenceModel


def test_one_turbine_case0():
    ti, ws, shear = 0.0592370641, 9.6833182032, 0.2

    # for i in [10, 14, 322]:
    #     print(sensors[i], mean_values[i], std_values[i])
    # "Free wind speed Vy gl. coo of gl. pos    0.00   0.00-110.00" 9.704401687125 0.7808578675666666
    # Aero rotor thrust 532.846421460325 26.557189388525
    # generator_servo inpvec   2  2: pelec [w] 3197471.4446449564 176585.66727525144

    wt = IEA34_130_1WT_Surrogate()
    assert wt.loadFunction.output_keys[1] == 'MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1'
    assert wt.loadFunction.wohler_exponents == [10, 10, 4, 4, 7]
    site = UniformSite(p_wd=[1], ti=ti, ws=ws)
    sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())([0], [0], wd=0, Alpha=shear)

    npt.assert_allclose(ws, 9.7, atol=.02)
    npt.assert_allclose(ti, 0.78 / 9.7, atol=.022)
    npt.assert_allclose(sim_res.Power, 3197471, atol=230)
    npt.assert_allclose(sim_res.CT, 532 * 1e3 / (1 / 2 * 1.225 * (65**2 * np.pi) * 9.7**2), atol=0.006)

    # for i in [28,29,1,2,9]:
    #     print(del_sensors[i], del_values[i])
    # MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1 2837.2258423768494
    # MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1 5870.8557931814585
    # MomentMx Mbdy:tower nodenr:   1 coo: tower  tower bottom moment 9154.009617791817
    # MomentMy Mbdy:tower nodenr:   1 coo: tower  tower bottom moment 5184.805475839391
    # MomentMz Mbdy:tower nodenr:  11 coo: tower  tower top/yaw bearing moment 2204.4440980948084
    loads = sim_res.loads(method='OneWT')
    npt.assert_allclose(loads.DEL.squeeze(), [2837, 5870, 9154, 5184, 2204], rtol=.11)
    f = 20 * 365 * 24 * 3600 / 1e7
    m = np.array([10, 10, 4, 4, 7])
    npt.assert_array_almost_equal(loads.LDEL.squeeze(), (loads.DEL.squeeze()**m * f)**(1 / m))

    loads = sim_res.loads(method='OneWT_WDAvg')
    npt.assert_allclose(loads.DEL.squeeze(), [2837, 5870, 9154, 5184, 2204], rtol=.11)
    npt.assert_array_almost_equal(loads.LDEL.squeeze(), (loads.DEL.squeeze()**m * f)**(1 / m))


def test_two_turbine_case0():
    ws, ti, shear, wdir, dist = 11.924050697, 0.2580934242, 0.2536493558, 20.5038383383, 3.8603095881

#     sensors = 'case,shaft_rot angle,shaft_rot angle speed,pitch1 angle,'.split(
#         ',')
#     mean_values = np.array("0,180.03464337189587,11.752259992475002,".split(","), dtype=np.float)
#     std_values = np.array("0,104.04391803126249,0.1227628373541667,".split(","), dtype=np.float)
#     for i in [10, 14, 322]:
#         print(sensors[i], mean_values[i], std_values[i])
    # "Free wind speed Vy gl. coo of gl. pos    0.00   0.00-110.00" 11.107441047808335 0.8527054147583332
    # Aero rotor thrust 381.01636820417497 37.541737944825
    # generator_servo inpvec   2  2: pelec [w] 3398038.328107514 13065.24538016912
    # ref from simulation statistic
    ws_ref = 11.1
    ws_std_ref = 0.85
    power_ref = 3398038
    thrust_ref = 381

    wt = IEA34_130_2WT_Surrogate()
    site = UniformSite(p_wd=[1], ti=ti, ws=ws)
    sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())([0, 0], [0, dist * 130], wd=wdir, Alpha=shear)

    npt.assert_allclose(ws, ws_ref, atol=.9)
    npt.assert_allclose(ti, ws_std_ref / ws_ref, atol=.19)
    npt.assert_allclose(sim_res.Power.sel(wt=0), power_ref, atol=1060)
    npt.assert_allclose(sim_res.CT.sel(wt=0), thrust_ref * 1e3 / (1 / 2 * 1.225 * (65**2 * np.pi) * ws_ref**2),
                        atol=0.06)

    # del_sensors = 'case,MomentMx Mbdy:tower nodenr:   1 coo: tower  tower bottom moment,'.split(",")
    # del_values = np.array("0, 11235.755826844477, 7551.985936926388, ".split(","), dtype=float)
    # for i in [28, 29, 1, 2, 9]:
    #     print(del_sensors[i], del_values[i])
    # MomentMx Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1 3863.6241845634827
    # MomentMy Mbdy:blade1 nodenr:   1 coo: blade1  blade root moment blade1 5774.294439888724
    # MomentMx Mbdy:tower nodenr:   1 coo: tower  tower bottom moment 11235.755826844477
    # MomentMy Mbdy:tower nodenr:   1 coo: tower  tower bottom moment 7551.985936926388
    # MomentMz Mbdy:tower nodenr:  11 coo: tower  tower top/yaw bearing moment 2384.173023068421
    loads = sim_res.loads(method='TwoWT')
    npt.assert_allclose(loads.DEL.sel(wt=0).squeeze(), [3863], rtol=.12)

    f = 20 * 365 * 24 * 3600 / 1e7
    m = 10
    npt.assert_array_almost_equal(loads.LDEL.sel(wt=0).squeeze(), (loads.DEL.sel(wt=0).squeeze()**m * f)**(1 / m))

    loads = sim_res.loads(method='TwoWT', softmax_base=100)
    npt.assert_allclose(loads.DEL.sel(wt=0).squeeze(), [3863], rtol=.11)
    npt.assert_array_almost_equal(loads.LDEL.sel(wt=0).squeeze(), (loads.DEL.sel(wt=0).squeeze()**m * f)**(1 / m))
