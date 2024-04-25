import pytest

import matplotlib.pyplot as plt
from py_wake import np
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.deficit_models.fuga import FugaDeficit, Fuga
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, IEA37SimpleBastankhahGaussianDeficit,\
    ZongGaussianDeficit, NiayifarGaussianDeficit, BastankhahGaussian, ZongGaussian,\
    NiayifarGaussian, CarbajofuertesGaussianDeficit, TurboGaussianDeficit, IEA37SimpleBastankhahGaussian,\
    BlondelSuperGaussianDeficit2020, BlondelSuperGaussianDeficit2023
from py_wake.deficit_models.gcl import GCLDeficit, GCL, GCLLocal
from py_wake.deficit_models.noj import NOJDeficit, NOJ, NOJLocalDeficit, NOJLocal, TurboNOJDeficit
from py_wake.deficit_models import VortexDipole
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm
from py_wake.flow_map import HorizontalGrid, XYGrid, YZGrid
from py_wake.superposition_models import SquaredSum, WeightedSum
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.turbulence_models.gcl_turb import GCLTurbulence
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.utils.model_utils import get_models
from numpy import newaxis as na
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.rotor_avg_models.rotor_avg_model import CGIRotorAvg
import warnings
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.site._site import UniformSite


class GCLLocalDeficit(GCLDeficit):
    def __init__(self):
        GCLDeficit.__init__(self, use_effective_ws=True, use_effective_ti=True)


@pytest.mark.parametrize(
    'deficitModel,aep_ref',
    # test that the result is equal to last run (no evidens that these number are correct)
    [
        (NOJDeficit(ct2a=ct2a_mom1d), (367205.0846866496, [9833.86287, 8416.99088, 10820.37673, 13976.26422, 22169.66036,
                                                           25234.9215, 37311.64388, 42786.37028, 24781.33444, 13539.82115,
                                                           14285.22744, 31751.29488, 75140.15677, 17597.10319, 11721.21226,
                                                           7838.84383])),
        (NOJLocalDeficit(ct2a=ct2a_mom1d), (335151.6404628441, [8355.71335, 7605.92379, 10654.172, 13047.6971, 19181.46408,
                                                                23558.34198, 36738.52415, 38663.44595, 21056.39764, 12042.79324,
                                                                13813.46269, 30999.42279, 63947.61202, 17180.40299, 11334.12323,
                                                                6972.14345])),
        (TurboNOJDeficit(ct2a=ct2a_mom1d), (354154.2962989713, [9320.85263, 8138.29496, 10753.75662, 13398.00865, 21189.29438,
                                                                24190.84895, 37081.91938, 41369.66605, 23488.54863, 12938.48451,
                                                                14065.00719, 30469.75602, 71831.78532, 16886.85274, 11540.51872,
                                                                7490.70156])),

        (BastankhahGaussianDeficit(ct2a=ct2a_mom1d), (355971.9717035484,
                                                      [9143.74048, 8156.71681, 11311.92915, 13955.06316, 19807.65346,
                                                       25196.64182, 39006.65223, 41463.31044, 23042.22602, 12978.30551,
                                                       14899.26913, 32320.21637, 67039.04091, 17912.40907, 12225.04134,
                                                       7513.75582])),
        (IEA37SimpleBastankhahGaussianDeficit(), read_iea37_windfarm(iea37_path + 'iea37-ex16.yaml')[2]),
        (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc',
                     smooth2zero_x=250, smooth2zero_y=50),
         (404441.6306021485, [9912.33731, 9762.05717, 12510.14066, 15396.76584, 23017.66483,
                              27799.7161, 43138.41606, 49623.79059, 24979.09001, 15460.45923,
                              16723.02619, 35694.35526, 77969.14805, 19782.41376, 13721.45739,
                              8950.79218])),
        (GCLDeficit(), (370863.6246093183,
                        [9385.75387, 8768.52105, 11450.13309, 14262.42186, 21178.74926,
                         25751.59502, 39483.21753, 44573.31533, 23652.09976, 13924.58752,
                         15106.11692, 32840.02909, 71830.22035, 18200.49805, 12394.7626,
                         8061.6033])),
        (GCLLocalDeficit(), (381187.36105425097,
                             [9678.85358, 9003.65526, 11775.06899, 14632.42259, 21915.85495,
                              26419.65189, 40603.68618, 45768.58091, 24390.71103, 14567.43106,
                              15197.82861, 32985.67922, 75062.92788, 18281.21981, 12470.01322,
                              8433.77587])),
        (ZongGaussianDeficit(eps_coeff=0.35, ct2a=ct2a_mom1d),
         (354263.1062694012, [8960.58672, 8095.48341, 11328.51572, 13924.50408, 19938.78428,
                              25141.4657, 39063.84731, 41152.04065, 22580.67854, 12944.55424,
                              14778.95385, 32294.41749, 66540.62665, 17898.1109, 12126.32111,
                              7494.21561])),
        (NiayifarGaussianDeficit(ct2a=ct2a_mom1d),
         (362228.4003016239, [9210.01349, 8330.06642, 11392.04851, 14082.13366, 20643.45247,
                              25426.07466, 39282.9259, 42344.50431, 23209.23399, 13390.08421,
                              14807.66824, 32535.89096, 69640.3283, 18031.93957, 12149.88163,
                              7752.15401])),
        (CarbajofuertesGaussianDeficit(ct2a=ct2a_mom1d),
         (352354.8742100223, [8832.53107, 8111.89563, 11297.1665, 13872.13038, 19672.92342,
                              25046.90207, 38955.74655, 41235.46947, 22257.97829, 12945.13725,
                              14736.20095, 32200.78736, 65757.99084, 17846.2195, 12091.2418,
                              7494.55314])),
        (TurboGaussianDeficit(ct2a=ct2a_mom1d, rotorAvgModel=None),
         (333612.8032572539, [7696.10577, 7913.67274, 11102.28219, 13503.62483, 17184.49031,
                              24381.54484, 38283.7317, 40227.83645, 19394.18655, 12536.27008,
                              14613.38763, 31681.28939, 58287.76333, 17558.30497, 11990.4719,
                              7257.84057])),
        (BlondelSuperGaussianDeficit2020(),
         (338236.360935599, [8184.10433669, 7688.07418809, 11210.94605493, 13517.06684326,
                             18220.2151193, 24405.81513366, 38658.43467216, 39081.04378946,
                             20623.94292847, 12206.04086561, 14690.26714648, 31821.77172395,
                             61172.26772292, 17636.16264219, 12053.55253044, 7066.65523799])),
        (BlondelSuperGaussianDeficit2023(),
         (355892.77826332045, [8979.31725425, 8180.44857196, 11292.09175721, 13964.6406542,
                               20066.69629875, 25213.93451452, 38938.24743865, 41583.94690748,
                               22627.8794807, 12933.21646472, 14824.4675625, 32363.56815599,
                               67336.57050905, 17936.43536356, 12163.66569231, 7487.65163747]))
    ])
def test_IEA37_ex16(deficitModel, aep_ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel,
                                 superpositionModel=SquaredSum(), turbulenceModel=GCLTurbulence())

    aep_ilk = wf_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000

    # check if ref is reasonable
    aep_est = 16 * 3.35 * 24 * 365 * .8  # n_wt * P_rated * hours_pr_year - 20% wake loss = 375628.8
    npt.assert_allclose(aep_ref[0], aep_est, rtol=.12)
    npt.assert_allclose(aep_ref[1], [9500, 8700, 11500, 14300, 21300, 25900, 39600, 44300, 23900,
                                     13900, 15200, 33000, 72100, 18300, 12500, 8000], rtol=.2)

    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)
    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


@pytest.mark.parametrize('deficitModel', get_models(WakeDeficitModel))
def test_huge_distance(deficitModel):
    ref = {"NOJDeficit": 9.799733,
           "FugaDeficit": 9.8,
           "FugaYawDeficit": 9.8,
           "FugaMultiLUTDeficit": 9.8,
           "BastankhahGaussianDeficit": 9.79916,
           "CarbajofuertesGaussianDeficit": 9.798723,
           "IEA37SimpleBastankhahGaussianDeficit": 9.799151,
           "NiayifarGaussianDeficit": 9.799162,
           "TurboGaussianDeficit": 9.67046,
           "ZongGaussianDeficit": 9.799159,
           "GCLDeficit": 9.728704,
           "NoWakeDeficit": 9.8,
           "NOJLocalDeficit": 9.797536,
           "TurboNOJDeficit": 9.795322,
           "BlondelSuperGaussianDeficit2020": 9.79177032,
           "BlondelSuperGaussianDeficit2023": 9.7990636, }
    site = IEA37Site(16)

    windTurbines = IEA37_WindTurbines()
    wfm = All2AllIterative(site, windTurbines, wake_deficitModel=deficitModel(), turbulenceModel=GCLTurbulence())
    sim_res = wfm([0, 100000], [0, 0], wd=[0, 90, 180, 270], yaw=0)
    # print(f'"{deficitModel.__name__}": {np.round(sim_res.WS_eff.sel(wt=1, ws=9.8, wd=270).item(),6)},')

    npt.assert_array_almost_equal([9.8, 9.8, 9.8, ref[deficitModel.__name__]], sim_res.WS_eff.sel(wt=1).squeeze())


@pytest.mark.parametrize('deficitModel', get_models(BlockageDeficitModel))
def test_huge_distance_blockage(deficitModel):
    if deficitModel is None:
        return
    ref = {"FugaDeficit": 9.8,
           "FugaYawDeficit": 9.8,
           "FugaMultiLUTDeficit": 9.8,
           "HybridInduction": 9.800001,
           "SelfSimilarityDeficit2020": 9.800001,
           "VortexDipole": 9.800001,
           "RankineHalfBody": 9.799999,
           "Rathmann": 9.800001,
           "RathmannScaled": 9.800001,
           "SelfSimilarityDeficit": 9.800001,
           "VortexCylinder": 9.799999, }
    site = IEA37Site(16)

    windTurbines = IEA37_WindTurbines()
    wfm = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                           blockage_deficitModel=deficitModel(),
                           turbulenceModel=GCLTurbulence())
    sim_res = wfm([0, 100000], [0, 0], wd=[0, 90, 180, 270], yaw=0)
    # print(f'"{deficitModel.__name__}": {np.round(sim_res.WS_eff.sel(wt=0, ws=9.8, wd=270).item(),6)},')

    npt.assert_array_almost_equal([9.8, 9.8, 9.8, ref[deficitModel.__name__]], sim_res.WS_eff.sel(wt=1).squeeze())


@pytest.mark.parametrize(
    'deficitModel,ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJDeficit(ct2a=ct2a_mom1d),
      [3.27, 3.27, 9.0, 7.46, 7.46, 7.46, 7.46, 7.31, 7.31, 7.31, 7.31, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3]),
     (NOJLocalDeficit(ct2a=ct2a_mom1d),
      [3.09, 3.09, 9., 9., 5.54, 5.54, 5.54, 5.54, 5.54, 5.54, 6.73, 6.73, 6.73, 6.73, 6.73, 6.73, 6.73]),
     (TurboNOJDeficit(ct2a=ct2a_mom1d),
      [3.51, 3.51, 3.51, 7.45, 7.45, 7.45, 7.45, 7.45, 7.13, 7.13, 7.13, 7.96, 7.96, 7.96, 7.96, 7.96, 7.96]),
     (BastankhahGaussianDeficit(ct2a=ct2a_mom1d),
      [0.18, 3.6, 7.27, 8.32, 7.61, 6.64, 5.96, 6.04, 6.8, 7.69, 8.08, 7.87, 7.59, 7.46, 7.55, 7.84, 8.19]),
     (IEA37SimpleBastankhahGaussianDeficit(),
      [3.32, 4.86, 7.0, 8.1, 7.8, 7.23, 6.86, 6.9, 7.3, 7.82, 8.11, 8.04, 7.87, 7.79, 7.85, 8.04, 8.28]),
     (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc'),
      [7.06, 7.87, 8.77, 8.85, 8.52, 7.96, 7.49, 7.55, 8.06, 8.58, 8.69, 8.45, 8.18, 8.05, 8.15, 8.41, 8.68]),
     (GCLDeficit(),
      [2.39, 5.01, 7.74, 8.34, 7.95, 7.58, 7.29, 7.32, 7.61, 7.92, 8.11, 8.09, 7.95, 7.83, 7.92, 8.1, 8.3]),
     (GCLLocalDeficit(),
      [3.05, 5.24, 7.61, 8.36, 8.08, 7.81, 7.61, 7.63, 7.82, 8.01, 8.11, 8.07, 7.94, 7.83, 7.92, 8.1, 8.3]),
     (ZongGaussianDeficit(ct2a=ct2a_mom1d, eps_coeff=0.35),
      [6.34, 7.08, 8.09, 8.36, 7.5, 6.2, 5.23, 5.34, 6.43, 7.64, 8.08, 7.75, 7.36, 7.19, 7.32, 7.69, 8.14]),
     (NiayifarGaussianDeficit(ct2a=ct2a_mom1d),
      [0.18, 3.6, 7.27, 8.32, 7.61, 6.64, 5.97, 6.04, 6.8, 7.69, 8.08, 7.87, 7.59, 7.46, 7.56, 7.84, 8.19]),
     (CarbajofuertesGaussianDeficit(ct2a=ct2a_mom1d),
      [0.17, 3.48, 7.15, 8.32, 7.54, 6.37, 5.52, 5.61, 6.57, 7.66, 8.06, 7.71, 7.29, 7.1, 7.24, 7.65, 8.13]),
     (TurboGaussianDeficit(ct2a=ct2a_mom1d),
      [3.27, 4.83, 7., 8.14, 7.54, 6.37, 5.52, 5.61, 6.57, 7.69, 8.02, 7.31, 6.43, 6., 6.32, 7.18, 8.05]),
     ])
def test_deficitModel_wake_map(deficitModel, ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel, superpositionModel=SquaredSum(),
                                 turbulenceModel=GCLTurbulence())

    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)

    flow_map = wf_model(x, y, wd=0, ws=9).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map.X, flow_map.Y
    Z = flow_map.WS_eff_xylk[:, :, 0, 0]

    mean_ref = [3.2, 4.9, 8., 8.2, 7.9, 7.4, 7., 7., 7.4, 7.9, 8.1, 8.1, 8., 7.8, 7.9, 8.1, 8.4]

    if 0:
        flow_map.plot_wake_map()
        plt.plot(X[49, 100:133:2], Y[49, 100:133:2], '.-')
        windTurbines.plot(x, y)
        plt.figure()
        plt.plot(Z[49, 100:133:2])
        plt.plot(ref, label='ref')
        plt.plot(mean_ref, label='Mean ref')
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(ref, Z[49, 100:133:2], 2)

    # check that ref is reasonable
    npt.assert_allclose(ref[3:], mean_ref[3:], atol=2.6)


@pytest.mark.parametrize(
    'deficitModel,wake_radius_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [
        (NOJDeficit(), [100., 75., 150., 100., 100.]),
        (NOJLocalDeficit(), [71., 46., 92., 71., 61.5]),
        (TurboNOJDeficit(), [99.024477, 61.553917, 123.107833, 92.439673, 97.034049]),
        (BastankhahGaussianDeficit(), [83.336286, 57.895893, 115.791786, 75.266662, 83.336286]),
        (IEA37SimpleBastankhahGaussianDeficit(), [103.166178, 67.810839, 135.621678, 103.166178, 103.166178]),
        (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc'), [90., 90., 115., 90., 90.]),
        (GCLDeficit(), [156.949964, 97.763333, 195.526667, 113.225695, 111.340236]),
        (GCLLocalDeficit(), [156.949964, 97.763333, 195.526667, 113.225695, 111.340236]),
        (ZongGaussianDeficit(eps_coeff=0.35), [91.15734, 66.228381, 132.456762, 94.90156, 79.198215]),
        (NiayifarGaussianDeficit(), [92.880786, 67.440393, 134.880786, 84.811162, 73.880786]),
        (CarbajofuertesGaussianDeficit(), [89.63, 62.315, 124.63, 89.63, 78.815]),
        (TurboGaussianDeficit(), [76.674176, 41.548202, 83.096405, 64.831198, 76.143396]),
    ])
def test_wake_radius(deficitModel, wake_radius_ref):

    mean_ref = [105, 68, 135, 93, 123]
    # check that ref is reasonable
    npt.assert_allclose(wake_radius_ref, mean_ref, rtol=.5)

    npt.assert_array_almost_equal(deficitModel.wake_radius(
        D_src_il=np.reshape([100, 50, 100, 100, 100], (5, 1)),
        dw_ijlk=np.reshape([500, 500, 1000, 500, 500], (5, 1, 1, 1)),
        WS_ilk=np.reshape([10, 10, 10, 10, 10], (5, 1, 1)),
        ct_ilk=np.reshape([.8, .8, .8, .4, .8], (5, 1, 1)),
        TI_ilk=np.reshape([.1, .1, .1, .1, .05], (5, 1, 1)),
        TI_eff_ilk=np.reshape([.1, .1, .1, .1, .05], (5, 1, 1)))[:, 0, 0, 0],
        wake_radius_ref)

    # Check that it works when called from WindFarmModel
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel, turbulenceModel=GCLTurbulence())
    wfm(x=[0, 500], y=[0, 0], wd=[30], ws=[10])

    if 0:
        ax1, ax2 = plt.subplots(2, 1)[1]
        sim_res = wfm([0], [0], wd=[270], ws=10)
        fm = sim_res.flow_map(HorizontalGrid(x=np.arange(-100, 1500, 10)))
        fm.WS_eff.plot(ax=ax1)
        x = np.arange(0, 1500, 10)
        wr = deficitModel.wake_radius(
            WS_ilk=np.reshape([10], (1, 1, 1)),
            D_src_il=np.reshape([130], (1, 1)),
            dw_ijlk=np.reshape(x, (1, len(x), 1, 1)),
            ct_ilk=sim_res.CT.values,
            TI_ilk=np.reshape(sim_res.TI.values, (1, 1, 1)),
            TI_eff_ilk=sim_res.TI_eff.values)[0, :, 0, 0]
        ax1.set_title(deficitModel.__class__.__name__)
        ax1.plot(x, wr)
        ax1.plot(x, -wr)
        ax1.axvline(500, linestyle='--', color='k')
        ax1.axis('equal')
        fm.WS_eff.sel(x=500).plot(ax=ax2)
        ax2.axvline(-wr[x == 500], color='k')
        ax2.axvline(wr[x == 500], color='k')
        ax2.grid()
        plt.show()


def test_wake_radius_not_implemented():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    class MyDeficitModel(WakeDeficitModel):

        def calc_deficit(self, WS_ilk, dw_ijlk, cw_ijlk, **_):
            # 10% deficit in downstream triangle
            ws_10pct_ijlk = 0.1 * WS_ilk[:, na]
            triangle_ijlk = ((.2 * dw_ijlk) > cw_ijlk)
            return ws_10pct_ijlk * triangle_ijlk

    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=MyDeficitModel(),
                            turbulenceModel=GCLTurbulence())
    with pytest.raises(NotImplementedError, match="wake_radius not implemented for MyDeficitModel"):
        wfm(x, y)


@pytest.mark.parametrize(
    'deficitModel,aep_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(BastankhahGaussianDeficit(ct2a=ct2a_mom1d), (345846.3355259293,
                                                   [8835.30563, 7877.90062, 11079.66832, 13565.65235, 18902.99769,
                                                    24493.53897, 38205.75284, 40045.9948, 22264.97018, 12662.90784,
                                                    14650.96535, 31289.90349, 65276.92307, 17341.39229, 12021.3049,
                                                    7331.15717])),
     (ZongGaussianDeficit(ct2a=ct2a_mom1d, eps_coeff=0.35),
      (342944.4057168523, [8674.44232, 7806.22033, 11114.78804, 13549.48197, 18895.50866,
                           24464.34244, 38326.85532, 39681.61999, 21859.59465, 12590.25899,
                           14530.24656, 31158.81189, 63812.14454, 17268.73912, 11922.25359,
                           7289.09731])),
     (NiayifarGaussianDeficit(ct2a=ct2a_mom1d),
      (349044.6891842835, [8888.36909, 8025.38191, 11134.3202, 13643.08009, 19503.25093,
                           24633.33906, 38394.20758, 40795.69138, 22398.69011, 12972.04355,
                           14504.45911, 31328.69308, 66049.04782, 17362.89014, 11901.09465,
                           7510.13048])),

     ])
def test_IEA37_ex16_convection(deficitModel, aep_ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel,
                                 superpositionModel=WeightedSum(), turbulenceModel=GCLTurbulence())

    aep_ilk = wf_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000

    # check if ref is reasonable
    aep_est = 16 * 3.35 * 24 * 365 * .8  # n_wt * P_rated * hours_pr_year - 20% wake loss = 375628.8
    npt.assert_allclose(aep_ref[0], aep_est, rtol=.11)
    npt.assert_allclose(aep_ref[1], [9500, 8700, 11500, 14300, 21300, 25900, 39600, 44300, 23900,
                                     13900, 15200, 33000, 72100, 18300, 12500, 8000], rtol=.15)

    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)
    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


@pytest.mark.parametrize(
    'deficitModel,ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(BastankhahGaussianDeficit(ct2a=ct2a_mom1d),
      [0.17, 3.55, 7.61, 8.12, 7.58, 6.63, 5.93, 6.04, 6.65, 7.35, 7.69, 7.65, 7.54, 7.45, 7.55, 7.84, 8.19]),
     (ZongGaussianDeficit(ct2a=ct2a_mom1d, eps_coeff=0.35),
      [6.34, 7.05, 7.9, 8.15, 7.45, 6.19, 5.21, 5.26, 6.38, 7.32, 7.7, 7.54, 7.34, 7.18, 7.32, 7.69, 8.14]),
     (NiayifarGaussianDeficit(ct2a=ct2a_mom1d),
      [0.17, 3.55, 7.61, 8.12, 7.58, 6.63, 5.93, 6.04, 6.65, 7.35, 7.69, 7.65, 7.55, 7.45, 7.56, 7.84, 8.19]),
     ])
def test_deficitModel_wake_map_convection(deficitModel, ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel, superpositionModel=WeightedSum(),
                                 turbulenceModel=GCLTurbulence())

    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)

    flow_map = wf_model(x, y, wd=0, ws=9).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map.X, flow_map.Y
    Z = flow_map.WS_eff_xylk[:, :, 0, 0]

    mean_ref = [3.2, 4.9, 8., 8.2, 7.9, 7.4, 7., 7., 7.4, 7.9, 8.1, 8.1, 8., 7.8, 7.9, 8.1, 8.4]

    if 0:
        flow_map.plot_wake_map()
        plt.plot(X[49, 100:133:2], Y[49, 100:133:2], '.-')
        windTurbines.plot(x, y)
        plt.figure()
        plt.plot(Z[49, 100:133:2], label='Actual')
        plt.plot(ref, label='Reference')
        plt.plot(mean_ref, label='Mean ref')
        plt.legend()
        plt.show()

    # check that ref is reasonable
    npt.assert_allclose(ref[2:], mean_ref[2:], atol=2.6)

    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


@pytest.mark.parametrize(
    'deficitModel,ref',
    # test that the result is equal to last run (no evidence that  these number are correct)
    [(ZongGaussianDeficit(ct2a=ct2a_mom1d, eps_coeff=0.35),
      [6.34, 7.05, 7.9, 8.23, 7.48, 6.2, 5.2, 5.26, 6.37, 7.31, 7.7, 7.54, 7.34, 7.18, 7.32, 7.7, 8.16])
     ])
def test_deficitModel_wake_map_convection_all2all(deficitModel, ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = All2AllIterative(site, windTurbines, wake_deficitModel=deficitModel, superpositionModel=WeightedSum(),
                                blockage_deficitModel=VortexDipole(), turbulenceModel=STF2017TurbulenceModel())

    x_j = np.linspace(-1500, 1500, 200)
    y_j = np.linspace(-1500, 1500, 100)

    flow_map = wf_model(x, y, wd=0, ws=9).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map.X, flow_map.Y
    Z = flow_map.WS_eff_xylk[:, :, 0, 0]

    mean_ref = [3.2, 4.9, 8., 8.2, 7.9, 7.4, 7., 7., 7.4, 7.9, 8.1, 8.1, 8., 7.8, 7.9, 8.1, 8.4]

    if 0:
        flow_map.plot_wake_map()
        plt.plot(X[49, 100:133:2], Y[49, 100:133:2], '.-')
        windTurbines.plot(x, y)
        plt.figure()
        plt.plot(Z[49, 100:133:2], label='Actual')
        print(np.round(Z[49, 100:133:2], 2).values.tolist())
        plt.plot(ref, label='Reference')
        plt.plot(mean_ref, label='Mean ref')
        plt.legend()
        plt.show()

    # check that ref is reasonable
    npt.assert_allclose(ref[2:], mean_ref[2:], atol=2.6)

    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


@pytest.mark.parametrize(
    'windFarmModel,aep_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJ,
      (368764.7199209298, [9863.98445, 8458.99574, 10863.93795, 14022.65446, 22279.85026,
                           25318.68166, 37461.855, 42999.89503, 24857.24081, 13602.39867,
                           14348.77375, 31867.18218, 75509.51435, 17661.32988, 11773.35282,
                           7875.07291])),
     (NOJLocal,
      (336587.0633777636, [8393.09366, 7644.39035, 10690.37193, 13095.25925, 19271.97188,
                           23644.21809, 36863.35147, 38858.98426, 21150.59603, 12106.08548,
                           13868.64937, 31092.31296, 64287.70342, 17231.88429, 11379.40461,
                           7008.78633])),
     (BastankhahGaussian, (355597.1277349052,
                           [9147.1739, 8136.64045, 11296.20887, 13952.43453, 19784.35422,
                            25191.89568, 38952.44439, 41361.25562, 23050.87822, 12946.16957,
                            14879.10238, 32312.09672, 66974.91909, 17907.90902, 12208.49426,
                            7495.1508])),
     (IEA37SimpleBastankhahGaussian, read_iea37_windfarm(iea37_path + 'iea37-ex16.yaml')[2]),
     (lambda *args, **kwargs: Fuga(tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc',
                                   *args, **kwargs),
      (404422.9302289747, [9911.82745, 9761.75297, 12509.30601, 15396.26107, 23016.74091,
                           27798.80471, 43135.53798, 49622.24427, 24977.80518, 15459.8399,
                           16721.53971, 35692.36221, 77966.92736, 19781.30917, 13720.23771,
                           8950.43363])),
     (GCL, (370863.6246093183,
            [9385.75387, 8768.52105, 11450.13309, 14262.42186, 21178.74926,
             25751.59502, 39483.21753, 44573.31533, 23652.09976, 13924.58752,
             15106.11692, 32840.02909, 71830.22035, 18200.49805, 12394.7626,
             8061.6033])),
     (GCLLocal, (381187.36105425097,
                 [9678.85358, 9003.65526, 11775.06899, 14632.42259, 21915.85495,
                  26419.65189, 40603.68618, 45768.58091, 24390.71103, 14567.43106,
                  15197.82861, 32985.67922, 75062.92788, 18281.21981, 12470.01322,
                  8433.77587])),
     (ZongGaussian, (354394.892004361,
                     [8980.72989, 8091.71749, 11310.53734, 13932.45776, 19973.7833,
                      25155.82651, 39001.85288, 41132.89722, 22631.43932, 12939.46983,
                      14755.07905, 32302.53737, 66685.94946, 17902.61107, 12106.73153,
                      7491.272])),
     (NiayifarGaussian, (362094.6051760222,
                         [9216.49947, 8317.79564, 11380.71714, 14085.18651, 20633.99691,
                          25431.58675, 39243.85219, 42282.12782, 23225.57866, 13375.79217,
                          14794.64817, 32543.64467, 69643.8641, 18036.2368, 12139.1985,
                          7743.87967])),
     ])
def test_IEA37_ex16_windFarmModel(windFarmModel, aep_ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        wf_model = windFarmModel(site, windTurbines, turbulenceModel=GCLTurbulence())
    wf_model.superpositionModel = SquaredSum()

    aep_ilk = wf_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000

    # check if ref is reasonable
    aep_est = 16 * 3.35 * 24 * 365 * .8  # n_wt * P_rated * hours_pr_year - 20% wake loss = 375628.8
    npt.assert_allclose(aep_ref[0], aep_est, rtol=.11)
    npt.assert_allclose(aep_ref[1], [9500, 8700, 11500, 14300, 21300, 25900, 39600, 44300, 23900,
                                     13900, 15200, 33000, 72100, 18300, 12500, 8000], rtol=.15)

    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)
    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


def test_own_deficit_is_zero():
    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()
    for deficitModel in get_models(WakeDeficitModel):
        wf_model = All2AllIterative(site, windTurbines, wake_deficitModel=deficitModel(),
                                    turbulenceModel=STF2017TurbulenceModel())
        sim_res = wf_model([0], [0], yaw=0)
        npt.assert_array_equal(sim_res.WS_eff, sim_res.WS.broadcast_like(sim_res.WS_eff))


@pytest.mark.parametrize('upstream_only,ref', [(False, [[9, 9, 9],  # -1 upstream
                                                        [7, 7, 7]]),  # - (1+2) downstream
                                               (True, [[9, 9, 9],  # -1 upstream
                                                       [8, 8, 8]])])  # -2 downstream
def test_wake_blockage_split(upstream_only, ref):
    class MyWakeModel(WakeDeficitModel):
        def calc_deficit(self, dw_ijlk, **kwargs):
            return np.ones_like(dw_ijlk) * 2

    class MyBlockageModel(BlockageDeficitModel):

        def __init__(self, upstream_only):
            BlockageDeficitModel.__init__(self, upstream_only=upstream_only)

        def calc_deficit(self, dw_ijlk, **kwargs):
            return np.ones_like(dw_ijlk)

    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()
    wf_model = All2AllIterative(site, windTurbines, wake_deficitModel=MyWakeModel(),
                                blockage_deficitModel=MyBlockageModel(upstream_only=upstream_only))
    sim_res = wf_model([0], [0], ws=10, wd=270)
    fm = sim_res.flow_map(XYGrid(x=[-100, 100], y=[-100, 0, 100]))
    if 0:
        sim_res.flow_map().plot_wake_map()
        print(fm.WS_eff.values.squeeze().T)
        plt.show()

    npt.assert_array_equal(fm.WS_eff.values.squeeze().T, ref)


@pytest.mark.parametrize('deficitModel', get_models(WakeDeficitModel))
def test_All2AllIterative_WakeDeficit_RotorAvg(deficitModel):
    if deficitModel == NOJLocalDeficit:
        site = IEA37Site(16)
        windTurbines = IEA37_WindTurbines()
        wf_model = All2AllIterative(site, windTurbines,
                                    wake_deficitModel=deficitModel(rotorAvgModel=CGIRotorAvg(4)),
                                    turbulenceModel=STF2017TurbulenceModel())
        sim_res = wf_model([0, 500, 1000, 1500], [0, 0, 0, 0], wd=270, ws=10)

        if 0:
            sim_res.flow_map(XYGrid(x=np.linspace(-200, 2000, 100))).plot_wake_map()
            plt.show()


@pytest.mark.parametrize('deficitModel', get_models(WakeDeficitModel))
def test_flow_map_symmetry(deficitModel):

    windTurbines = IEA37_WindTurbines()
    wf_model = All2AllIterative(UniformSite(), windTurbines,
                                wake_deficitModel=deficitModel(),
                                turbulenceModel=STF2017TurbulenceModel())
    sim_res = wf_model([0], [0], wd=270, ws=10, yaw=0)
    ws = sim_res.flow_map(YZGrid(x=0, y=np.linspace(-150, 150, 20), z=windTurbines.hub_height())).WS_eff.squeeze()
    ws_center = sim_res.flow_map(XYGrid(x=np.linspace(-150, 150, 20), y=0)).WS_eff.squeeze()

    if 0:
        ax1, ax2 = plt.subplots(2)[1]
        ws.plot(ax=ax1, label=deficitModel.__name__)
        ax1.legend()
        ws_center.plot(ax=ax2)
        plt.show()
    npt.assert_array_almost_equal(ws[:10], ws[10:][::-1])
    if deficitModel is not NoWakeDeficit:
        assert min(ws) < 9
        npt.assert_array_less(ws_center[10:], 9)
    npt.assert_array_almost_equal(ws_center[:10], 10)
