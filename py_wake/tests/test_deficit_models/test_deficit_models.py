import pytest

import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models.deficit_model import DeficitModel, WakeDeficitModel, BlockageDeficitModel
from py_wake.deficit_models.fuga import FugaDeficit, Fuga
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, IEA37SimpleBastankhahGaussianDeficit,\
    ZongGaussianDeficit, NiayifarGaussianDeficit, BastankhahGaussian, IEA37SimpleBastankhahGaussian, ZongGaussian,\
    NiayifarGaussian
from py_wake.deficit_models.gcl import GCLDeficit, GCL, GCLLocal
from py_wake.deficit_models.noj import NOJDeficit, NOJ, NOJLocalDeficit, NOJLocal, TurboNOJDeficit
from py_wake.deficit_models import VortexDipole
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm
from py_wake.flow_map import HorizontalGrid, XYGrid
from py_wake.superposition_models import SquaredSum, WeightedSum
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.turbulence_models.gcl_turb import GCLTurbulence
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.utils.model_utils import get_models
from numpy import newaxis as na


class GCLLocalDeficit(GCLDeficit):
    def __init__(self):
        GCLDeficit.__init__(self, use_effective_ws=True, use_effective_ti=True)


@pytest.mark.parametrize(
    'deficitModel,aep_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJDeficit(), (367205.0846866496, [9833.86287, 8416.99088, 10820.37673, 13976.26422, 22169.66036,
                                         25234.9215, 37311.64388, 42786.37028, 24781.33444, 13539.82115,
                                         14285.22744, 31751.29488, 75140.15677, 17597.10319, 11721.21226,
                                         7838.84383])),
     (NOJLocalDeficit(), (335151.6404628441, [8355.71335, 7605.92379, 10654.172, 13047.6971, 19181.46408,
                                              23558.34198, 36738.52415, 38663.44595, 21056.39764, 12042.79324,
                                              13813.46269, 30999.42279, 63947.61202, 17180.40299, 11334.12323,
                                              6972.14345])),
     (TurboNOJDeficit(), (354154.2962989713, [9320.85263, 8138.29496, 10753.75662, 13398.00865, 21189.29438,
                                              24190.84895, 37081.91938, 41369.66605, 23488.54863, 12938.48451,
                                              14065.00719, 30469.75602, 71831.78532, 16886.85274, 11540.51872,
                                              7490.70156])),

     (BastankhahGaussianDeficit(), (355971.9717035484,
                                    [9143.74048, 8156.71681, 11311.92915, 13955.06316, 19807.65346,
                                        25196.64182, 39006.65223, 41463.31044, 23042.22602, 12978.30551,
                                        14899.26913, 32320.21637, 67039.04091, 17912.40907, 12225.04134,
                                        7513.75582])),
     (IEA37SimpleBastankhahGaussianDeficit(), read_iea37_windfarm(iea37_path + 'iea37-ex16.yaml')[2]),
     (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/'),
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
     (ZongGaussianDeficit(),
      (354263.1062694012, [8960.58672, 8095.48341, 11328.51572, 13924.50408, 19938.78428,
                           25141.4657, 39063.84731, 41152.04065, 22580.67854, 12944.55424,
                           14778.95385, 32294.41749, 66540.62665, 17898.1109, 12126.32111,
                           7494.21561])),
     (NiayifarGaussianDeficit(),
      (362228.4003016239, [9210.01349, 8330.06642, 11392.04851, 14082.13366, 20643.45247,
                           25426.07466, 39282.9259, 42344.50431, 23209.23399, 13390.08421,
                           14807.66824, 32535.89096, 69640.3283, 18031.93957, 12149.88163,
                           7752.15401])),

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
    npt.assert_allclose(aep_ref[0], aep_est, rtol=.11)
    npt.assert_allclose(aep_ref[1], [9500, 8700, 11500, 14300, 21300, 25900, 39600, 44300, 23900,
                                     13900, 15200, 33000, 72100, 18300, 12500, 8000], rtol=.15)

    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)
    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


@pytest.mark.parametrize(
    'deficitModel,ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJDeficit(),
      [3.27, 3.27, 9.0, 7.46, 7.46, 7.46, 7.46, 7.31, 7.31, 7.31, 7.31, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3]),
     (NOJLocalDeficit(),
      [3.09, 3.09, 9., 9., 5.54, 5.54, 5.54, 5.54, 5.54, 5.54, 6.73, 6.73, 6.73, 6.73, 6.73, 6.73, 6.73]),
     (TurboNOJDeficit(),
      [3.09, 3.09, 9., 9., 5.54, 5.54, 5.54, 5.54, 5.54, 5.54, 6.73, 6.73, 6.73, 6.73, 6.73, 6.73, 6.73]),
     (BastankhahGaussianDeficit(),
      [0.18, 3.6, 7.27, 8.32, 7.61, 6.64, 5.96, 6.04, 6.8, 7.69, 8.08, 7.87, 7.59, 7.46, 7.55, 7.84, 8.19]),
     (IEA37SimpleBastankhahGaussianDeficit(),
      [3.32, 4.86, 7.0, 8.1, 7.8, 7.23, 6.86, 6.9, 7.3, 7.82, 8.11, 8.04, 7.87, 7.79, 7.85, 8.04, 8.28]),
     (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/'),
      [6.91, 7.87, 8.77, 8.88, 8.55, 7.88, 7.24, 7.32, 8.01, 8.62, 8.72, 8.42, 8.05, 7.85, 8., 8.37, 8.69]),
     (GCLDeficit(),
      [2.39, 5.01, 7.74, 8.34, 7.95, 7.58, 7.29, 7.32, 7.61, 7.92, 8.11, 8.09, 7.95, 7.83, 7.92, 8.1, 8.3]),
     (GCLLocalDeficit(),
      [3.05, 5.24, 7.61, 8.36, 8.08, 7.81, 7.61, 7.63, 7.82, 8.01, 8.11, 8.07, 7.94, 7.83, 7.92, 8.1, 8.3]),
     (ZongGaussianDeficit(),
      [6.34, 7.08, 8.09, 8.36, 7.5, 6.2, 5.23, 5.34, 6.43, 7.64, 8.08, 7.75, 7.36, 7.19, 7.32, 7.69, 8.14]),
     (NiayifarGaussianDeficit(),
      [0.18, 3.6, 7.27, 8.32, 7.61, 6.64, 5.97, 6.04, 6.8, 7.69, 8.08, 7.87, 7.59, 7.46, 7.56, 7.84, 8.19]),
     ][1:2])
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
    npt.assert_allclose(ref[2:], mean_ref[2:], atol=2.6)


@pytest.mark.parametrize(
    'deficitModel,wake_radius_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJDeficit(), [100., 75., 150., 100., 100.]),
     (NOJLocalDeficit(), [71., 46., 92., 71., 61.5]),
     (TurboNOJDeficit(), [99.024477, 61.553917,
                          123.107833, 92.439673, 97.034049]),
     (BastankhahGaussianDeficit(),
      [83.336286, 57.895893, 115.791786, 75.266662, 83.336286]),
     (IEA37SimpleBastankhahGaussianDeficit(),
      [83.336286, 57.895893, 115.791786, 75.266662, 83.336286]),
     (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/'),
      [100, 50, 100, 100, 100]),
     (GCLDeficit(),
      [156.949964, 97.763333, 195.526667, 113.225695, 111.340236]),
     (GCLLocalDeficit(),
      [156.949964, 97.763333, 195.526667, 113.225695, 111.340236]),
     (ZongGaussianDeficit(), [91.15734, 66.228381, 132.456762, 94.90156, 79.198215]),
     (NiayifarGaussianDeficit(), [92.880786, 67.440393, 134.880786, 84.811162, 73.880786]),
     ])
def test_wake_radius(deficitModel, wake_radius_ref):

    mean_ref = [105, 68, 135, 93, 123]
    # check that ref is reasonable
    npt.assert_allclose(wake_radius_ref, mean_ref, rtol=.5)

    npt.assert_array_almost_equal(deficitModel.wake_radius(
        D_src_il=np.reshape([100, 50, 100, 100, 100], (5, 1)),
        dw_ijlk=np.reshape([500, 500, 1000, 500, 500], (5, 1, 1, 1)),
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
        sim_res = wfm([0], [0], wd=[270], ws=10)
        sim_res.flow_map(HorizontalGrid(x=np.arange(-100, 1500, 10))).WS_eff.plot()
        x = np.arange(0, 1500, 10)
        wr = deficitModel.wake_radius(
            D_src_il=np.reshape([130], (1, 1)),
            dw_ijlk=np.reshape(x, (1, len(x), 1, 1)),
            ct_ilk=sim_res.CT.values,
            TI_ilk=np.reshape(sim_res.TI.values, (1, 1, 1)),
            TI_eff_ilk=sim_res.TI_eff.values)[0, :, 0, 0]
        plt.title(deficitModel.__class__.__name__)
        plt.plot(x, wr)
        plt.plot(x, -wr)
        plt.axis('equal')
        plt.show()


def test_wake_radius_not_implemented():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    class MyDeficitModel(WakeDeficitModel):
        args4deficit = ['WS_ilk', 'dw_ijlk', 'cw_ijlk']

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
    [(BastankhahGaussianDeficit(), (345846.3355259293,
                                    [8835.30563, 7877.90062, 11079.66832, 13565.65235, 18902.99769,
                                     24493.53897, 38205.75284, 40045.9948, 22264.97018, 12662.90784,
                                     14650.96535, 31289.90349, 65276.92307, 17341.39229, 12021.3049,
                                     7331.15717])),
     (ZongGaussianDeficit(),
      (342944.4057168523, [8674.44232, 7806.22033, 11114.78804, 13549.48197, 18895.50866,
                           24464.34244, 38326.85532, 39681.61999, 21859.59465, 12590.25899,
                           14530.24656, 31158.81189, 63812.14454, 17268.73912, 11922.25359,
                           7289.09731])),
     (NiayifarGaussianDeficit(),
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
    [(BastankhahGaussianDeficit(),
      [0.17, 3.55, 7.61, 8.12, 7.58, 6.63, 5.93, 6.04, 6.65, 7.35, 7.69, 7.65, 7.54, 7.45, 7.55, 7.84, 8.19]),
     (ZongGaussianDeficit(),
      [6.34, 7.05, 7.9, 8.15, 7.45, 6.19, 5.21, 5.26, 6.38, 7.32, 7.7, 7.54, 7.34, 7.18, 7.32, 7.69, 8.14]),
     (NiayifarGaussianDeficit(),
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
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(ZongGaussianDeficit(),
      [6.34, 7.05, 8.18, 8.25, 7.49, 6.2, 5.2, 5.26, 6.37, 7.33, 7.7, 7.54, 7.34, 7.18, 7.32, 7.7, 8.16])
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
      (367205.0846866496, [9833.86287, 8416.99088, 10820.37673, 13976.26422, 22169.66036,
                           25234.9215, 37311.64388, 42786.37028, 24781.33444, 13539.82115,
                           14285.22744, 31751.29488, 75140.15677, 17597.10319, 11721.21226,
                           7838.84383])),
     (NOJLocal,
      (335151.6404628441, [8355.71335, 7605.92379, 10654.172, 13047.6971, 19181.46408,
                           23558.34198, 36738.52415, 38663.44595, 21056.39764, 12042.79324,
                           13813.46269, 30999.42279, 63947.61202, 17180.40299, 11334.12323,
                           6972.14345])),
     (BastankhahGaussian, (355971.9717035484,
                           [9143.74048, 8156.71681, 11311.92915, 13955.06316, 19807.65346,
                            25196.64182, 39006.65223, 41463.31044, 23042.22602, 12978.30551,
                            14899.26913, 32320.21637, 67039.04091, 17912.40907, 12225.04134,
                            7513.75582])),
     (IEA37SimpleBastankhahGaussian, read_iea37_windfarm(iea37_path + 'iea37-ex16.yaml')[2]),
     (lambda *args, **kwargs: Fuga(tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/', *args, **kwargs),
      (404441.6306021485, [9912.33731, 9762.05717, 12510.14066, 15396.76584, 23017.66483,
                           27799.7161, 43138.41606, 49623.79059, 24979.09001, 15460.45923,
                           16723.02619, 35694.35526, 77969.14805, 19782.41376, 13721.45739,
                           8950.79218])),
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
     (ZongGaussian,
      (354263.1062694012, [8960.58672, 8095.48341, 11328.51572, 13924.50408, 19938.78428,
                           25141.4657, 39063.84731, 41152.04065, 22580.67854, 12944.55424,
                           14778.95385, 32294.41749, 66540.62665, 17898.1109, 12126.32111,
                           7494.21561])),
     (NiayifarGaussian,
      (362228.4003016239, [9210.01349, 8330.06642, 11392.04851, 14082.13366, 20643.45247,
                           25426.07466, 39282.9259, 42344.50431, 23209.23399, 13390.08421,
                           14807.66824, 32535.89096, 69640.3283, 18031.93957, 12149.88163,
                           7752.15401])),
     ])
def test_IEA37_ex16_windFarmModel(windFarmModel, aep_ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
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
    for deficitModel in get_models(WakeDeficitModel):
        site = Hornsrev1Site()
        windTurbines = IEA37_WindTurbines()
        wf_model = All2AllIterative(site, windTurbines, wake_deficitModel=deficitModel(),
                                    turbulenceModel=STF2017TurbulenceModel())
        sim_res = wf_model([0], [0])
        npt.assert_array_equal(sim_res.WS_eff, sim_res.WS.broadcast_like(sim_res.WS_eff))


@pytest.mark.parametrize('upstream_only,ref', [(False, [[9, 9, 9],  # -1 upstream
                                                        [7, 7, 7]]),  # - (1+2) downstream
                                               (True, [[9, 9, 9],  # -1 upstream
                                                       [8, 8, 8]])])  # -2 downstream
def test_wake_blockage_split(upstream_only, ref):
    class MyWakeModel(WakeDeficitModel):
        args4deficit = []

        def calc_deficit(self, dw_ijlk, **kwargs):
            return np.ones_like(dw_ijlk) * 2

    class MyBlockageModel(BlockageDeficitModel):
        args4deficit = []

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
