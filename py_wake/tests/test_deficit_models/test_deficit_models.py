import numpy as np
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windrose,\
    read_iea37_windfarm
from py_wake.tests import npt
from py_wake import IEA37SimpleBastankhahGaussian, BastankhahGaussian
from py_wake.flow_map import HorizontalGrid
import matplotlib.pyplot as plt
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, BastankhahGaussian,\
    IEA37SimpleBastankhahGaussianDeficit
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.superposition_models import SquaredSum
from py_wake.turbulence_models.gcl import GCLTurbulenceModel
import pytest
from py_wake.deficit_models.gcl import GCLDeficitModel
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deficit_models.fuga import FugaDeficit
from py_wake.tests.test_files import tfp
from py_wake.deficit_models.noj import NOJDeficit


@pytest.mark.parametrize(
    'deficitModel,aep_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJDeficit(), (367205.0846866496, [9833.86287, 8416.99088, 10820.37673, 13976.26422, 22169.66036,
                                         25234.9215, 37311.64388, 42786.37028, 24781.33444, 13539.82115,
                                         14285.22744, 31751.29488, 75140.15677, 17597.10319, 11721.21226,
                                         7838.84383])),
     (BastankhahGaussianDeficit(), (355971.9717035484,
                                    [9143.74048, 8156.71681, 11311.92915, 13955.06316, 19807.65346,
                                        25196.64182, 39006.65223, 41463.31044, 23042.22602, 12978.30551,
                                        14899.26913, 32320.21637, 67039.04091, 17912.40907, 12225.04134,
                                        7513.75582])),
     (IEA37SimpleBastankhahGaussianDeficit(), read_iea37_windfarm(iea37_path + 'iea37-ex16.yaml')[2]),
     (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00014617Zi=00399Zeta0=0.00E+0/'),
      (398833.4594353128, [9630.32321, 9731.27952, 12459.23261, 15329.04663, 22194.04385,
                           27677.4453, 42962.87107, 49467.33757, 24268.41448, 15412.81128,
                           16675.27526, 35497.95388, 75248.37632, 19673.5648, 13682.27714,
                           8923.20653])),
     (GCLDeficitModel(), (370863.6246093183,
                          [9385.75387, 8768.52105, 11450.13309, 14262.42186, 21178.74926,
                           25751.59502, 39483.21753, 44573.31533, 23652.09976, 13924.58752,
                           15106.11692, 32840.02909, 71830.22035, 18200.49805, 12394.7626,
                           8061.6033]))])
def test_IEA37_ex16(deficitModel, aep_ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel,
                                 superpositionModel=SquaredSum(), turbulenceModel=GCLTurbulenceModel())

    aep_ilk = wf_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
    aep_MW_l = aep_ilk.sum((0, 2)) * 1000

    # check if ref is reasonable
    aep_est = 16 * 3.35 * 24 * 365 * .8  # n_wt * P_rated * hours_pr_year - 20% wake loss = 375628.8
    npt.assert_allclose(aep_ref[0], aep_est, rtol=.1)
    npt.assert_allclose(aep_ref[1], [9500, 8700, 11500, 14300, 21300, 25900, 39600, 44300, 23900,
                                     13900, 15200, 33000, 72100, 18300, 12500, 8000], rtol=.15)

    npt.assert_almost_equal(aep_MW_l.sum(), aep_ref[0], 5)
    npt.assert_array_almost_equal(aep_MW_l, aep_ref[1], 5)


@pytest.mark.parametrize(
    'deficitModel,ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJDeficit(),
      [3.27, 3.27, 9.0, 7.46, 7.46, 7.46, 7.46, 7.31, 7.31, 7.31, 7.31, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3]),
     (BastankhahGaussianDeficit(),
      [0.18, 3.6, 7.27, 8.32, 7.61, 6.64, 5.96, 6.04, 6.8, 7.69, 8.08, 7.87, 7.59, 7.46, 7.55, 7.84, 8.19]),
     (IEA37SimpleBastankhahGaussianDeficit(),
      [3.32, 4.86, 7.0, 8.1, 7.8, 7.23, 6.86, 6.9, 7.3, 7.82, 8.11, 8.04, 7.87, 7.79, 7.85, 8.04, 8.28]),
     (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00014617Zi=00399Zeta0=0.00E+0/'),
      [6.91, 7.87, 8.77, 8.88, 8.55, 7.88, 7.24, 7.32, 8.01, 8.62, 8.72, 8.42, 8.05, 7.85, 8., 8.37, 8.69]),
     (GCLDeficitModel(),
      [2.39, 5.01, 7.74, 8.34, 7.95, 7.58, 7.29, 7.32, 7.61, 7.92, 8.11, 8.09, 7.95, 7.83, 7.92, 8.1, 8.3])])
def test_deficitModel_wake_map(deficitModel, ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel, superpositionModel=SquaredSum(),
                                 turbulenceModel=GCLTurbulenceModel())

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

    # check that ref is reasonable
    npt.assert_allclose(ref[2:], mean_ref[2:], atol=2.6)

    npt.assert_array_almost_equal(Z[49, 100:133:2], ref, 2)


v = []


@pytest.mark.parametrize(
    'deficitModel,wake_radius_ref',
    # test that the result is equal to last run (no evidens that  these number are correct)
    [(NOJDeficit(), [100., 75., 150., 100., 100.]),
     (BastankhahGaussianDeficit(),
      [83.336286, 57.895893, 115.791786, 75.266662, 83.336286]),
     (IEA37SimpleBastankhahGaussianDeficit(),
      [83.336286, 57.895893, 115.791786, 75.266662, 83.336286]),
     (FugaDeficit(LUT_path=tfp + 'fuga/2MW/Z0=0.00014617Zi=00399Zeta0=0.00E+0/'),
      [100, 50, 100, 100, 100]),
     (GCLDeficitModel(),
      [156.949964, 97.763333, 195.526667, 113.225695, 111.340236])])
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
    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=deficitModel, turbulenceModel=GCLTurbulenceModel())
    wfm(x=[0, 500], y=[0, 0], wd=[30], ws=[10])


def test_wake_radius_not_implemented():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                            turbulenceModel=GCLTurbulenceModel())
    with pytest.raises(NotImplementedError, match="wake_radius not implemented for NoWakeDeficit"):
        wfm(x, y)
