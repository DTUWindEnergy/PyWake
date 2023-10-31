import matplotlib.pyplot as plt
import pytest

from py_wake import np
from py_wake.deficit_models.deficit_model import WakeRadiusTopHat, BlockageDeficitModel
from py_wake.deficit_models.gaussian import ZongGaussian, BastankhahGaussianDeficit
from py_wake.deficit_models.noj import NOJ, NOJDeficit
from py_wake.examples.data.dtu10mw._dtu10mw import ct_curve
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.rotor_avg_models.rotor_avg_model import EqGridRotorAvg
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.utils.model_utils import get_models
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.wind_turbines._wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.superposition_models import WeightedSum, SquaredSum
import warnings
from py_wake.ground_models.ground_models import Mirror


def test_overlapping_area_factor_shapes():
    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()
    wfm = ZongGaussian(site, windTurbines, rotorAvgModel=EqGridRotorAvg(9),
                       turbulenceModel=CrespoHernandez())
    wfm([0, 1000], [0, 0])


def test_area_overlapping():
    wts = WindTurbines(names=['V80'] * 2, diameters=[80] * 2,
                       hub_heights=[70] * 2,
                       # only define for ct
                       powerCtFunctions=[PowerCtFunction(['ws'], lambda ws, run_only: np.interp(ws, ct_curve[:, 0], ct_curve[:, 1]), 'w'),
                                         PowerCtFunction(['ws'], lambda ws, run_only: ws * 0, 'w')])

    site = UniformSite([1], 0.1)
    wfm = NOJ(site, wts)

    y_lst = np.linspace(-250, 250, 50)
    y = np.r_[0, y_lst]
    x = np.r_[0, y_lst * 0 + 400]
    t = np.r_[0, y_lst * 0 + 1]
    WS_eff = wfm(x, y, type=t, ws=10, wd=270).WS_eff.values[1:, 0, 0]
    plt.plot(y_lst, WS_eff, '.-')
    # print(list(np.round(WS_eff[12:23], 2)))
    ref = [10.0, 9.99, 9.88, 9.71, 9.51, 9.29, 9.06, 8.83, 8.64, 8.58, 8.58]
    npt.assert_array_almost_equal(ref, WS_eff[12:23], decimal=2)
    wfm = PropagateDownwind(site, wts, NOJDeficit(rotorAvgModel=None))
    WS_eff = wfm(x, y, type=t, ws=10, wd=270).WS_eff.values[1:, 0, 0]
    # print(list(np.round(WS_eff[12:23], 2)))
    plt.plot(y_lst, WS_eff)
    ref = [10.0, 10.0, 10.0, 10.0, 10.0, 8.58, 8.58, 8.58, 8.58, 8.58, 8.58]
    plt.plot(y_lst[12:23], ref)
    if 0:
        plt.show()
    npt.assert_array_almost_equal(ref, WS_eff[12:23], decimal=2)
    plt.close('all')


@pytest.mark.parametrize('turbulenceModel', get_models(TurbulenceModel))
def test_AreaOverlapAvgModel_turbulence(turbulenceModel):
    if turbulenceModel is None:
        return
    wfm = PropagateDownwind(UniformSite(), V80(), NOJDeficit(),
                            turbulenceModel=turbulenceModel(rotorAvgModel=AreaOverlapAvgModel()))
    if isinstance(wfm.turbulenceModel, WakeRadiusTopHat):
        wfm([0, 1000], [0, 0])
    else:
        with pytest.raises(AssertionError, match='AreaOverlapAvgModel uses the wake_radius'):
            wfm([0, 1000], [0, 0])


@pytest.mark.parametrize('turbulenceModel', get_models(TurbulenceModel))
def test_GaussianOverlapAvgModel_turbulence(turbulenceModel):
    if turbulenceModel is None:
        return
    wfm = PropagateDownwind(UniformSite(), V80(), BastankhahGaussianDeficit(),
                            turbulenceModel=turbulenceModel(rotorAvgModel=GaussianOverlapAvgModel()))
    with pytest.raises(AttributeError, match=f"'{turbulenceModel.__name__}' has no attribute 'sigma_ijlk'"):
        wfm([0, 1000], [0, 0])


@pytest.mark.parametrize('blockageDeficitModel', get_models(BlockageDeficitModel))
def test_GaussianOverlapAvgModel_blockage(blockageDeficitModel):
    if blockageDeficitModel is None:
        return
    wfm = All2AllIterative(UniformSite(), V80(), BastankhahGaussianDeficit(),
                           blockage_deficitModel=blockageDeficitModel(rotorAvgModel=GaussianOverlapAvgModel()))
    with pytest.raises(AttributeError, match=f"'{blockageDeficitModel.__name__}' has no attribute 'sigma_ijlk'"):
        wfm([0, 1000], [0, 0], yaw=0)


def test_GaussianOverlapAvgModel_WeightedSum():
    with pytest.raises(AssertionError, match=r"WeightedSum and CumulativeWakeSum only works with NodeRotorAvgModel-based rotor average models"):
        wfm = PropagateDownwind(UniformSite(), V80(), BastankhahGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()),
                                WeightedSum())
    wfm = PropagateDownwind(UniformSite(), V80(), BastankhahGaussianDeficit(groundModel=Mirror()),
                            WeightedSum())
    with pytest.raises(NotImplementedError, match=r"calc_deficit_convection \(WeightedSum\) cannot be used in combination with GroundModels"):
        wfm([0, 1000], [0, 0])


def test_area_overlapping_deprecated_way():
    wts = WindTurbines(names=['V80'] * 2, diameters=[80] * 2,
                       hub_heights=[70] * 2,
                       # only define for ct
                       powerCtFunctions=[PowerCtFunction(['ws'], lambda ws, run_only: np.interp(ws, ct_curve[:, 0], ct_curve[:, 1]), 'w'),
                                         PowerCtFunction(['ws'], lambda ws, run_only: ws * 0, 'w')])

    site = UniformSite([1], 0.1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        wfm = PropagateDownwind(site, wts, rotorAvgModel=AreaOverlapAvgModel(),
                                wake_deficitModel=NOJDeficit(rotorAvgModel=None),
                                superpositionModel=SquaredSum())

    y_lst = np.linspace(-250, 250, 50)
    y = np.r_[0, y_lst]
    x = np.r_[0, y_lst * 0 + 400]
    t = np.r_[0, y_lst * 0 + 1]
    WS_eff = wfm(x, y, type=t, ws=10, wd=270).WS_eff.values[1:, 0, 0]
    plt.plot(y_lst, WS_eff, '.-')
    # print(list(np.round(WS_eff[12:23], 2)))
    ref = [10.0, 9.99, 9.88, 9.71, 9.51, 9.29, 9.06, 8.83, 8.64, 8.58, 8.58]
    npt.assert_array_almost_equal(ref, WS_eff[12:23], decimal=2)
    wfm = PropagateDownwind(site, wts, NOJDeficit(rotorAvgModel=None))
    WS_eff = wfm(x, y, type=t, ws=10, wd=270).WS_eff.values[1:, 0, 0]
    # print(list(np.round(WS_eff[12:23], 2)))
    plt.plot(y_lst, WS_eff)
    ref = [10.0, 10.0, 10.0, 10.0, 10.0, 8.58, 8.58, 8.58, 8.58, 8.58, 8.58]
    plt.plot(y_lst[12:23], ref)
    if 0:
        plt.show()
    npt.assert_array_almost_equal(ref, WS_eff[12:23], decimal=2)
    plt.close('all')
