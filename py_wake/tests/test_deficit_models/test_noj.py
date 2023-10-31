import pytest
from py_wake import np
from py_wake import NOJ, NOJLocal
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.flow_map import HorizontalGrid
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.superposition_models import LinearSum, WeightedSum
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.deficit_models.utils import ct2a_mom1d


def NibeA0():
    # Two turbines, 0: Nibe-A, 1:Ct=0
    return WindTurbines(names=['Nibe-A', 'None'], diameters=[40] * 2,
                        hub_heights=[50] * 2,
                        # only define for ct
                        powerCtFunctions=[PowerCtFunction(['ws'], lambda ws, run_only: ws * 0 + 8 / 9, 'w'),
                                          PowerCtFunction(['ws'], lambda ws, run_only: ws * 0, 'w')])


def test_NOJ_Nibe_result():
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).

    site = UniformSite([1], 0.1)
    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]
    wfm = All2AllIterative(site, NibeA0(), wake_deficitModel=NOJDeficit(ct2a=ct2a_mom1d),
                           superpositionModel=LinearSum())
    WS_eff = wfm(x_i, y_i, h_i, type=[0, 1, 1], wd=0.0, ws=8.1).WS_eff
    npt.assert_array_almost_equal(WS_eff.squeeze(), [8.1, 4.35, 5.7])


def test_NOJ_Nibe_result_wake_map():
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).
    def ct_func(_):
        return 8 / 9

    def power_func(*_):
        return 0
    windTurbines = NibeA0()
    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, windTurbines, ct2a=ct2a_mom1d)
    sim_res = wake_model(x=[0], y=[0], wd=[0], ws=[8.1])
    WS_eff_xy = sim_res.flow_map(HorizontalGrid(x=[0], y=[-40, -100], h=50)).WS_eff_xylk.mean(['wd', 'ws'])
    npt.assert_array_almost_equal(WS_eff_xy[:, 0], [4.35, 5.7])


@pytest.mark.parametrize('wdir,x,y', [(0, [0, 0, 0], [0, -40, -100]),
                                      (90, [0, -40, -100], [0, 0, 0]),
                                      (180, [0, 0, 0], [0, 40, 100]),
                                      (270, [0, 40, 100], [0, 0, 0])])
def test_NOJ_two_turbines_in_row(wdir, x, y):
    # Two turbines in a row, North-South
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).

    windTurbines = NibeA0()
    site = UniformSite([1], 0.1)
    wfm = NOJ(site, windTurbines, ct2a=ct2a_mom1d)
    wfm.verbose = False
    h_i = [50, 50, 50]
    WS_eff = wfm(x, y, h_i, wd=wdir, ws=8.1).WS_eff
    ws_wt3 = 8.1 - np.hypot(8.1 * 2 / 3 * (20 / 26)**2, 8.1 * 2 / 3 * (20 / 30)**2)
    npt.assert_array_almost_equal(WS_eff.squeeze(), [8.1, 4.35, ws_wt3])


def test_NOJ_6_turbines_in_row():
    n_wt = 6
    x = [0] * n_wt
    y = - np.arange(n_wt) * 40 * 2

    site = UniformSite([1], 0.1)
    wfm = NOJ(site, NibeA0(), ct2a=ct2a_mom1d)
    wfm.verbose = False
    WS_eff = wfm(x, y, 50, wd=0.0, ws=11.0).WS_eff.squeeze()
    npt.assert_array_almost_equal(
        WS_eff[1:], 11 - np.sqrt(np.cumsum(((11 * 2 / 3 * 20**2)**2) / (20 + 8 * np.arange(1, 6))**4)))


def test_NOJLocal_6_turbines_in_row():
    n_wt = 6
    x = [0] * n_wt
    y = - np.arange(n_wt) * 40 * 2

    site = UniformSite([1], 0.1)
    wfm = NOJLocal(site, NibeA0(), turbulenceModel=STF2017TurbulenceModel(), ct2a=ct2a_mom1d)
    wfm.verbose = False
    WS_eff = wfm(x, y, 50, wd=0.0, ws=11.0).WS_eff
    npt.assert_array_almost_equal(WS_eff.squeeze()[1:], [5.62453869, 5.25806829, 5.64808912, 6.07792364, 6.44549094])


def test_NOJConvection():
    site = UniformSite([1], 0.1)
    with pytest.raises(AssertionError, match='WeightedSum and CumulativeWakeSum only works with NodeRotorAvgModel-based rotor average models'):
        wfm = NOJ(site, NibeA0(), superpositionModel=WeightedSum())
        wfm([0, 500], [0, 0])


def test_NOJConvection2():
    site = UniformSite([1], 0.1)
    with pytest.raises(NotImplementedError, match='calc_deficit_convection not implemented for NOJ'):
        wfm = NOJ(site, NibeA0(), rotorAvgModel=None, superpositionModel=WeightedSum())
        wfm([0, 500], [0, 0])


def test_NOJLocal_ti_eff_dependence():
    site = UniformSite([1], 0.1)
    wfm = NOJLocal(site, NibeA0(), ct2a=ct2a_mom1d)
    sim_res = wfm([0, 40], [0, 0], wd=270, ws=[10, 10], TI=[[[.1, .2]]])
    npt.assert_array_almost_equal(sim_res.TI.squeeze(), [.1, .2])
    npt.assert_array_almost_equal(sim_res.TI_eff.squeeze(), [[.1, .2],
                                                             [0.43738364, 0.47043007]])
    npt.assert_array_almost_equal(sim_res.WS_eff.isel(wt=1), [[4.32651153, 5.045581]])
