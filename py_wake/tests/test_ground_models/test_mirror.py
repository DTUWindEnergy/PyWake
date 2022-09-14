from py_wake.site._site import UniformSite
from py_wake.examples.data.hornsrev1 import V80
from py_wake.ground_models import Mirror
from py_wake.deficit_models.noj import NOJ, NOJDeficit
import matplotlib.pyplot as plt
from py_wake.flow_map import YZGrid
from py_wake import np
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
import pytest
from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
import warnings


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind,
                                     All2AllIterative])
@pytest.mark.parametrize('superpositionModel', [LinearSum(), SquaredSum()])
def test_Mirror_NOJ(wfm_cls, superpositionModel):
    # Compare points in flow map with ws of WT at same position
    site = UniformSite([1], ti=0.1)
    V80_D0 = V80()
    V80_D0._diameters = [0]
    wt = WindTurbines.from_WindTurbine_lst([V80(), V80_D0])
    wfm_ref = wfm_cls(site, wt, wake_deficitModel=NOJDeficit(k=.5), superpositionModel=superpositionModel)
    fm = wfm_ref([0, 0], [0, 0], h=[50, -50], wd=0).flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10))
    ref = fm.WS_eff.squeeze()

    wfm = wfm_cls(site, wt, wake_deficitModel=NOJDeficit(k=.5, groundModel=Mirror()),
                  superpositionModel=superpositionModel,)
    sim_res = wfm([0], [0], h=[50], wd=0)
    fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10)).WS_eff.squeeze()
    with warnings.catch_warnings():
        # D=0 gives divide by zero warning
        warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')
        res = np.array([wfm([0, 0], [0, y], [50, 10], type=[0, 1], wd=0).WS_eff.sel(wt=1).item()
                        for y in [-70, -50, -30, -10]])  # ref.y])

    if 0:
        sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 10, 1))).plot_wake_map()
        plt.plot(ref.y, ref.y * 0 + ref.h, '.')
        plt.plot(ref.y, ref * 10, label='ref, WS*10')
        plt.plot(ref.y, res * 10, label='Res, WS*10')
        plt.plot(fm_res.y, fm_res * 10, label='Res flowmap, WS*10')

        plt.legend()
        plt.show()
    plt.close('all')
    npt.assert_array_equal(res, ref)
    npt.assert_array_equal(fm_res, ref)


def test_Mirror_All2AllIterative():
    # Compare points in flow map with ws of WT at same position
    site = UniformSite([1], ti=0.1)
    V80_D0 = V80()
    V80_D0._diameters = [0]
    wt = WindTurbines.from_WindTurbine_lst([V80(), V80_D0])
    wfm = All2AllIterative(site, wt, NOJDeficit(k=.5, groundModel=Mirror()))
    sim_res = wfm([0], [0], h=[50], wd=0)
    fm_ref = sim_res.flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10))
    ref = fm_ref.WS_eff_xylk[:, 0, 0, 0].values
    with warnings.catch_warnings():
        # D=0 gives divide by zero warning
        warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')
        res = np.array([wfm([0, 0], [0, y], [50, 10], type=[0, 1], wd=0).WS_eff.sel(wt=1).item() for y in fm_ref.X[0]])

    if 0:
        fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 10, 1)))
        fm_res.plot_wake_map()
        plt.plot(fm_ref.X[0], fm_ref.Y[0], '.')
        plt.plot(fm_ref.X[0], ref * 10, label='ref, WS*10')
        plt.plot(fm_ref.X[0], res * 10, label='Res, WS*10')

        plt.legend()
        plt.show()
    plt.close('all')
    npt.assert_array_equal(res, ref)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
def test_Mirror(wfm_cls):
    # Compare points in flow map with ws of WT at same position. All2Alliterative failing with NOJ and WT.diameter=0
    # and therefore this cannot be tested above
    site = UniformSite([1], ti=0.1)
    wt = V80()
    wfm = wfm_cls(site, wt, ZongGaussianDeficit(a=[0, 1], groundModel=Mirror()),
                  turbulenceModel=STF2017TurbulenceModel())
    sim_res = wfm([0], [0], h=[50], wd=0,)
    fm_ref = sim_res.flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10))
    ref = fm_ref.WS_eff_xylk[:, 0, 0, 0].values

    res = np.array([wfm([0, 0], [0, y], [50, 10], wd=0).WS_eff.sel(wt=1).item() for y in fm_ref.X[0]])

    if 0:
        fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 10, 1)))
        fm_res.plot_wake_map()
        plt.plot(fm_ref.X[0], fm_ref.Y[0], '.')
        plt.plot(fm_ref.X[0], ref * 10, label='ref, WS*10')
        plt.plot(fm_ref.X[0], res * 10, label='Res, WS*10')

        plt.legend()
        plt.show()
    plt.close('all')
    npt.assert_array_equal(res, ref)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('groundModel,superpositionModel', [(Mirror(), LinearSum()),
                                                            (Mirror(), SquaredSum())])
def test_Mirror_flow_map(wfm_cls, groundModel, superpositionModel):
    site = UniformSite([1], ti=0.1)
    wt = V80()
    wfm = NOJ(site, wt, k=.5, superpositionModel=superpositionModel)

    fm_ref = wfm([0, 0 + 1e-20], [0, 0 + 1e-20], wd=0, h=[50, -50]
                 ).flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)))
    fm_ref.plot_wake_map()
    plt.title("Underground WT added manually")

    plt.figure()
    wfm = wfm_cls(site, wt, NOJDeficit(k=.5, groundModel=groundModel),
                  superpositionModel=superpositionModel)
    fm_res = wfm([0], [0], wd=0, h=[50]).flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)))
    fm_res.plot_wake_map()
    plt.title("With Mirror GroundModel")

    if 0:
        plt.show()
    plt.close('all')
    npt.assert_array_equal(fm_ref.WS_eff, fm_res.WS_eff)


def test_Mirror_flow_map_multiple_wd():
    site = UniformSite([1], ti=0.1)
    wt = V80()
    wfm = NOJ(site, wt, k=.5, superpositionModel=LinearSum())

    fm_ref = wfm([0, 0 + 1e-20], [0, 0 + 1e-20], wd=[0, 5], h=[50, -50]
                 ).flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)))
    fm_ref.plot_wake_map()
    plt.title("Underground WT added manually")

    plt.figure()
    wfm = All2AllIterative(site, wt, NOJDeficit(k=.5, groundModel=Mirror()),
                           superpositionModel=LinearSum())
    fm_res = wfm([0], [0], wd=[0, 5], h=[50]).flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)))
    fm_res.plot_wake_map()
    plt.title("With Mirror GroundModel")

    if 0:
        plt.show()
    plt.close('all')
    npt.assert_array_equal(fm_ref.WS_eff, fm_res.WS_eff)
