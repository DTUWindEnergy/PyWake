import pytest

import numpy as np
from py_wake import NOJ, turbulence_models
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.flow_map import HorizontalGrid
from py_wake.site._site import UniformSite
from py_wake.superposition_models import LinearSum, MaxSum
from py_wake.superposition_models import SquaredSum
from py_wake.tests import npt
from py_wake.tests.test_deficit_models.test_noj import NibeA0
from py_wake.turbulence_models.stf import STF2005TurbulenceModel, STF2017TurbulenceModel, IECWeight
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.turbulence_models.gcl_turb import GCLTurbulence
import matplotlib.pyplot as plt
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.deficit_models.gaussian import BastankhahGaussian, IEA37SimpleBastankhahGaussianDeficit,\
    IEA37SimpleBastankhahGaussian
import xarray as xr
import os
import pkgutil
import inspect
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter, EqGridRotorAvg, GQGridRotorAvg, CGIRotorAvg
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel

WindFarmModel.verbose = False


def get_all_turbulence_models():
    all_turbulence_modules = []
    for loader, module_name, _ in pkgutil.walk_packages([os.path.dirname(turbulence_models.__file__)]):

        _module = loader.find_module(module_name).load_module(module_name)
        for n in dir(_module):
            v = _module.__dict__[n]
            if inspect.isclass(v) and issubclass(v, TurbulenceModel) and \
                    v not in [TurbulenceModel]:
                all_turbulence_modules.append(v)
    return all_turbulence_modules


@pytest.mark.parametrize('turbulence_model,ref_ti', [
    (STF2005TurbulenceModel(), [0.075, 0.075, 0.075, 0.104, 0.167, 0.124, 0.075, 0.075,
                                0.075, 0.075, 0.075, 0.075, 0.104, 0.136, 0.098, 0.104]),
    (STF2017TurbulenceModel(), [0.075, 0.075, 0.075, 0.114, 0.197, 0.142, 0.075, 0.075,
                                0.075, 0.075, 0.075, 0.075, 0.115, 0.158, 0.108, 0.115]),
    (STF2017TurbulenceModel(weight_function=IECWeight()), [0.075, 0.075, 0.075, 0.215, 0.229, 0.179, 0.075, 0.075, 0.075,
                                                           0.075, 0.075, 0.075, 0.075, 0.215, 0.075, 0.075]),
    (GCLTurbulence(), [0.075, 0.075, 0.075, 0.117, 0.151, 0.135, 0.075, 0.075, 0.075,
                       0.075, 0.075, 0.075, 0.128, 0.127, 0.117, 0.128]),
    (CrespoHernandez(), [0.075, 0.075, 0.075, 0.129, 0.17, 0.151, 0.075,
                         0.075, 0.075, 0.075, 0.075, 0.075, 0.143, 0.141, 0.13, 0.143])
])
def test_models_with_noj(turbulence_model, ref_ti):
    # setup site, turbines and wind farm model
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    for wake_model in [NOJ(site, windTurbines, turbulenceModel=turbulence_model),
                       All2AllIterative(site, windTurbines,
                                        wake_deficitModel=NOJDeficit(),
                                        superpositionModel=SquaredSum(),
                                        turbulenceModel=turbulence_model),
                       ]:

        res = wake_model(x, y)
        # print(turbulence_model.__class__.__name__, np.round(res.TI_eff_ilk[:, 0, 0], 3).tolist())
        if 0:
            plt.title("%s, %s" % (wake_model.__class__.__name__, turbulence_model.__class__.__name__))
            res.flow_map(wd=0).plot_ti_map()
            plt.show()

        npt.assert_array_almost_equal(res.TI_eff_ilk[:, 0, 0], ref_ti, 3)


@pytest.mark.parametrize('turbulence_model,ref_ti', [
    (GCLTurbulence(), [0.075, 0.075, 0.075, 0.097, 0.151, 0.135,
                       0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.128, 0.123, 0.116, 0.128]),
    (CrespoHernandez(), [0.075, 0.075, 0.075, 0.104, 0.17, 0.151, 0.075,
                         0.075, 0.075, 0.075, 0.075, 0.075, 0.143, 0.137, 0.129, 0.143])
])
def test_models_with_BastankhahGaussian(turbulence_model, ref_ti):
    # setup site, turbines and wind farm model
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    for wake_model in [BastankhahGaussian(site, windTurbines, turbulenceModel=turbulence_model)]:

        res = wake_model(x, y)
        # print(turbulence_model.__class__.__name__, np.round(res.TI_eff_ilk[:, 0, 0], 3).tolist())
        if 0:
            res.flow_map(wd=0).plot_ti_map()
            plt.show()

        npt.assert_array_almost_equal(res.TI_eff_ilk[:, 0, 0], ref_ti, 3)


def test_max_sum():
    maxSum = MaxSum()
    npt.assert_array_equal(maxSum.calc_effective_TI(TI_xxx=[.2, .4], add_turb_jxxx=[[0.1, .4], [.3, .2]]), [.5, .8])


def test_superposition_model_indices():
    class WTSite(UniformSite):
        def local_wind(self, x_i=None, y_i=None, h_i=None, wd=None,
                       ws=None, time=False, wd_bin_size=None, ws_bins=None):
            lw = UniformSite.local_wind(self, x_i=x_i, y_i=y_i, h_i=h_i, wd=wd, ws=ws,
                                        wd_bin_size=wd_bin_size, ws_bins=ws_bins)
            lw['TI'] = xr.DataArray(lw.TI_ilk + np.arange(len(x_i))[:, np.newaxis, np.newaxis] * .1,
                                    [('wt', [0, 1, 2]), ('wd', np.atleast_1d(wd)), ('ws', np.atleast_1d(ws))])
            return lw

    site = WTSite([1], 0.1)

    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]

    # WS_ilk different at each wt position
    TI_ilk = site.local_wind(x_i, y_i, h_i, wd=0, ws=8.1).TI_ilk
    npt.assert_array_almost_equal(TI_ilk, np.reshape([0.1, 0.2, 0.3], (3, 1, 1)), 10)

    def get_wf_model(cls):
        return cls(site, NibeA0, wake_deficitModel=NoWakeDeficit(),
                   superpositionModel=LinearSum(),
                   turbulenceModel=STF2017TurbulenceModel())
    for wake_model in [get_wf_model(PropagateDownwind),
                       get_wf_model(All2AllIterative)]:

        # No wake (ct = 0), i.e. WS_eff == WS
        TI_eff_ilk = wake_model.calc_wt_interaction(x_i, y_i, h_i, [1, 1, 1], 0.0, 8.1)[1]
        npt.assert_array_equal(TI_eff_ilk, TI_ilk)

        # full wake (CT=8/9)
        ref_TI_eff_ilk = TI_ilk + np.reshape([0, 0.33738364, np.sum([0.19369135, 0.21239116])], (3, 1, 1))

        TI_eff_ilk = wake_model.calc_wt_interaction(x_i, y_i, h_i, [0, 0, 0], 0.0, 8.1)[1]
        npt.assert_array_almost_equal(TI_eff_ilk, ref_TI_eff_ilk)

        sim_res = wake_model(x_i, y_i, h_i, [0, 0, 0], 0.0, 8.1)
        TI_eff_ilk = sim_res.flow_map(HorizontalGrid(x=[0], y=y_i, h=50)).TI_eff_xylk[:, 0]

        npt.assert_array_almost_equal(TI_eff_ilk, ref_TI_eff_ilk)


@pytest.mark.parametrize('turbulenceModel', get_all_turbulence_models())
def test_own_turbulence_is_zero(turbulenceModel):
    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()
    wf_model = All2AllIterative(site, windTurbines, wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                                turbulenceModel=turbulenceModel())
    sim_res = wf_model([0], [0])
    npt.assert_array_equal(sim_res.TI_eff, sim_res.TI.broadcast_like(sim_res.TI_eff))


def test_RotorAvg_deficit():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site,
                                        windTurbines,
                                        turbulenceModel=STF2017TurbulenceModel())
    flow_map = wfm([0, 500], [0, 0], wd=270, ws=10).flow_map(HorizontalGrid(x=[500], y=np.arange(-100, 100)))
    plt.plot(flow_map.Y[:, 0], flow_map.TI_eff_xylk[:, 0, 0, 0])
    R = windTurbines.diameter() / 2

    for name, rotorAvgModel, ref1 in [
            ('None', None, 0.22292190804089568),
            ('RotorCenter', RotorCenter(), 0.22292190804089568),
            ('RotorGrid100', EqGridRotorAvg(100), 0.1985255601976247),
            ('RotorGQGrid_4,3', GQGridRotorAvg(4, 3), 0.1982984399750206),
            ('RotorCGI4', CGIRotorAvg(4), 0.19774602325558865),
            ('RotorCGI4', CGIRotorAvg(21), 0.19849398318014355)]:

        # test with PropagateDownwind
        wfm = IEA37SimpleBastankhahGaussian(site,
                                            windTurbines,
                                            turbulenceModel=STF2017TurbulenceModel(rotorAvgModel=rotorAvgModel))
        sim_res = wfm([0, 500], [0, 0], wd=270, ws=10)
        npt.assert_almost_equal(ref1, sim_res.TI_eff_ilk[1, 0, 0], ref1, err_msg=name)

        # test with All2AllIterative
        wfm = All2AllIterative(site, windTurbines,
                               IEA37SimpleBastankhahGaussianDeficit(),
                               turbulenceModel=STF2017TurbulenceModel(rotorAvgModel=rotorAvgModel),
                               superpositionModel=SquaredSum())
        sim_res = wfm([0, 500], [0, 0], wd=270, ws=10)
        npt.assert_almost_equal(ref1, sim_res.TI_eff_ilk[1, 0, 0])

        plt.plot([-R, R], [sim_res.WS_eff_ilk[1, 0, 0]] * 2, label=name)
    if 0:
        plt.legend()
        plt.show()
    plt.close('all')


@pytest.mark.parametrize('WFM', [All2AllIterative, PropagateDownwind])
@pytest.mark.parametrize('turbulenceModel', get_all_turbulence_models())
def test_with_all_turbulence_models(WFM, turbulenceModel):
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()

    wfm = WFM(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
              rotorAvgModel=CGIRotorAvg(4),
              superpositionModel=LinearSum(),
              turbulenceModel=turbulenceModel())

    wfm2 = WFM(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
               superpositionModel=LinearSum(),
               turbulenceModel=turbulenceModel(rotorAvgModel=CGIRotorAvg(4)))
    kwargs = {'x': [0, 0, 500, 500], 'y': [0, 500, 0, 500], 'wd': [0], 'ws': [8]}
    npt.assert_array_equal(wfm(**kwargs).TI_eff, wfm2(**kwargs).TI_eff, turbulenceModel.__name__)
