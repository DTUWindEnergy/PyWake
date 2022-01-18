import pytest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from py_wake.deficit_models.fuga import FugaDeficit
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.flow_map import XYGrid
from py_wake.rotor_avg_models.rotor_avg_model import CGIRotorAvg
from py_wake.superposition_models import LinearSum
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.site.xrsite import XRSite
from py_wake.deficit_models.rathmann import Rathmann
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.wind_turbines._wind_turbines import WindTurbines
from py_wake.examples.data import wtg_path
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80


class FugaDeficitCount(FugaDeficit):
    counter = 0

    def _calc_layout_terms(self, dw_ijlk, hcw_ijlk, h_il, dh_ijlk, D_src_il, **_):
        I, J = dw_ijlk.shape[:2]
        if I > 1 and I == J:
            # only count All2All
            self.counter += 1
        return FugaDeficit._calc_layout_terms(self, dw_ijlk, hcw_ijlk, h_il, dh_ijlk, D_src_il, **_)


@pytest.mark.parametrize('deflection_model,count',
                         [(None, 1),
                          (JimenezWakeDeflection(), 4)])
def test_All2AllIterativeDeflection(deflection_model, count):

    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    deficit_model = FugaDeficitCount()
    wf_model = All2AllIterative(site, windTurbines,
                                wake_deficitModel=deficit_model,
                                superpositionModel=LinearSum(),
                                blockage_deficitModel=SelfSimilarityDeficit(),
                                rotorAvgModel=CGIRotorAvg(4),
                                deflectionModel=deflection_model, convergence_tolerance=0)
    sim_res = wf_model([0, 500, 1000, 1500], [0, 0, 0, 0],
                       wd=270, ws=10, yaw=[30, -30, 30, -30])
    assert wf_model.wake_deficitModel.counter == count
    if 0:
        sim_res.flow_map(
            XYGrid(x=np.linspace(-200, 2000, 100))).plot_wake_map()
        plt.show()


class RathmannCounter(Rathmann):
    counter = 0

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        I, J = dw_ijlk.shape[:2]
        if I > 1 and I == J:
            # only count All2All
            self.counter += 1
        return Rathmann.calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_)


def get_convergence_wfm(x, speedup, wake_deficitModel=NoWakeDeficit()):
    # Unstable from beginning
    ti = 0.1

    ds = xr.Dataset(data_vars={'Speedup': (['x', 'y'], np.array([speedup] * 2).T), 'P': 1, 'TI': ti},
                    coords={'x': x, 'y': [0, 10000]})
    site = XRSite(ds, interp_method='nearest')

    wt = WindTurbines.from_WAsP_wtg(wtg_path + "Vestas-V80.wtg")
    blockage_deficitModel = RathmannCounter()
    return All2AllIterative(site, wt,
                            wake_deficitModel=wake_deficitModel,
                            blockage_deficitModel=blockage_deficitModel,
                            superpositionModel=LinearSum(),
                            convergence_tolerance=1e-6)


def test_convergence_hornsrev():
    site = Hornsrev1Site()
    wfm = All2AllIterative(site, windTurbines=V80(),
                           wake_deficitModel=NOJDeficit(),
                           blockage_deficitModel=RathmannCounter())
    x, y = site.initial_position.T
    sim_res = wfm(x, y, wd=90)
    assert wfm.blockage_deficitModel.counter == 13

    if 0:
        sim_res.flow_map().plot_wake_map()
        plt.show()


def test_convergence():
    """Unstable from beginning
    it:0, wt0 off, wt1 on due to site effects
    it:1, wt0 on(speedup from wt1), wt1 on
    it:2, wt0 on, wt1 off due to blockage from wt0
    it:3, wt0 off(no speedup from wt0), wt1 off
    and repeat if not handled
    """
    wfm = get_convergence_wfm([0, 200], [1.005, .995])
    sim_res = wfm(np.r_[200, [0] * 9], np.r_[-50, np.arange(9) * 200],
                  wd=270, ws=4)
    assert wfm.blockage_deficitModel.counter == 5

    if 0:
        sim_res.flow_map().plot_wake_map()
        plt.show()


def test_convergence2():
    # stable case. WT 0 should turn on due to speedup of wt1
    wfm = get_convergence_wfm([0, 250], [1.005, .995])
    sim_res = wfm(np.r_[250, [0] * 9], np.r_[-50, np.arange(9) * 200],
                  wd=270, ws=4)
    assert wfm.blockage_deficitModel.counter == 4
    assert np.all(sim_res.Power > 0)

    if 0:
        sim_res.flow_map().plot_wake_map()
        plt.show()


def test_convergence3():
    # Wake of WT1 makes WT0 and WT2 unstable as in test_convergence
    wfm = get_convergence_wfm([-400, -200, 0, 200], [1, 1, 1, .955], NOJDeficit())

    sim_res = wfm(np.r_[200, -400, 0, [0] * 9], np.r_[-100, -50, 0, np.arange(9) * 200 + 500],
                  wd=270, ws=4.6)
    assert wfm.blockage_deficitModel.counter == 7

    if 0:
        sim_res.flow_map().plot_wake_map()
        plt.show()
