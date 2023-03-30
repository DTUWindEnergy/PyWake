import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import V80
from py_wake.site.xrsite import UniformSite
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.input_modifier_models.multirotor import MultiRotorWindTurbines, MultiRotor
import pytest
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.tests import npt
import numpy as np
from py_wake.deficit_models.utils import ct2a_mom1d


def test_multirotor_windturbine_not_instantiated():
    with pytest.raises(ValueError, match="Did you forget the brackets: V80()"):
        MultiRotorWindTurbines(V80)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
def test_multirotor(wfm_cls):

    wts = MultiRotorWindTurbines(V80(), [[0, 0, 0],  # 0: normal
                                         [50, 10, 0],  # 1: right rotor
                                         [-50, 10, 0]  # 2: left rotor
                                         ])

    wfm = wfm_cls(site=UniformSite(), windTurbines=wts, wake_deficitModel=BastankhahGaussianDeficit(ct2a=ct2a_mom1d),
                  inputModifierModels=MultiRotor())

    sim_res = wfm([0, 0, 500], [0, 0, 0], type=[1, 2, 0], wd=[0, 90, 180, 270])
    if 0:
        for wd, ax in zip(sim_res.wd, plt.subplots(2, 2)[1].flatten()):
            sim_res.flow_map(wd=wd).plot_wake_map(ax=ax)
        plt.show()
    wfm = All2AllIterative(site=UniformSite(), windTurbines=wts,
                           wake_deficitModel=BastankhahGaussianDeficit(ct2a=ct2a_mom1d))
    sim_res2 = wfm([[50, -10, -50, 10], [-50, -10, 50, 10], [500, 500, 500, 500]],
                   [[-10, -50, 10, 50], [-10, 50, 10, -50], [0, 0, 0, 0]],
                   wd=[0, 90, 180, 270])

    for k in ['x', 'y']:
        npt.assert_array_almost_equal(sim_res[k], sim_res2[k], 10)
    npt.assert_array_almost_equal(sim_res.WS_eff, sim_res2.WS_eff, 4)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
def test_multirotor_deficit_profile(wfm_cls):
    wts = MultiRotorWindTurbines(V80(), [[0, 0, 0],  # 0: normal
                                         [50, 10, 0],  # 1: right rotor
                                         [-50, 10, 0]  # 2: left rotor
                                         ])

    wfm = wfm_cls(site=UniformSite(), windTurbines=wts, wake_deficitModel=BastankhahGaussianDeficit(ct2a=ct2a_mom1d),
                  inputModifierModels=MultiRotor())
    sim_res = wfm([0, 0, 500], [0, 0, 0], type=[1, 2, 0], wd=np.linspace(250, 290, 11))
    if 0:
        print(np.round(sim_res.WS_eff.sel(wt=2).squeeze().values, 2).tolist())
        sim_res.WS_eff.sel(wt=2).squeeze().plot()
        plt.show()
    npt.assert_array_almost_equal(sim_res.WS_eff.sel(wt=2).squeeze(),
                                  [11.99, 11.87, 11.02, 9.23, 8.91, 9.69, 8.91, 9.23, 11.02, 11.87, 11.99], 2)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
def test_dynamic_input_modifier(wfm_cls):
    class DynamicMultiRotor(MultiRotor):

        def __call__(self, x_ilk, y_ilk, ct_ilk, **_):
            return {'x_ilk': x_ilk + ct_ilk, 'y_ilk': y_ilk + ct_ilk}

    wts = MultiRotorWindTurbines(V80(), [[0, 0, 0],  # 0: normal
                                         [50, 10, 0],  # 1: right rotor
                                         [-50, 10, 0]  # 2: left rotor
                                         ])

    wfm = wfm_cls(site=UniformSite(), windTurbines=wts, wake_deficitModel=BastankhahGaussianDeficit(),
                  inputModifierModels=DynamicMultiRotor())

    sim_res = wfm([0, 0, 500], [0, 0, 0], type=[1, 2, 0], wd=[0, 90, 180, 270])
    if 0:
        for wd, ax in zip(sim_res.wd, plt.subplots(2, 2)[1].flatten()):
            sim_res.flow_map(wd=wd).plot_wake_map(ax=ax)
        plt.show()
    wfm = All2AllIterative(site=UniformSite(), windTurbines=wts, wake_deficitModel=BastankhahGaussianDeficit())
    sim_res2 = wfm([[50, -10, -50, 10], [-50, -10, 50, 10], [500, 500, 500, 500]],
                   [[-10, -50, 10, 50], [-10, 50, 10, -50], [0, 0, 0, 0]],
                   wd=[0, 90, 180, 270])

    for k in ['x', 'y']:
        npt.assert_array_almost_equal(sim_res[k], (sim_res2[k] + sim_res2.CT).squeeze(), 4)
