import pytest
import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models import VortexCylinder
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.examples.data import hornsrev1
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.superposition_models import LinearSum
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit, SelfSimilarityDeficit2020
from py_wake.deficit_models.vortexdipole import VortexDipole
from py_wake.deficit_models.rankinehalfbody import RankineHalfBody
from py_wake.deficit_models.hybridinduction import HybridInduction
from py_wake.deficit_models.rathmann import Rathmann, RathmannScaled
from py_wake.flow_map import XYGrid

debug = False


@pytest.fixture(scope='module')
def setup():
    site = Hornsrev1Site()
    windTurbines = hornsrev1.HornsrevV80()

    return site, windTurbines


@pytest.mark.parametrize('blockage_model,center_ref,side_ref', [
    (SelfSimilarityDeficit,
     [9.943, 9.915684, 9.865418, 9.763897, 9.542707, 10.0, 10.0, 10.0, 10.0, 10.0],
     [9.938289, 9.905028, 9.836523, 9.662802, 9.060236, 10.0, 10.0, 10.0, 10.0, 10.0]),
    (SelfSimilarityDeficit2020,
     [9.937128, 9.907163, 9.852059, 9.742072, 9.551456, 10.0, 10.0, 10.0, 10.0, 10.0],
     [9.931709, 9.89535, 9.821447, 9.6399, 9.072077, 10.0, 10.0, 10.0, 10.0, 10.0]),
    (VortexCylinder,
     [9.950421, 9.926564, 9.88241, 9.792084, 9.598908, 10.030945, 10.395886, 10.205344, 10.116382, 10.072811],
     [9.94641, 9.917526, 9.858037, 9.707178, 9.183896, 10.0, 10.0, 10.0, 10.0, 10.0]),
    (VortexDipole,
     [9.949202, 9.924034, 9.876704, 9.780608, 9.633199, 10.008154, 10.365033, 10.216741, 10.121995, 10.075302],
     [9.944808, 9.913676, 9.84628, 9.652978, 8.598016, 10.0, 10.0, 10.0, 10.0, 10.0]),
    (RankineHalfBody,
     [9.949202, 9.924034, 9.876704, 9.780608, 9.633199, 10.008154, 10.365033, 10.216741, 10.121995, 10.075302],
     [9.944808, 9.913676, 9.84628, 9.652978, 8.598016, 10.0, 10.0, 10.0, 10.0, 10.0]),
    (HybridInduction,
     [9.937128, 9.907163, 9.852059, 9.742072, 9.551456, 10.008154, 10.365033, 10.216741, 10.121995, 10.075302],
     [9.931709, 9.89535, 9.821447, 9.6399, 9.072077, 10.0, 10.0, 10.0, 10.0, 10.0]),
    (Rathmann,
     [9.950432, 9.926598, 9.882538, 9.792603, 9.597037, 10.041663, 10.397465, 10.204838, 10.116258, 10.072778],
     [9.94641, 9.917526, 9.858037, 9.707178, 9.190401, 10.0, 10.0, 10.0, 10.0, 10.0]),
    (RathmannScaled,
     [9.929791, 9.897708, 9.838744, 9.719092, 9.456743, 10.056169, 10.535845, 10.277511, 10.159648, 10.101453],
     [9.924278, 9.88539, 9.805788, 9.60523, 8.908535, 10.0, 10.0, 10.0, 10.0, 10.0]),
][::-1])
def test_blockage_map(setup, blockage_model, center_ref, side_ref):
    site, windTurbines = setup
    wm = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                          superpositionModel=LinearSum(), blockage_deficitModel=blockage_model())

    xy = np.linspace(-200, 200, 500)
    flow_map = wm(x=[0], y=[0], wd=[270], ws=[10]).flow_map(XYGrid(x=xy[::50], y=xy[[190, 250]]))
    X_j, Y_j = flow_map.XY
    WS_eff = flow_map.WS_eff_xylk[:, :, 0, 0]

    if debug:
        flow_map_full = wm(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        X_j_full, Y_j_full = flow_map_full.XY
        WS_eff_full = flow_map_full.WS_eff_xylk[:, :, 0, 0]
        plt.contourf(X_j_full, Y_j_full, WS_eff_full)
        plt.plot(X_j.T, Y_j.T, '.-')

        print(list(np.round(np.array(WS_eff[0]), 6)))
        print(list(np.round(np.array(WS_eff[1]), 6)))
        plt.title(blockage_model.__name__)
        plt.show()

    npt.assert_array_almost_equal(WS_eff[0], center_ref)
    npt.assert_array_almost_equal(WS_eff[1], side_ref)


@pytest.mark.parametrize('blockage_model,center_ref,side_ref', [
    (SelfSimilarityDeficit,
     [9.943, 9.915684, 9.865418, 9.763897, 9.542707, 10.0, 10.0, 6.223921, 6.782925, 7.226399],
     [9.938289, 9.905028, 9.836523, 9.662802, 9.060236, 4.560631, 5.505472, 6.223921, 6.782925, 7.226399]),
    (SelfSimilarityDeficit2020,
     [9.937128, 9.907163, 9.852059, 9.742072, 9.551456, 10.0, 10.0, 6.223921, 6.782925, 7.226399],
     [9.931709, 9.89535, 9.821447, 9.6399, 9.072077, 4.560631, 5.505472, 6.223921, 6.782925, 7.226399]),
    (VortexCylinder,
     [9.950421, 9.926564, 9.88241, 9.792084, 9.598908, 10.030945, 10.395886, 6.429265, 6.899307, 7.299209],
     [9.94641, 9.917526, 9.858037, 9.707178, 9.183896, 4.560631, 5.505472, 6.223921, 6.782925, 7.226399]),
    (VortexDipole,
     [9.949202, 9.924034, 9.876704, 9.780608, 9.633199, 10.008154, 10.365033, 6.440661, 6.90492, 7.301701],
     [9.944808, 9.913676, 9.84628, 9.652978, 8.598016, 4.560631, 5.505472, 6.223921, 6.782925, 7.226399]),
    (RankineHalfBody,
     [9.949202, 9.924034, 9.876704, 9.780608, 9.633199, 10.008154, 10.365033, 6.440661, 6.90492, 7.301701],
     [9.944808, 9.913676, 9.84628, 9.652978, 8.598016, 4.560631, 5.505472, 6.223921, 6.782925, 7.226399]),
    (HybridInduction,
     [9.937128, 9.907163, 9.852059, 9.742072, 9.551456, 10.008154, 10.365033, 6.440661, 6.90492, 7.301701],
     [9.931709, 9.89535, 9.821447, 9.6399, 9.072077, 4.560631, 5.505472, 6.223921, 6.782925, 7.226399]),
    (Rathmann,
     [9.950432, 9.926598, 9.882538, 9.792603, 9.597037, 10.041663, 10.397465, 6.428758, 6.899183, 7.299176],
     [9.94641, 9.917526, 9.858037, 9.707178, 9.190401, 4.560631, 5.505472, 6.223921, 6.782925, 7.226399]),
    (RathmannScaled,
     [9.929791, 9.897708, 9.838744, 9.719092, 9.456743, 10.056169, 10.535845, 6.501432, 6.942573, 7.327852],
     [9.924278, 9.88539, 9.805788, 9.60523, 8.908535, 4.560631, 5.505472, 6.223921, 6.782925, 7.226399]),
][::-1])
def test_wake_and_blockage(setup, blockage_model, center_ref, side_ref):
    site, windTurbines = setup
    noj_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NOJDeficit(),
                              blockage_deficitModel=blockage_model(), superpositionModel=LinearSum())

    xy = np.linspace(-200, 200, 500)
    flow_map = noj_ss(x=[0], y=[0], wd=[270], ws=[10]).flow_map(XYGrid(x=xy[::50], y=xy[[190, 250]]))
    X_j, Y_j = flow_map.XY
    WS_eff = flow_map.WS_eff_xylk[:, :, 0, 0]

    if debug:
        flow_map_full = noj_ss(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        X_j_full, Y_j_full = flow_map_full.XY
        WS_eff_full = flow_map_full.WS_eff_xylk[:, :, 0, 0]
        plt.contourf(X_j_full, Y_j_full, WS_eff_full)
        plt.plot(X_j.T, Y_j.T, '.-')

        print(list(np.round(np.array(WS_eff[0]), 6)))
        print(list(np.round(np.array(WS_eff[1]), 6)))
        plt.title(blockage_model.__name__)
        plt.show()

    npt.assert_array_almost_equal(WS_eff[0], center_ref)
    npt.assert_array_almost_equal(WS_eff[1], side_ref)


@pytest.mark.parametrize('blockage_model,blockage_loss', [
    (SelfSimilarityDeficit, 0.4864793),
    (SelfSimilarityDeficit2020, 0.5372064),
    (VortexCylinder, 0.4233034),
    (VortexDipole, 0.4321021),
    (RankineHalfBody, 0.4321021),
    (HybridInduction, 0.5372064),
    (Rathmann, 0.4233034298949511),
    (RathmannScaled, 0.5928161590163314)
][::-1])
def test_aep_two_turbines(setup, blockage_model, blockage_loss):
    site, windTurbines = setup

    nwm_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                              blockage_deficitModel=blockage_model(), superpositionModel=LinearSum())

    sim_res = nwm_ss(x=[0, 80 * 3], y=[0, 0])
    aep_no_blockage = sim_res.aep_ilk(with_wake_loss=False).sum(2)
    aep = sim_res.aep_ilk().sum(2)

    # blockage reduce aep(wd=270) by .5%
    npt.assert_almost_equal((aep_no_blockage[0, 270] - aep[0, 270]) / aep_no_blockage[0, 270] * 100, blockage_loss)

    if debug:
        plt.plot(sim_res.WS_eff_ilk[:, :, 7].T)
        plt.title(blockage_model.__name__)
        plt.show()
