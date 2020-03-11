import numpy as np
import pytest
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake import NOJ, Fuga
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.tests.test_files.fuga import LUT_path_2MW_z0_0_03
from py_wake.flow_map import HorizontalGrid
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.superposition_models import SquaredSum
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
from py_wake.turbulence_models.stf import STF2005TurbulenceModel
from py_wake.deflection_models.jimenez import JimenezWakeDeflection


def test_wake_model():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(site, windTurbines)

    with pytest.raises(ValueError, match="Turbines 0 and 1 are at the same position"):
        wake_model([0, 0], [0, 0], wd=np.arange(0, 360, 22.5), ws=[9.8])


def test_wec():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wake_model = Fuga(LUT_path_2MW_z0_0_03, site, wts)
    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    flow_map_wec1 = wake_model(wt_x, wt_y, 70, wd=[30], ws=[10]).flow_map(HorizontalGrid(x_j, y_j))
    Z_wec1 = flow_map_wec1.WS_eff_xylk[:, :, 0, 0]
    wake_model.wec = 2
    flow_map_wec2 = wake_model(wt_x, wt_y, 70, wd=[30], ws=[10]).flow_map(HorizontalGrid(x_j, y_j))
    X, Y = flow_map_wec1.XY
    Z_wec2 = flow_map_wec2.WS_eff_xylk[:, :, 0, 0]

    if 0:
        import matplotlib.pyplot as plt

        flow_map_wec1.plot_wake_map(levels=np.arange(6, 10.5, .1), plot_colorbar=False)
        plt.plot(X[0], Y[140])
        wts.plot(wt_x, wt_y)
        plt.figure()
        c = flow_map_wec2.plot_wake_map(levels=np.arange(6, 10.5, .1), plot_colorbar=False)
        plt.colorbar(c)
        plt.plot(X[0], Y[140])
        wts.plot(wt_x, wt_y)

        plt.figure()
        plt.plot(X[0], Z_wec1[140, :], label="Z=70m")
        plt.plot(X[0], Z_wec2[140, :], label="Z=70m")
        plt.plot(X[0, 100:400:10], Z_wec1[140, 100:400:10], '.')
        plt.plot(X[0, 100:400:10], Z_wec2[140, 100:400:10], '.')
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(
        Z_wec1[140, 100:400:10],
        [10.0547, 10.0519, 10.0718, 10.0093, 9.6786, 7.8589, 6.8539, 9.2199,
         9.9837, 10.036, 10.0796, 10.0469, 10.0439, 9.1866, 7.2552, 9.1518,
         10.0449, 10.0261, 10.0353, 9.9256, 9.319, 8.0062, 6.789, 8.3578,
         9.9393, 10.0332, 10.0191, 10.0186, 10.0191, 10.0139], 4)
    npt.assert_array_almost_equal(
        Z_wec2[140, 100:400:10],
        [10.0297, 9.9626, 9.7579, 9.2434, 8.2318, 7.008, 6.7039, 7.7303, 9.0101,
         9.6877, 9.9068, 9.7497, 9.1127, 7.9505, 7.26, 7.9551, 9.2104, 9.7458,
         9.6637, 9.1425, 8.2403, 7.1034, 6.5109, 7.2764, 8.7653, 9.7139, 9.9718,
         10.01, 10.0252, 10.0357], 4)


def test_str():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wf_model = All2AllIterative(site, windTurbines,
                                wake_deficitModel=NOJDeficit(),
                                superpositionModel=SquaredSum(),
                                blockage_deficitModel=SelfSimilarityDeficit(),
                                deflectionModel=JimenezWakeDeflection(),
                                turbulenceModel=STF2005TurbulenceModel())
    assert str(wf_model) == "All2AllIterative(EngineeringWindFarmModel, NOJDeficit-wake, SelfSimilarityDeficit-blockage, SquaredSum-superposition, JimenezWakeDeflection-deflection, STF2005TurbulenceModel-turbulence)"


@pytest.mark.parametrize('deflectionModel', [(None),
                                             (JimenezWakeDeflection())])
def test_huge_flow_map(deflectionModel):
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(site, windTurbines, deflectionModel=deflectionModel,
                     turbulenceModel=STF2005TurbulenceModel())
    n_wt = 2
    flow_map = wake_model(*site.initial_position[:n_wt].T, wd=0).flow_map(HorizontalGrid(resolution=1000))
    # check that deficit matrix > 10MB (i.e. it enters the memory saving loop)
    assert (np.prod(flow_map.WS_eff_xylk.shape) * n_wt * 8 / 1024**2) > 10
    assert flow_map.WS_eff_xylk.shape == (1000, 1000, 1, 1)


def test_aep():
    site = UniformSite([1], ti=0)
    windTurbines = IEA37_WindTurbines()
    wake_model = NOJ(site, windTurbines)
    sim_res = wake_model([0], [0], wd=270)

    npt.assert_almost_equal(sim_res.aep(), 3.35 * 24 * 365 / 360 / 1000)
    npt.assert_almost_equal(sim_res.aep(normalize_probabilities=True), 3.35 * 24 * 365 / 1000)


def test_All2AllIterativeDeflection():
    site = IEA37Site(16)
    windTurbines = IEA37_WindTurbines()
    wf_model = All2AllIterative(site, windTurbines,
                                wake_deficitModel=NOJDeficit(),
                                superpositionModel=SquaredSum(),
                                deflectionModel=JimenezWakeDeflection())
    wf_model([0], [0])
