import pytest

import matplotlib.pyplot as plt
from py_wake import np
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit2020
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37WindTurbines
from py_wake.flow_map import XYGrid
from py_wake.superposition_models import WeightedSum
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.wind_farm_models.engineering_models import All2All


@pytest.mark.parametrize('kwargs', [
    # deflection and turbulence
    dict(wake_deficitModel=BastankhahGaussianDeficit(),
         turbulenceModel=STF2017TurbulenceModel(),
         deflectionModel=JimenezWakeDeflection()),
    # wake_radius and blockage
    dict(wake_deficitModel=NOJDeficit(), blockage_deficitModel=SelfSimilarityDeficit2020()),
    # weightedsum and blockage
    dict(wake_deficitModel=BastankhahGaussianDeficit(),
         superpositionModel=WeightedSum(),
         blockage_deficitModel=SelfSimilarityDeficit2020())])
def test_All2All(kwargs):
    site = IEA37Site(16)
    windTurbines = IEA37WindTurbines()

    wfm = All2All(site, windTurbines, **kwargs)

    sim_res = wfm([0, 500, 1000, 1500], [0, 0, 0, 0],
                  wd=270, ws=10, yaw=[30, -30, 30, -30], tilt=0)

    if 0:
        sim_res.flow_map(
            XYGrid(x=np.linspace(-200, 2000, 100))).plot_wake_map()
        plt.show()
