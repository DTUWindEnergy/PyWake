import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.deficit_models.noj import NOJ
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine, SimpleGenericWindTurbine
from py_wake.utils import fuga_utils
from py_wake.wind_farm_models.minimalistic_wind_farm_model import MinimalisticWindFarmModel, CorrectionFactorCalibrated
from py_wake.tests import npt
import pytest


def test_MinimalisticWindFarmModel():

    wt = SimpleGenericWindTurbine(name='Simple', diameter=80, hub_height=70, power_norm=2000)

    ti = fuga_utils.ti(z0=0.0001, zref=wt.hub_height(), zeta0=0)[0]

    site = Hornsrev1Site(ti=ti)
    x, y = site.initial_position.T
    for wfm, eff, ref in [(MinimalisticWindFarmModel(site, wt, 3, 55), 1, 547.179521),
                          (MinimalisticWindFarmModel(site, wt, CorrectionFactorCalibrated(), 55), .91, 586.5399377399355)]:
        res = wfm.aep(x, y) * eff
        npt.assert_allclose(res, ref, rtol=0.001)


def test_MinimalisticWindFarmModel_specify_args():
    wt = V80()

    ti = fuga_utils.ti(z0=0.0001, zref=wt.hub_height(), zeta0=0)[0]

    site = Hornsrev1Site(ti=ti)
    x, y = site.initial_position.T
    with pytest.raises(AttributeError, match="'V80' object has no attribute 'max_cp'"):
        MinimalisticWindFarmModel(site, wt, 3, 55)

    aep1 = MinimalisticWindFarmModel(site, wt, 3, 55, max_cp=.48).aep(x, y)
    aep2 = MinimalisticWindFarmModel(site, wt, 3, 55, max_cp=.48, ws_cutin=4, ws_cutout=25).aep(x, y)
    npt.assert_allclose(aep1, aep2)
