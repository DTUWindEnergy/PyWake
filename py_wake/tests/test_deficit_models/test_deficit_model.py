from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models.gcl import GCLTurbulenceModel
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
import pytest


def test_wake_radius_not_implemented():
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()
    wfm = PropagateDownwind(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                            superpositionModel=LinearSum(), turbulenceModel=GCLTurbulenceModel())
    with pytest.raises(NotImplementedError, match="wake_radius not implemented for NoWakeDeficit"):
        wfm(x, y)
