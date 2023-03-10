from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.ground_models.ground_models import Mirror
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.deficit_models.utils import ct2a_mom1d
import os


class TurbOPark(PropagateDownwind):
    def __init__(self, site, windTurbines):

        wake_deficitModel = TurboGaussianDeficit(
            ct2a=ct2a_mom1d,
            groundModel=Mirror(),
            rotorAvgModel=GaussianOverlapAvgModel())

        # Ørsted scales the deficit with respect to the ambient wind speed of the downstream turbine:
        wake_deficitModel.WS_key = 'WS_jlk'

        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=wake_deficitModel,
                                   superpositionModel=SquaredSum())
