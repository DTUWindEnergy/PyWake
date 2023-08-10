from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.superposition_models import SquaredSum
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.deficit_models.utils import ct2a_madsen
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80


class Jensen_1983(PropagateDownwind):
    """
    Implemented according to:
        Niels Otto Jensen, “A note on wind generator interaction.” (1983)

    Features:
        - Top-hat shape for the wake profile.
        - Wake superposition is done with the squared sum model as default.
        - Given the top-hat shape, the Area Overlap Average model is used as default, which includes the wake radius.
    """

    def __init__(self, site, windTurbines, rotorAvgModel=AreaOverlapAvgModel(),
                 ct2a=ct2a_madsen, k=.1, superpositionModel=SquaredSum(), deflectionModel=None, turbulenceModel=None,
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float, default 0.1
            wake expansion factor
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel, default None
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NOJDeficit(
                                       k=k, ct2a=ct2a, rotorAvgModel=rotorAvgModel, groundModel=groundModel),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


def main():
    if __name__ == '__main__':
        site = Hornsrev1Site()
        windTurbines = V80()
        x, y = site.initial_position.T
        wfm = Jensen_1983(site, windTurbines)
        sim_res = wfm(x, y)
        aep = sim_res.aep().sum()
        print(aep)


main()
