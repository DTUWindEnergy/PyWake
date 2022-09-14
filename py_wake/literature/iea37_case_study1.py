from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37WindTurbines


class IEA37CaseStudy1(PropagateDownwind):
    """Wind Farm model corresponding to the setup for IEA37 Wind Farm Layout Optimization Case Studies 1 and 2
    https://github.com/IEAWindTask37/iea37-wflo-casestudies/tree/master/cs1-2"""

    def __init__(self, n_wt, deflectionModel=None):
        """
        Parameters
        ----------
        n_wt : {16, 32, 64}site : Site
            Number of wind turbines
        """
        site = IEA37Site(n_wt)
        windTurbines = IEA37WindTurbines()
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                                   superpositionModel=SquaredSum(),
                                   deflectionModel=deflectionModel)
