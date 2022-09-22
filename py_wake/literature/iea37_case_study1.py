from py_wake import np
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37WindTurbines
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import All2All
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm


class IEA37CaseStudy1(All2All):
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
        site.default_wd = np.arange(0, 360, 22.5)

        windTurbines = IEA37WindTurbines()
        All2All.__init__(self, site, windTurbines,
                         wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit(),
                         superpositionModel=SquaredSum(),
                         deflectionModel=deflectionModel)


def main():
    if __name__ == '__main__':
        n_wt = 64
        wfm = IEA37CaseStudy1(n_wt)
        x, y, aep_ref = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)
        print(n_wt, aep_ref[0] * 1e-3, wfm(x, y).aep().sum().item())


main()
