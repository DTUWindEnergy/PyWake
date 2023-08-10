from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.deficit_models.fuga import FugaDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.tests.test_files import tfp


class Ott_Nielsen_2014(PropagateDownwind):
    """
    Implemented according to:
        Ott, S., & Nielsen, M. (2014).
        Developments of the offshore wind turbine wake model Fuga.
        DTU Wind Energy. DTU Wind Energy E No. 0046

    Description:
        - Fuga is a linearized CFD model that can predict wake effects for offshore wind farms.
        - It has the ability to work with different types of turbines for the same project, which makes it useful for inter farm interactions.
    """

    def __init__(self, LUT_path, site, windTurbines,
                 rotorAvgModel=None, deflectionModel=None, turbulenceModel=None, remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=FugaDeficit(LUT_path, remove_wriggles=remove_wriggles),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)


class Ott_Nielsen_2014_Blockage(All2AllIterative):
    """
    Implemented according to:
        Ott, S., & Nielsen, M. (2014).
        Developments of the offshore wind turbine wake model Fuga.
        DTU Wind Energy. DTU Wind Energy E No. 0046

    Description:
        - Fuga is a linearized CFD model that can predict wake effects for offshore wind farms.
        - It has the ability to work with different types of turbines for the same project, which makes it useful for inter farm interactions.
        - An additional blockage model is added and the All2AllIterative wind farm model is used to model blockage effects.
    """

    def __init__(self, LUT_path, site, windTurbines, rotorAvgModel=None,
                 deflectionModel=None, turbulenceModel=None, convergence_tolerance=1e-6, remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        fuga_deficit = FugaDeficit(LUT_path, remove_wriggles=remove_wriggles)
        All2AllIterative.__init__(self, site, windTurbines, wake_deficitModel=fuga_deficit,
                                  rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                  deflectionModel=deflectionModel, blockage_deficitModel=fuga_deficit,
                                  turbulenceModel=turbulenceModel, convergence_tolerance=convergence_tolerance)


def main():
    if __name__ == '__main__':

        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'

        site = Hornsrev1Site()
        windTurbines = V80()
        x, y = site.initial_position.T

        for wf_model in [Ott_Nielsen_2014(path, site, windTurbines),
                         Ott_Nielsen_2014_Blockage(path, site, windTurbines)]:

            # run wind farm simulation
            sim_res = wf_model(x, y)

            # calculate AEP
            aep = sim_res.aep().sum()
            print(aep)


main()
