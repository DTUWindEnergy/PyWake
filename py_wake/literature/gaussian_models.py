import numpy as np
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.superposition_models import LinearSum, WeightedSum
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, NiayifarGaussianDeficit, ZongGaussianDeficit,\
    BlondelSuperGaussianDeficit2020, CarbajofuertesGaussianDeficit
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.superposition_models import SqrMaxSum
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80


class Bastankhah_PorteAgel_2014(PropagateDownwind):
    """
    Implemented according to:
        Bastankhah M and Porté-Agel F.
        A new analytical model for wind-turbine wakes.
        J. Renew. Energy. 2014;70:116-23.

    Description:
        - Conservation of mass and momentum is applied with the assumption of a Gaussian shape for the wake profile in the calculation of the wake deficit.
        - Only one parameter needed to determine the velocity distribution: the wake expansion parameter k.
    """

    def __init__(self, site, windTurbines,
                 k, ceps=.2, ct2a=ct2a_mom1d, use_effective_ws=True,
                 rotorAvgModel=None,
                 superpositionModel=LinearSum(),
                 deflectionModel=None,
                 turbulenceModel=None,
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float
            Wake expansion factor
        use_effective_ws : bool
            Option to use either the local (True) or free-stream wind speed (False) experienced by the ith turbine
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=BastankhahGaussianDeficit(ct2a=ct2a, k=k, ceps=ceps,
                                                                               use_effective_ws=use_effective_ws,
                                                                               rotorAvgModel=rotorAvgModel,
                                                                               groundModel=groundModel),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class Niayifar_PorteAgel_2016(PropagateDownwind):
    """
    Implemented according to:
        Amin Niayifar and Fernando Porté-Agel
        Analytical Modeling of Wind Farms: A New Approach for Power Prediction
        Energies 2016, 9, 741; doi:10.3390/en9090741 [1]

    Features:
        - Conservation of mass and momentum and a Gaussian shape for the wake profile.
        - Wake expansion function of local turbulence intensity.

    Description:
        - The expansion rate 'k' varies linearly with local turbluence intensity: k = a1 I + a2.
        - The default constants are set according to publications by Porte-Agel's group, which are based on LES simulations. Lidar field measurements by Fuertes et al. (2018) indicate that a = [0.35, 0.0] is also a valid selection.
        - Wake superposition is represented by linearly adding the wakes.
        - The Crespo Hernandez turbulence model is used to calculate the added streamwise turbulence intensity, Eq 14 in [1].
    """

    def __init__(self, site, windTurbines,
                 a=[0.3837, 0.003678], ceps=.2, ct2a=ct2a_mom1d, use_effective_ws=True, use_effective_ti=True,
                 superpositionModel=LinearSum(),
                 deflectionModel=None,
                 turbulenceModel=CrespoHernandez(ct2a=ct2a_mom1d, c=[0.73, 0.8325, 0.0325, -0.32], addedTurbulenceSuperpositionModel=SqrMaxSum()),
                 rotorAvgModel=GaussianOverlapAvgModel(),
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        superpositionModel : SuperpositionModel, default LinearSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default CrespoHernandez
            Model describing the amount of added turbulence in the wake
        use_effective_ws : bool
            Option to use either the local (True) or free-stream (False) wind speed experienced by the ith turbine
        use_effective_ti : bool
            Option to use either the local (True) or free-stream (False) turbulence intensity experienced by the ith turbine
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NiayifarGaussianDeficit(a=a, ceps=ceps, ct2a=ct2a,
                                                                             rotorAvgModel=rotorAvgModel,
                                                                             groundModel=groundModel,
                                                                             use_effective_ws=use_effective_ws,
                                                                             use_effective_ti=use_effective_ti),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class CarbajoFuertes_etal_2018(PropagateDownwind):
    """
    Implemented according to:
        Carbajo Fuertes, F., Markfort, C. D., & Porté-Agel, F.:
        Wind turbine wake characterization with nacelle-mounted wind lidars for
        analytical wake model validation, Remote Sensing, 10(5), 668, (2018)
        https://doi.org/10.3390/rs10050668

    The paper does not present a wind farm model, so here we used the one by Niayifar.
    """

    def __init__(self, site, windTurbines,
                 a=[0.35, 0], ceps=[-1.91, 0.34], ct2a=ct2a_mom1d, use_effective_ws=True, use_effective_ti=True,
                 superpositionModel=LinearSum(),
                 deflectionModel=None,
                 turbulenceModel=CrespoHernandez(ct2a=ct2a_mom1d, c=[0.73, 0.8325, 0.0325, -0.32], addedTurbulenceSuperpositionModel=SqrMaxSum()),
                 rotorAvgModel=GaussianOverlapAvgModel(),
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        superpositionModel : SuperpositionModel, default LinearSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default CrespoHernandez
            Model describing the amount of added turbulence in the wake
        use_effective_ws : bool
            Option to use either the local (True) or free-stream (False) wind speed experienced by the ith turbine
        use_effective_ti : bool
            Option to use either the local (True) or free-stream (False) turbulence intensity experienced by the ith turbine
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=CarbajofuertesGaussianDeficit(a=a, ceps=ceps, ct2a=ct2a,
                                                                                   rotorAvgModel=rotorAvgModel,
                                                                                   groundModel=groundModel,
                                                                                   use_effective_ws=use_effective_ws,
                                                                                   use_effective_ti=use_effective_ti),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class Zong_PorteAgel_2020(PropagateDownwind):
    """
    Implemented according to:
        Haohua Zong and Fernando Porté-Agel
        "A momentum-conserving wake superposition method for wind farm power prediction"
        J. Fluid Mech. (2020), vol. 889, A8; doi:10.1017/jfm.2020.77

    Features:
        - Conservation of mass and momentum and a Gaussian shape for the wake profile.
        - Wake expansion function of local turbulence intensity.
        - New wake width expression following the approach by Shapiro et al. (2018).

    Description:
        Extension of the Niayifar et al. (2016) implementation with adapted
        Shapiro wake model components, namely a gradual growth of the thrust
        force and an expansion factor not falling below the rotor diameter.
        Shapiro modelled the pressure and thrust force as a combined momentum
        source, that are distributed in the streamwise direction with a Gaussian
        kernel with a certain characteristic length. As a result the induction
        changes following an error function. Zong chose to use a characteristic
        length of D/sqrt(2) and applies it directly to the thrust not the induction
        as Shapiro. This leads to the full thrust being active only 2D downstream of
        the turbine. Zong's wake width expression is inspired by Shapiro's, however
        the start of the linear wake expansion region (far-wake) was related to
        the near-wake length by Vermeulen (1980). The epsilon factor that in the
        original Gaussian model was taken to be a function of CT is now a constant
        as proposed by Bastankhah (2016), as the near-wake length now effectively
        dictates the origin of the far-wake.
    """

    def __init__(self, site, windTurbines,
                 a=[0.38, 4e-3], deltawD=1. / np.sqrt(2), eps_coeff=0.35, lam=7.5, B=3,
                 use_effective_ws=True, use_effective_ti=True,
                 rotorAvgModel=None,
                 superpositionModel=WeightedSum(),
                 deflectionModel=None,
                 turbulenceModel=CrespoHernandez(ct2a=ct2a_mom1d, c=[0.73, 0.83, 0.03, -0.32], addedTurbulenceSuperpositionModel=SqrMaxSum()),
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        use_effective_ws : bool
            Option to use either the local (True) or free-stream (False) wind speed experienced by the ith turbine
        use_effective_ti : bool
            Option to use either the local (True) or free-stream (False) turbulence intensity experienced by the ith turbine
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=ZongGaussianDeficit(a=a, deltawD=deltawD, eps_coeff=eps_coeff, lam=lam, B=B,
                                                                         rotorAvgModel=rotorAvgModel,
                                                                         groundModel=groundModel,
                                                                         use_effective_ws=use_effective_ws,
                                                                         use_effective_ti=use_effective_ti),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


class Blondel_Cathelain_2020(PropagateDownwind):
    """
    Implemented according to:
        Blondel and Cathelain (2020)
        An alternative form of the super-Gaussian wind turbine wake model
        Wind Energ. Sci., 5, 1225–1236, 2020 https://doi.org/10.5194/wes-5-1225-2020 [1]

    Features:
        - Wake profile transitions from top-hat at near wake to Gaussian at far wake.
        - characteristic wake width (sigma) function of turbulence intensity and CT.
        - evolution of super gaussian "n" order function of downwind distance and turbulence intensity.

    Description:
        - Super gaussian wake order "n" is determined with the calibrated parameters: a_f, b_f, c_f; with a_f kept constant at 3.11.
        - Calibrated parameters taken from Table 2 and 3 in [1].
        - Linear summation of the wakes based on Shapiro (2019) https://www.mdpi.com/1996-1073/12/15/2956
        - Turbulence model is set to None. The Crespo Hernandez model is recommended.
    """

    def __init__(self, site, windTurbines,
                 use_effective_ws=True, use_effective_ti=True,
                 superpositionModel=LinearSum(),
                 deflectionModel=None,
                 turbulenceModel=None,
                 rotorAvgModel=None,
                 groundModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        use_effective_ws : bool
            Option to use either the local (True) or free-stream (False) wind speed experienced by the ith turbine
        use_effective_ti : bool
            Option to use either the local (True) or free-stream (False) turbulence intensity experienced by the ith turbine
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=BlondelSuperGaussianDeficit2020(rotorAvgModel=rotorAvgModel,
                                                                                     groundModel=groundModel,
                                                                                     use_effective_ws=use_effective_ws,
                                                                                     use_effective_ti=use_effective_ti),
                                   superpositionModel=superpositionModel, deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


def main():
    if __name__ == '__main__':

        site = Hornsrev1Site()
        windTurbines = V80()
        x, y = site.initial_position.T

        for wf_model in [Bastankhah_PorteAgel_2014(site, windTurbines, k=0.0324555),
                         Niayifar_PorteAgel_2016(site, windTurbines),
                         CarbajoFuertes_etal_2018(site, windTurbines),
                         Zong_PorteAgel_2020(site, windTurbines),
                         Blondel_Cathelain_2020(site, windTurbines, turbulenceModel=CrespoHernandez())]:

            # run wind farm simulation
            sim_res = wf_model(x, y)

            # calculate AEP
            aep = sim_res.aep().sum()
            print(aep)


main()
