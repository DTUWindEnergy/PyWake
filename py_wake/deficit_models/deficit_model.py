from abc import ABC, abstractmethod
import numpy as np
from numpy import newaxis as na


class DeficitModel(ABC):
    deficit_initalized = False

    def _calc_layout_terms(self, **_):
        """Calculate layout dependent terms, which is not updated during simulation"""

    @abstractmethod
    def calc_deficit(self):
        """Calculate wake deficit caused by the x'th most upstream wind turbines
        for all wind directions(l) and wind speeds(k) on a set of points(j)

        This method must be overridden by subclass

        Arguments required by this method must be added to the class list
        args4deficit

        See documentation of EngineeringWindFarmModel for a list of available input arguments

        Returns
        -------
        deficit_jlk : array_like
        """

    def calc_deficit_downwind(self, yaw_ilk, **kwargs):
        if np.any(yaw_ilk != 0):
            deficit_normal = self.calc_deficit(yaw_ilk=yaw_ilk, **kwargs)
            return deficit_normal
            return np.cos(yaw_ilk[:, na]) * deficit_normal
        else:
            return self.calc_deficit(yaw_ilk=yaw_ilk, **kwargs)


class BlockageDeficitModel(DeficitModel):
    def __init__(self, upstream_only=False, superpositionModel=None):
        """Parameters
        ----------
        upstream_only : bool, optional
            if true, downstream deficit from this model is set to zero
        superpositionModel : SuperpositionModel or None
            Superposition model used to sum blockage deficit.
            If None, the superposition model of the wind farm model is used
        """
        self.upstream_only = upstream_only
        self.superpositionModel = superpositionModel

    def calc_blockage_deficit(self, dw_ijlk, **kwargs):
        deficit_ijlk = self.calc_deficit(dw_ijlk=dw_ijlk, **kwargs)
        if self.upstream_only:
            rotor_pos = -1e-10
            deficit_ijlk *= (dw_ijlk < rotor_pos)
        return deficit_ijlk


class WakeDeficitModel(DeficitModel, ABC):

    def wake_radius(self, dw_ijlk, **_):
        """Calculates the radius of the wake of the i'th turbine
        for all wind directions(l) and wind speeds(k) at a set of points(j)

        This method must be overridden by subclass

        Arguments required by this method must be added to the class list
        args4deficit

        Returns
        -------
        wake_radius_ijlk : array_like
        """
        raise NotImplementedError("wake_radius not implemented for %s" % self.__class__.__name__)


class ConvectionDeficitModel(WakeDeficitModel):

    @abstractmethod
    def calc_deficit_convection(self):
        """Calculate wake deficit caused by the x'th most upstream wind turbines
        for all wind directions(l) and wind speeds(k) on a set of points(j)

        This method must be overridden by subclass

        Arguments required by this method must be added to the class list
        args4deficit

        See documentation of EngineeringWindFarmModel for a list of available input arguments

        Returns
        -------
        deficit_centre_ijlk : array_like
            Wind speed deficit caused by the i'th turbine at j'th downstream location, without accounting for crosswind distance (ie cw = 0)
        uc_ijlk : array_like
            Convection velocity of the i'th turbine at locations j
        sigma_sqr_ijlk : array_like
            Squared wake width of i'th turbine at j
        """
