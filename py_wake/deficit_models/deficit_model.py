from abc import ABC, abstractmethod


class DeficitModel(ABC):
    deficit_initalized = False

    def __init__(self):
        self.args4deficit = ['WS_ilk', 'dw_ijlk']

    def _calc_layout_terms(self, **_):
        pass

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


class ConvectionDeficitModel(DeficitModel):

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
