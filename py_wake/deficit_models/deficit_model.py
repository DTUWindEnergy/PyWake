from abc import ABC, abstractmethod


class DeficitModel(ABC):
    deficit_initalized = False
    args4deficit = ['WS_ilk', 'dw_ijlk']

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
