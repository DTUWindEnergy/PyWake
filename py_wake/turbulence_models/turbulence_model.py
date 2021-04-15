from abc import abstractmethod
from py_wake.superposition_models import AddedTurbulenceSuperpositionModel, LinearSum, SqrMaxSum, MaxSum
from py_wake.utils.model_utils import check_model


class TurbulenceModel():

    def __init__(self, addedTurbulenceSuperpositionModel, rotorAvgModel=None):
        check_model(
            addedTurbulenceSuperpositionModel,
            AddedTurbulenceSuperpositionModel,
            'addedTurbulenceSuperpositionModel')
        self.addedTurbulenceSuperpositionModel = addedTurbulenceSuperpositionModel
        self.rotorAvgModel = rotorAvgModel

    @abstractmethod
    def calc_added_turbulence(self):
        """Calculate added turbulence intensity caused by the x'th most upstream wind turbines
        for all wind directions(l) and wind speeds(k) on a set of points(j)

        This method must be overridden by subclass

        Arguments required by this method must be added to the class list
        args4addturb

        See class documentation for examples and available arguments

        Returns
        -------
        add_turb_jlk : array_like
        """

    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        return self.addedTurbulenceSuperpositionModel.calc_effective_TI(TI_xxx, add_turb_jxxx)


# Aliases for backward compatibility. Use
# from py_wake.super_position_models import LinearSum, SqrMaxSum, MaxSum
LinearSum = LinearSum
SqrMaxSum = SqrMaxSum
MaxSum = MaxSum
