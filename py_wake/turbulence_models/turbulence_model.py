from abc import abstractmethod, ABC
from py_wake.superposition_models import AddedTurbulenceSuperpositionModel, LinearSum
from py_wake.utils.model_utils import check_model, method_args, RotorAvgAndGroundModelContainer, XRLUTModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel
from py_wake.ground_models.ground_models import GroundModel


class TurbulenceModel(ABC, RotorAvgAndGroundModelContainer):

    def __init__(self, addedTurbulenceSuperpositionModel=LinearSum(), rotorAvgModel=None, groundModel=None):
        for model, cls, name in [(addedTurbulenceSuperpositionModel, AddedTurbulenceSuperpositionModel, 'addedTurbulenceSuperpositionModel'),
                                 (rotorAvgModel, RotorAvgModel, 'rotorAvgModel'),
                                 (groundModel, GroundModel, 'groundModel')]:
            check_model(model, cls, name)

        self.addedTurbulenceSuperpositionModel = addedTurbulenceSuperpositionModel
        RotorAvgAndGroundModelContainer.__init__(self, groundModel, rotorAvgModel)

    @property
    def args4model(self):
        args4model = RotorAvgAndGroundModelContainer.args4model.fget(self)  # @UndefinedVariable
        args4model |= method_args(self.calc_added_turbulence)
        return args4model

    def __call__(self, **kwargs):
        f = self.calc_added_turbulence
        if self.rotorAvgModel:
            f = self.rotorAvgModel.wrap(f)
        return f(**kwargs)

    @abstractmethod
    def calc_added_turbulence(self):
        """Calculate added turbulence intensity caused by the x'th most upstream wind turbines
        for all wind directions(l) and wind speeds(k) on a set of points(j)

        This method must be overridden by subclass

        See class documentation for examples and available arguments

        Returns
        -------
        add_turb_jlk : array_like
        """

    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        return self.addedTurbulenceSuperpositionModel.calc_effective_TI(TI_xxx, add_turb_jxxx)


class XRLUTTurbulenceModel(TurbulenceModel, XRLUTModel, ):
    def __init__(self, da, get_input=None, get_output=None, bounds='limit',
                 addedTurbulenceSuperpositionModel=LinearSum(), rotorAvgModel=None, groundModel=None):
        XRLUTModel.__init__(self, da, get_input=get_input, get_output=get_output, bounds=bounds)
        TurbulenceModel.__init__(self, addedTurbulenceSuperpositionModel, rotorAvgModel=rotorAvgModel,
                                 groundModel=groundModel)

    def calc_added_turbulence(self, **kwargs):
        return XRLUTModel.__call__(self, **kwargs)

    @property
    def args4model(self):
        return TurbulenceModel.args4model.fget(self) | XRLUTModel.args4model.fget(self)  # @UndefinedVariable
