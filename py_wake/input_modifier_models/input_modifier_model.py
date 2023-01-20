from py_wake.utils.model_utils import method_args


class InputModifierModel():
    @property
    def args4model(self):
        args4model = method_args(self.__call__)
        args4model |= method_args(self.setup)
        return args4model - {'kwargs', 'model_kwargs'}

    def __call__(self, **_):
        """"""
        return {}

    def setup(self, **_):
        """"""
        return {}
