from numpy import newaxis as na
from py_wake import np
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models import WakeDeficitModel


class NoWakeDeficit(WakeDeficitModel):
    def __init__(self, **_):
        DeficitModel.__init__(self, groundModel=None, rotorAvgModel=None)

    def calc_deficit(self, WS_ilk, dw_ijlk, **_):
        return (WS_ilk)[:, na] * (dw_ijlk > 0) * 0

    def wake_radius(self, dw_ijlk, **_):
        return np.zeros_like(dw_ijlk)
