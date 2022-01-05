from numpy import newaxis as na
import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models import WakeDeficitModel
from py_wake.ground_models.ground_models import NoGround


class NoWakeDeficit(WakeDeficitModel):
    args4deficit = ['WS_ilk']

    def __init__(self):
        DeficitModel.__init__(self, groundModel=NoGround())

    def calc_deficit(self, WS_ilk, dw_ijlk, **_):
        return (WS_ilk)[:, na] * (dw_ijlk > 0) * 0

    def wake_radius(self, dw_ijlk, **_):
        return np.zeros_like(dw_ijlk)
