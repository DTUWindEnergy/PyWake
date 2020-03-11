from py_wake.deficit_models import DeficitModel
from numpy import newaxis as na


class NoWakeDeficit(DeficitModel):
    def calc_deficit(self, WS_ilk, dw_ijlk, **_):
        return (WS_ilk)[:, na] * (dw_ijlk > 0) * 0
