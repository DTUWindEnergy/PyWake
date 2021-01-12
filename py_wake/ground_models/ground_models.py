import numpy as np
from numpy import newaxis as na


class GroundModel():
    def __init__(self):
        pass


class NoGround(GroundModel):
    args4deficit = []

    def __call__(self, calc_deficit, **kwargs):
        return calc_deficit(**kwargs)


class Mirror(GroundModel):
    """The WindFarmModel-to-GroundModel API will most likely if a new model is added, while the
    User-to-WindFarmModel API wrt. GroundModel will hopefully persist
    """
    args4deficit = ['dh_ijlk', 'h_il', 'hcw_ijlk']

    def _calc(self, calc_deficit, **kwargs):
        dh_ijlk_mirror = 2 * kwargs['h_il'][:, na, :, na] + kwargs['dh_ijlk']
        cw_ijlk_mirror = None
        if 'cw_ijlk' in kwargs:
            cw_ijlk_mirror = np.sqrt(dh_ijlk_mirror**2 + kwargs['hcw_ijlk']**2)
        above_ground = ((kwargs['h_il'][:, na, :, na] + kwargs['dh_ijlk']) > 0)
        return np.array([calc_deficit(**kwargs),
                         calc_deficit(dh_ijlk=dh_ijlk_mirror,
                                      cw_ijlk=cw_ijlk_mirror,
                                      **{k: v for k, v in kwargs.items() if k not in ['dh_ijlk', 'cw_ijlk']})]) * above_ground[na]

    def __call__(self, calc_deficit, **kwargs):
        return np.sum(self._calc(calc_deficit, **kwargs), 0)


class MirrorSquaredSum(Mirror):
    def __call__(self, calc_deficit, **kwargs):
        return np.sqrt(np.sum(self._calc(calc_deficit, **kwargs)**2, 0))
