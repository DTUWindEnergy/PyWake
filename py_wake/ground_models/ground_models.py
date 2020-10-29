import numpy as np
from numpy import newaxis as na


class GroundModel():
    pass


class NoGround(GroundModel):
    args4deficit = []

    def __call__(self, calc_deficit, **kwargs):
        return calc_deficit(**kwargs)


class Mirror(GroundModel):
    """The WindFarmModel-to-GroundModel API will most likely if a new model is added, while the
    User-to-WindFarmModel API wrt. GroundModel will hopefully persist
    """
    args4deficit = ['dh_ijl', 'h_il', 'hcw_ijlk']

    def __call__(self, calc_deficit, **kwargs):
        dh_ijl_mirror = 2 * kwargs['h_il'][:, na] + kwargs['dh_ijl']
        cw_ijlk_mirror = None
        if 'cw_ijlk' in kwargs:
            cw_ijlk_mirror = np.sqrt(dh_ijl_mirror[..., na]**2, kwargs['hcw_ijlk']**2)
        deficit_ijlk = np.sum([calc_deficit(**kwargs),
                               calc_deficit(dh_ijl=dh_ijl_mirror,
                                            cw_ijlk=cw_ijlk_mirror,
                                            **{k: v for k, v in kwargs.items() if k not in ['dh_ijl', 'cw_ijlk']})], 0)
        return deficit_ijlk * ((kwargs['h_il'][:, na] + kwargs['dh_ijl'])[..., na] > 0)  # remove deficit below ground
