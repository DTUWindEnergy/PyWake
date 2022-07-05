from py_wake import np
from numpy import newaxis as na


class GroundModel():
    def __init__(self):
        pass

    @property
    def windFarmModel(self):
        return self.deficitModel.windFarmModel

    def _calc_layout_terms(self, deficitModel, **kwargs):
        self.windFarmModel.rotorAvgModel._calc_layout_terms(deficitModel, **kwargs)


class NoGround(GroundModel):
    args4deficit = []

    def __call__(self, calc_deficit, **kwargs):
        return calc_deficit(**kwargs)


class Mirror(GroundModel):
    """Consider the ground as a mirror (modeled by adding underground wind turbines).
    The deficits caused by the above- and below-ground turbines are summed
    by the superpositionModel of the windFarmModel
    """
    args4deficit = ['dh_ijlk', 'h_il', 'hcw_ijlk', 'IJLK']

    def _update_kwargs(self, **kwargs):
        def add_mirror_wt(k, v):
            if np.shape(v)[0] > 1 or '_ijlk' in k:
                return np.concatenate([v, v], 0)
            else:
                return v

        new_kwargs = {k: add_mirror_wt(k, v) for k, v in kwargs.items()}
        new_kwargs['dh_ijlk'] = np.concatenate([kwargs['dh_ijlk'],
                                                kwargs['dh_ijlk'] + (2 * kwargs['h_il'][:, na, :, na])],
                                               0)
        if 'cw_ijlk' in kwargs:
            new_kwargs['cw_ijlk'] = np.sqrt(new_kwargs['dh_ijlk']**2 + new_kwargs['hcw_ijlk']**2)
        return new_kwargs

    def __call__(self, calc_deficit, **kwargs):
        new_kwargs = self._update_kwargs(**kwargs)
        above_ground = ((new_kwargs['h_il'][:, na, :, na] + new_kwargs['dh_ijlk']) > 0)
        deficit_mijlk = np.reshape(calc_deficit(**new_kwargs) * above_ground, (2,) + kwargs['IJLK'])
        return self.windFarmModel.superpositionModel(deficit_mijlk)

    def _calc_layout_terms(self, deficitModel, **kwargs):
        new_kwargs = self._update_kwargs(**kwargs)
        self.windFarmModel.rotorAvgModel._calc_layout_terms(deficitModel, **new_kwargs)
