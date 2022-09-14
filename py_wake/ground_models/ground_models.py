from py_wake import np
from numpy import newaxis as na
from py_wake.utils.model_utils import ModelMethodWrapper, Model


class GroundModel(Model, ModelMethodWrapper):
    """"""


class NoGround(GroundModel):
    """ Using this model corresponds to groundModel=None, but it can be used to override the groundModel
     specified for the windFarmModel in e.g. the turbulence model"""

    def __call__(self, func, **kwargs):
        return func(**kwargs)


class Mirror(GroundModel):
    """Consider the ground as a mirror (modeled by adding underground wind turbines).
    The deficits caused by the above- and below-ground turbines are summed
    by the superpositionModel of the windFarmModel
    """

    def _update_kwargs(self, **kwargs):
        def add_mirror_wt(k, v):
            if (np.shape(v)[0] > 1 or '_ijlk' in k) and '_jlk' not in k:
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

    def __call__(self, func, h_il, dh_ijlk, IJLK, **kwargs):
        new_kwargs = self._update_kwargs(h_il=h_il, dh_ijlk=dh_ijlk, IJLK=IJLK, **kwargs)
        above_ground = ((new_kwargs['h_il'][:, na, :, na] + new_kwargs['dh_ijlk']) > 0)
        values_pijlk = func(**new_kwargs)
        deficit_mijlk = np.reshape(values_pijlk * above_ground, (2,) + IJLK)
        return self.windFarmModel.superpositionModel(deficit_mijlk)

    def _calc_layout_terms(self, func, **kwargs):
        new_kwargs = self._update_kwargs(**kwargs)
        func(**new_kwargs)
