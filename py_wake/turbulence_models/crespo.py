from numpy import newaxis as na
import numpy as np
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.utils.area_overlapping_factor import AreaOverlappingFactor
from py_wake.superposition_models import SqrMaxSum


class CrespoHernandez(TurbulenceModel, AreaOverlappingFactor):
    """
    Implemented according to:
    A. Crespo and J. Hernández
    Turbulence characteristics in wind-turbine wakes
    J. of Wind Eng. and Industrial Aero. 61 (1996) 71-85

    """
    args4addturb = ['dw_ijlk', 'cw_ijlk', 'D_src_il', 'ct_ilk', 'TI_ilk', 'D_dst_ijl', 'wake_radius_ijlk']

    def __init__(self, addedTurbulenceSuperpositionModel=SqrMaxSum(), **kwargs):
        TurbulenceModel.__init__(self, addedTurbulenceSuperpositionModel, **kwargs)

    def calc_added_turbulence(self, dw_ijlk, cw_ijlk, D_src_il, ct_ilk, TI_ilk, D_dst_ijl, wake_radius_ijlk, **_):
        """ Calculate the added turbulence intensity at locations specified by
        downstream distances (dw_jl) and crosswind distances (cw_jl)
        caused by the wake of a turbine (diameter: D_src_l, thrust coefficient: Ct_lk).

        Returns
        -------
        TI_add_ijlk: array:float
            Added turbulence intensity weighted by wake-turbine overlap [-]
        """
        # induction factor
        a_ilk = 0.5 * (1 - np.sqrt(1 - ct_ilk))
        # added turbulence (Eq. 21)
        dw_ijlk_gt0 = np.maximum(dw_ijlk, 1e-10)  # avoid divide by zero and sqrt of negative number
        TI_add_ijlk = 0.73 * a_ilk[:, na, :, :]**0.8325 * TI_ilk[:, na, :, :]**0.0325 * \
            np.abs(D_src_il[:, na, :, na] / dw_ijlk_gt0)**(0.32) * (dw_ijlk > 0)

        area_overlap_ijlk = self.overlapping_area_factor(wake_radius_ijlk, dw_ijlk, cw_ijlk, D_src_il, D_dst_ijl)

        return TI_add_ijlk * area_overlap_ijlk * (dw_ijlk > 0)  # ensure zero upstream
