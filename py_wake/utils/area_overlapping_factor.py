from numpy import newaxis as na

from py_wake import np
from py_wake.utils.gradients import cabs
from autograd.numpy.numpy_boxes import ArrayBox
from py_wake.utils import gradients


class AreaOverlappingFactor():

    def overlapping_area_factor(self, wake_radius_ijlk, dw_ijlk, cw_ijlk, D_src_il, D_dst_ijl):
        """Calculate overlapping factor

        Parameters
        ----------
        dw_jl : array_like
            down wind distance [m]
        cw_jl : array_like
            cross wind distance [m]
        D_src_l : array_like
            Diameter of source turbines [m]
        D_dst_jl : array_like or None
            Diameter of destination turbines [m]. If None destination is assumed to be a point

        Returns
        -------
        A_ol_factor_jl : array_like
            area overlaping factor
        """

        if np.all(D_dst_ijl == 0) or D_dst_ijl is None:
            return wake_radius_ijlk > cw_ijlk
        else:
            if wake_radius_ijlk.ndim == 5:
                shape = np.maximum(cw_ijlk.shape, wake_radius_ijlk.shape)
                return self._cal_overlapping_area_factor(
                    np.broadcast_to(wake_radius_ijlk, shape),
                    np.broadcast_to(D_dst_ijl[..., na, na] / 2, shape),
                    cabs(cw_ijlk))
            else:
                return self._cal_overlapping_area_factor(wake_radius_ijlk,
                                                         (D_dst_ijl[..., na] / 2),
                                                         cabs(cw_ijlk))

    def _cal_overlapping_area_factor(self, R1, R2, d):
        """ Calculate the overlapping area of two circles with radius R1 and
        R2, centers distanced d.

        The calculation formula can be found in Eq. (A1) of :
        [Ref] Feng J, Shen WZ, Solving the wind farm layout optimization
        problem using Random search algorithm, Renewable Energy 78 (2015)
        182-192
        Note that however there are typos in Equation (A1), '2' before alpha
        and beta should be 1.

        Parameters
        ----------
        R1: array:float
            Radius of the first circle [m]

        R2: array:float
            Radius of the second circle [m]

        d: array:float
            Distance between two centers [m]

        Returns
        -------
        A_ol: array:float
            Overlapping area [m^2]
        """
        # treat all input as array
        shape = tuple(np.max([R1.shape, R2.shape, d.shape], 0))
        R1, R2, d = [np.broadcast_to(a, shape) for a in [R1, R2, d]]

        # make sure R_big >= R_small
        Rmax = np.where(R1 < R2, R2, R1)
        Rmin = np.where(R1 < R2, R1, R2)

        # full wake cases
        index_fullwake = (d <= (Rmax - Rmin))
        dtype = (float, np.complex128)[bool(np.any([np.iscomplexobj(x) for x in [R1, R2, d]]))]
        A_ol_f = np.where(index_fullwake, 1, 0).astype(dtype)

        # partial wake cases
        mask = (d > (Rmax - Rmin)) & (d < (Rmin + Rmax))
        if any([isinstance(x, ArrayBox) for x in [R1, R2, d]]):
            p_wake_mask = mask
            mask = slice(None)
        else:
            p_wake_mask = None

        # in somecases cos_alpha or cos_beta can be larger than 1 or less than
        # -1.0, cause problem to arccos(), resulting nan values, here fix this
        # issue.
        eps = 2 * np.finfo(float).eps

        def arccos_lim(x):
            return np.arccos((np.clip(x, -1.0 + eps, +1.0 - eps)))  # eps to avoid inf in gradient

        alpha = arccos_lim((Rmax[mask]**2.0 + d[mask]**2 - Rmin[mask]**2) /
                           (2.0 * np.maximum(Rmax[mask] * d[mask], eps)))

        beta = arccos_lim((Rmin[mask]**2.0 + d[mask]**2 - Rmax[mask]**2) /
                          (2.0 * np.maximum(Rmin[mask] * d[mask], eps)))

        # p = (R1[mask] + R2[mask] + d[mask]) / 2.0
        # A_triangle = 2 * np.sqrt(gradients.cabs(p * (p - Rmin[mask]) *
        #                                         (p - Rmax[mask]) * (p - d[mask])))
        A_triangle = np.sin(alpha) * R1[mask] * d[mask]

        p_wake_f = (alpha * Rmax[mask]**2 + beta * Rmin[mask]**2 -
                    A_triangle) / (R2[mask]**2 * np.pi)

        if p_wake_mask is None:
            A_ol_f[mask] = p_wake_f
        else:
            # autograd
            A_ol_f = np.where(p_wake_mask, p_wake_f, A_ol_f)
        return A_ol_f
