"""Added turbulence models based on the work of Quarton and Ainslie.

References:

    Quarton, D C and Ainslie, J F. “Turbulence in Wind Turbine Wakes,”
    Wind Engineering, Vol. 14 No. 1, 1990.

    Hassan, U. “A Wind Tunnel Investigation of the Wake Structure within
    Small Wind Turbine Farms”, Department of Energy, E/5A/CON/5113/1890,
    1992.

    Vermeulen, P E J. "An experimental analysis of wind turbine wakes,"
    Proc. 3rd International Symposium of Wind Energy Systems,
    Copenhagen, August 1980.

    Vermeulen, P E J and Builtjes, P. “Mathematical Modelling of Wake
    Interaction in Wind Turbine Arrays, Part 1,” report TNO 81-01473,
    1981.

    Vermeulen, P E J and Vijge, J. “Mathematical Modelling of Wake
    Interaction in Wind Turbine Arrays, Part 2,” report TNO 81-02834,
    1981.

"""

from typing import Any, Optional

from numpy import newaxis as na
from py_wake import np
from py_wake.ground_models import GroundModel
from py_wake.rotor_avg_models import RotorAvgModel, RotorCenter
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.superposition_models import AddedTurbulenceSuperpositionModel, SqrMaxSum
from py_wake.turbulence_models.turbulence_model import TurbulenceModel


class QuartonAndAinslieTurbulenceModel(TurbulenceModel):
    """The model described in the paper by Quarton and Ainslie (1990).

    The added turbulence intensity in the wake is described as an
    empirical function of the turbine thrust coefficient, the ambient
    turbulence intensity and the downstream distance normalised by the
    length of the near wake. The estimate of the near wake length is
    based upon an empirical calculation method proposed by Vermeulen and
    colleagues (see references above).

    The paper (Quarton and Ainslie, 1990) does not propose a method of
    predicting the variation in wake added turbulence intensity as a
    function of radial (cross-wind) distance from the wake centreline.
    In the implementation here, it is simply assumed the turbulence
    inside the wake is constant, and for a rotor partially inside the
    wake the result is weighted by the proportion of the area insider
    the wake radius. For Gaussian wake profiles that do not have a clear
    border, the radius should define where the conditions are
    substantially as in the freestream.

    For the modified version of the model later proposed by Hassan, see
    ``ModifiedQuartonAndAinslieTurbulenceModel``.
    """

    def __init__(
        self,
        addedTurbulenceSuperpositionModel: AddedTurbulenceSuperpositionModel = SqrMaxSum(),
        rotorAvgModel: Optional[RotorAvgModel] = RotorCenter(),
        groundModel: Optional[GroundModel] = None,
        use_effective_ws: bool = True,
        use_effective_ti: bool = False,
    ) -> None:
        """Initiate a ``QuartonAndAinslieTurbulenceModel``.

        :param addedTurbulenceSuperpositionModel: the superposition
            model to use for combining the added wake turbulence from
            multiple source turbines
        :param rotorAvgModel: the rotor averaging model to use in the
            added turbulence calculations
        :param groundModel: the ground model to use in the added
            turbulence calculations
        :param use_effective_ws: whether to use the effective
            (wake-affected) incident wind speeds in the calculations of
            the tip speed ratio for the near wake length estimation
        :param use_effective_ti: whether to use the effective incident
            turbulence intensity in the calculations
        """
        TurbulenceModel.__init__(
            self,
            addedTurbulenceSuperpositionModel=addedTurbulenceSuperpositionModel,
            rotorAvgModel=rotorAvgModel,
            groundModel=groundModel,
        )
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti
        self.area_overlap_avg_model = AreaOverlapAvgModel()
        self.add_turbulence_factor = 4.8
        self.add_turbulence_decay_exponent = -0.57

    def calc_added_turbulence(
        self,
        WS_ilk: np.ndarray,
        WS_eff_ilk: np.ndarray,
        TI_ilk: np.ndarray,
        TI_eff_ilk: np.ndarray,
        dw_ijlk: np.ndarray,
        cw_ijlk: np.ndarray,
        D_src_il: np.ndarray,
        D_dst_ijl: np.ndarray,
        wake_radius_ijlk: np.ndarray,
        ct_ilk: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        """Calculate the estimate of wake added turbulence intensity.

        Note that the wake added turbulence intensity calculated by this
        model provides an estimate of additional turbulence intensity
        relative to the freestream wind speed. For values relative to
        the reduced wind speed in the wake, the results need to be
        re-normalised relative to the waked wind speed.

        :param WS_ilk: the free wind speed at the source turbine
        :param WS_eff_ilk: the waked wind speed at the source turbine
        :param TI_ilk: the ambient TI at the source turbine
        :param TI_eff_ilk: the waked TI at the source turbine
        :param dw_ijlk: the down-wind distance from the source to the
            destination turbine
        :param cw_ijlk: the cross-wind distance from the source to the
            destination turbine
        :param D_src_il: the diameter of the source turbine
        :param D_dst_ijl: the diameter of the destination turbine
        :param wake_radius_ijlk: the radius of the wake from the source
            turbine at the location of the destination turbine
        :param ct_ilk: the source turbine thrust coefficient
        :return: the estimated added dimensionless turbulence intensity
            due to wake effects (as an array by the ith source turbine,
            the jth destination turbine, in the lth wind direction
            sector and for the kth wind speed bin)
        """
        ws_ref_ilk: np.ndarray
        if self.use_effective_ws:
            ws_ref_ilk = WS_eff_ilk
        else:
            ws_ref_ilk = WS_ilk

        if np.min(ws_ref_ilk) < 0.0:
            raise ValueError("Negative wind speed values are not valid.")

        ti_ref_ilk: np.ndarray
        if self.use_effective_ti:
            ti_ref_ilk = TI_eff_ilk
        else:
            ti_ref_ilk = TI_ilk

        if np.min(ti_ref_ilk) < 0.0:
            raise ValueError("Negative turbulence intensity values are not valid.")

        near_wake_length_ilk = self.calc_near_wake_length_ilk(
            ws_ref_ilk=ws_ref_ilk,
            ti_ref_ilk=ti_ref_ilk,
            D_src_il=D_src_il,
            ct_ilk=ct_ilk,
        )
        dw_norm_xn_ijlk = dw_ijlk / near_wake_length_ilk[:, na, :, :]
        dw_norm_xn_ijlk = np.where(dw_norm_xn_ijlk > 0.1, dw_norm_xn_ijlk, 0.1)

        added_ti_ijlk = (
            self.add_turbulence_factor * np.power(ct_ilk[:, na, :, :], 0.7) *
            np.power(ti_ref_ilk[:, na, :, :] * 100.0, 0.68) *
            np.power(dw_norm_xn_ijlk, self.add_turbulence_decay_exponent) * 0.01
        )

        # Use "built-in" adjustment for cross-wind fraction only if the
        # ``rotorAvgModel`` is ``None`` or a ``RotorCenter`` instance
        if self.rotorAvgModel is None or isinstance(self.rotorAvgModel, RotorCenter):
            crosswind_fraction_ijlk = self._calc_crosswind_fraction_ijlk(
                cw_ijlk=cw_ijlk,
                D_dst_ijl=D_dst_ijl,
                wake_radius_ijlk=wake_radius_ijlk,
            )
            added_ti_ijlk = (
                added_ti_ijlk * crosswind_fraction_ijlk
            )

        # Ensure added turbulence only downstream
        added_ti_ijlk = added_ti_ijlk * (dw_ijlk > 0.0)

        # Ensure added turbulence is positive
        added_ti_ijlk = np.maximum(added_ti_ijlk, 0.0)

        return added_ti_ijlk

    def _calc_crosswind_fraction_ijlk(
        self,
        cw_ijlk: np.ndarray,
        D_dst_ijl: np.ndarray,
        wake_radius_ijlk: np.ndarray,
    ) -> np.ndarray:
        """Calculate the fractional weight due to crosswind distance.

        The weight is taken simply as the square root of the fraction of
        the destination turbine rotor area that is inside the bounds of
        the wake radius. The square root is used due to the root sum
        square combination in the superposition model.

        :param cw_ijlk: the cross wind distance from the source to the
            destination turbine
        :param D_dst_ijl: the diameter of the destination turbine
        :param wake_radius_ijlk: the radius of the wake from the source
            turbine at the location of the destination turbine
        :return: the weight to apply to account for crosswind distance
            (as an array by the ith source turbine, the jth destination
            turbine, in the lth wind direction sector and for the kth
            wind speed bin)
        """
        # This makes use of the existing PyWake function for calculating
        # area overlap. It would be better to use the RotorAvgModel
        # directly, but is currently not possible as it requires a
        # WakeRadiusTopHat to work. This should be possible to solve and
        # can replace this temporary solution in due course.
        overlapping_area_factor = self.area_overlap_avg_model.overlapping_area_factor(
            wake_radius_ijlk=wake_radius_ijlk,
            cw_ijlk=cw_ijlk,
            D_dst_ijl=D_dst_ijl,
        )
        overlapping_area_factor = np.maximum(overlapping_area_factor, 0.0)
        overlapping_area_factor = np.minimum(overlapping_area_factor, 1.0)
        return np.sqrt(overlapping_area_factor)

    @staticmethod
    def calc_near_wake_length_ilk(
        ws_ref_ilk: np.ndarray,
        ti_ref_ilk: np.ndarray,
        D_src_il: np.ndarray,
        ct_ilk: np.ndarray,
        rotor_speed_ilk: np.ndarray = np.array([[[20.0]]]),
    ) -> np.ndarray:
        """Calculate the estimate of the dimensional near wake length.

        This estimate is based on the work of Vermeulen and colleagues,
        which Quarton and Ainslie (1990) refers to. See full references
        in the documentation of ``QuartonAndAinslieTurbulenceModel``.

        As a temporary solution, the rotor speed is set to a default
        value of 20 rpm if not included in the argument.

        :param ws_ref_ilk: the relevant wind speed variable (free or
            waked) at the source turbine
        :param ti_ref_ilk: the waked TI at the source turbine
        :param D_src_il: the diameter of the source turbine
        :param ct_ilk: the source turbine thrust coefficient
        :param rotor_speed_ilk: the source turbine rotor speed
        :return: the estimated length of the near wake of the source
            turbine (as an array by the ith turbine, in the lth wind
            direction sector and for the kth wind speed bin)
        """
        # Limit Ct value for the proceeding calculations to avoid
        # negative values in square roots and division by zero
        ct_limited_ilk = np.minimum(ct_ilk, 0.96)

        m_ilk = 1.0 / np.sqrt(1.0 - ct_limited_ilk)

        r0_ilk = (D_src_il[:, :, na] / 2.0) * np.sqrt((m_ilk + 1.0) / 2.0)

        n_term1_ilk = np.sqrt(0.214 + 0.144 * m_ilk)
        n_term2_ilk = np.sqrt(0.134 + 0.124 * m_ilk)
        n_ilk = n_term1_ilk * (1.0 - n_term2_ilk) / (n_term2_ilk * (1.0 - n_term1_ilk))

        # Wake growth contribution from ambient turbulence
        drdx_alpha_ilk = 2.5 * ti_ref_ilk + 0.005

        # Wake growth contribution from turbulence generated by shear
        drdx_m_ilk = (1.0 - m_ilk) * np.sqrt(1.49 + m_ilk) / (9.76 * (1.0 + m_ilk))

        # Wake growth contribution from mechanical turbulence
        ws_ref_limited_ilk = np.maximum(ws_ref_ilk, 1.0)
        tip_speed_ratio_ilk = (
            np.pi * rotor_speed_ilk * D_src_il[:, :, na] / (60.0 * ws_ref_limited_ilk)
        )
        drdx_lambda_ilk = 0.012 * 3 * tip_speed_ratio_ilk

        # Total wake growth rate
        drdx_ilk = np.sqrt(
            np.power(drdx_alpha_ilk, 2.0) + np.power(drdx_m_ilk, 2.0) +
            np.power(drdx_lambda_ilk, 2.0)
        )

        return n_ilk * r0_ilk / drdx_ilk


class ModifiedQuartonAndAinslieTurbulenceModel(QuartonAndAinslieTurbulenceModel):
    """The modified version of the model proposed by Hassan (1992).

    This model is the same as the original model developed by Quarton
    and Ainslie (1990), except for adjusting the factor and stream-wise
    decay exponent in the relationship between added turbulence and
    downstream distance.

    See also the documentation for ``QuartonAndAinslieTurbulenceModel``.
    """

    def __init__(
        self,
        addedTurbulenceSuperpositionModel: AddedTurbulenceSuperpositionModel = SqrMaxSum(),
        rotorAvgModel: Optional[RotorAvgModel] = RotorCenter(),
        groundModel: Optional[GroundModel] = None,
        use_effective_ws: bool = True,
        use_effective_ti: bool = False,
    ) -> None:
        """Initiate a ``ModifiedQuartonAndAinslieTurbulenceModel``.

        See documentation of ``QuartonAndAinslieTurbulenceModel`` for
        details.
        """
        super().__init__(
            addedTurbulenceSuperpositionModel=addedTurbulenceSuperpositionModel,
            rotorAvgModel=rotorAvgModel,
            groundModel=groundModel,
            use_effective_ws=use_effective_ws,
            use_effective_ti=use_effective_ti,
        )
        self.add_turbulence_factor = 5.7
        self.add_turbulence_decay_exponent = -0.96
