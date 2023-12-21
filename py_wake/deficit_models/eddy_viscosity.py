"""Module for the Eddy Viscosity (EV) model.

"""

import warnings
from pathlib import Path
from typing import Any, Final, Optional

import xarray as xr
from numpy import newaxis as na
from scipy.interpolate import RegularGridInterpolator

from py_wake import np
from py_wake.deficit_models import eddy_viscosity_lookup_table_generator
from py_wake.deficit_models.deficit_model import (
    DeficitModel,
    WakeDeficitModel,
)
from py_wake.deficit_models.eddy_viscosity_formulations import (
    SimplifiedEddyViscosityDeficitFormulation,
    SimplifiedEddyViscosityFormulationProvider,
)
from py_wake.deflection_models import DeflectionModel
from py_wake.ground_models.ground_models import GroundModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel
from py_wake.rotor_avg_models.simplified_gaussian_rotor_average_model import (
    SimplifiedGaussianRotorAverageModel,
)
from py_wake.site import XRSite
from py_wake.superposition_models import MaxSum, SuperpositionModel
from py_wake.turbulence_models.quarton_and_ainslie import (
    ModifiedQuartonAndAinslieTurbulenceModel,
)
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.wind_turbines import WindTurbines

DEFAULT_MAXIMUM_WAKE_DISTANCE: Final[float] = 50.0


class EddyViscosityNearWakeUserWarning(UserWarning):
    """Warning when using the EV model at distances below two rotors.

    The EV model does not have a defined solution for the near wake,
    within two rotor diameters of the source turbine.
    """

    pass


class EddyViscosityDeficitModel(WakeDeficitModel):
    """Eddy Viscosity (EV) wake deficit model.

    See documentation of ``simplified_eddy_viscosity_formulations`` and
    ``simplified_eddy_viscosity_lookup_table_generator`` for details.
    """

    def __init__(
        self,
        rotorAvgModel: Optional[RotorAvgModel] = SimplifiedGaussianRotorAverageModel(),
        groundModel: Optional[GroundModel] = None,
        use_effective_ws: bool = False,
        use_effective_ti: bool = True,
        use_mixing_function: bool = True,
        normalise_ti_to_waked_ws: bool = True,
        maximum_wake_distance: float = DEFAULT_MAXIMUM_WAKE_DISTANCE,
        formulation: Optional[SimplifiedEddyViscosityFormulationProvider] = (
            SimplifiedEddyViscosityDeficitFormulation()
        ),
        lookup_table_filepath: Optional[Path] = None,
    ) -> None:
        """Initiate an ``EddyViscosityDeficitModel``.

        :param rotorAvgModel: the rotor averaging model to use in the
            deficit calculations
        :param groundModel: the ground model to use in the deficit
            calculations
        :param use_effective_ws: whether to scale the wake wind speeds
            with the effective (wake-affected) incident wind speeds;
            this should be set to ``False`` when using the default
            maximum sum superposition model
        :param use_effective_ti: whether to use the effective (waked)
            incident turbulence intensity in the deficit calculations;
            this should be set to ``True`` when used with the default
            Quarton and Ainslie turbulence model
        :param use_mixing_function: whether to apply the mixing filter
            function in the near wake region
        :param normalise_ti_to_waked_ws: whether to normalise the
            turbulence intensity values to the incident waked wind speed
            values
        :param maximum_wake_distance: the maximum dimensionless wake
            distance (in source rotor diameters), beyond which wake
            effects are ignored (set to zero)
        :param formulation: the class providing the EV model formulation
            functions (initial value and derivative), which must
            implement the ``SimplifiedEddyViscosityFormulationProvider``
            protocol; this class is used by the lookup table generator
            if provided
        :param lookup_table_filepath: the path of the NetCDF file
            containing the lookup table data, which can optionally be
            used instead of providing a ``formulation``
        """
        DeficitModel.__init__(
            self,
            rotorAvgModel=rotorAvgModel,
            groundModel=groundModel,
            use_effective_ws=use_effective_ws,
            use_effective_ti=use_effective_ti,
        )
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti
        self.use_mixing_function = use_mixing_function
        self.normalise_ti_to_waked_ws = normalise_ti_to_waked_ws
        self.maximum_wake_distance = maximum_wake_distance

        lookup_table: xr.DataArray
        if lookup_table_filepath is None:
            if formulation is None:
                raise ValueError(
                    "either a formulation or a lookup table path must be provided"
                )
            lookup_table = eddy_viscosity_lookup_table_generator.generate_lookup_table(
                formulation=formulation,
                use_mixing_function=use_mixing_function,
            )
        else:
            lookup_table = xr.open_dataarray(lookup_table_filepath, engine="h5netcdf")

        # The lookup table interpolator can perhaps be replaced
        # by the standard PyWake lookup table solution
        self.interpolator = RegularGridInterpolator(
            points=(
                lookup_table.coords["ti0"],
                lookup_table.coords["ct"],
                lookup_table.coords["dw"],
            ),
            values=lookup_table.data,
            method="linear",
            bounds_error=True,
        )

    def calc_deficit(
        self,
        WS_ilk: np.ndarray,
        WS_eff_ilk: np.ndarray,
        TI_ilk: np.ndarray,
        TI_eff_ilk: np.ndarray,
        dw_ijlk: np.ndarray,
        cw_ijlk: np.ndarray,
        D_src_il: np.ndarray,
        ct_ilk: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        """Calculate the estimate of wake velocity deficit.

        :param WS_ilk: the free wind speed at the source turbine
        :param WS_eff_ilk: the waked wind speed at the source turbine
        :param TI_ilk: the ambient TI at the source turbine
        :param TI_eff_ilk: the waked TI at the source turbine
        :param dw_ijlk: the down-wind distance from the source to the
            destination turbine
        :param cw_ijlk: the cross-wind distance from the source to the
            destination turbine
        :param D_src_il: the diameter of the source turbine
        :param ct_ilk: the source turbine thrust coefficient
        :return: the estimated dimensional wind speed deficit due to
            wake effects (as an array by the ith source turbine, the jth
            destination turbine, in the lth wind direction sector and
            for the kth wind speed bin)
        """
        centre_frac_deficit_ijlk, wake_width_ijlk = self._calc_deficit_terms(
            WS_ilk=WS_ilk,
            WS_eff_ilk=WS_eff_ilk,
            TI_ilk=TI_ilk,
            TI_eff_ilk=TI_eff_ilk,
            dw_ijlk=dw_ijlk,
            D_src_il=D_src_il,
            ct_ilk=ct_ilk,
        )

        # Convert downwind and crosswind distance to dimensionless quantities
        # normalised by the source turbine rotor diameter
        dw_norm_ijlk = dw_ijlk / D_src_il[:, na, :, na]
        cw_norm_ijlk = cw_ijlk / D_src_il[:, na, :, na]

        fractional_deficit_ijlk = centre_frac_deficit_ijlk * np.exp(
            -3.56 * np.square(
                np.divide(
                    cw_norm_ijlk,
                    wake_width_ijlk,
                    where=~np.isclose(wake_width_ijlk, 0.0),
                    out=-999999.0 * np.ones_like(
                        centre_frac_deficit_ijlk * cw_norm_ijlk * wake_width_ijlk
                    ),
                )
            )
        )

        ws_ref_ilk: np.ndarray
        if self.use_effective_ws:
            ws_ref_ilk = WS_eff_ilk
        else:
            ws_ref_ilk = WS_ilk

        # Convert dimensionless deficit to dimensional deficit in 'm/s'
        deficit_ijlk = ws_ref_ilk[:, na] * fractional_deficit_ijlk

        # Limit wake impacts to the maximum wake distance
        deficit_ijlk = np.where(dw_norm_ijlk <= self.maximum_wake_distance, deficit_ijlk, 0.0)

        # Filter to compute deficit only for positive downstream distances
        deficit_ijlk = deficit_ijlk * np.logical_or(dw_norm_ijlk > 0.0, np.isclose(dw_norm_ijlk, 0.0))

        if np.any(np.logical_and(dw_norm_ijlk < 1.95, fractional_deficit_ijlk > 0.05)):
            warnings.warn(
                message=(
                    "The Eddy Viscosity wake model is not appropriate for turbine spacings "
                    "less than two rotor diameters; the solution at two rotor diameters was "
                    "used for smaller distances."
                ),
                category=EddyViscosityNearWakeUserWarning,
            )

        return deficit_ijlk

    def wake_radius(
        self,
        WS_ilk: np.ndarray,
        WS_eff_ilk: np.ndarray,
        TI_ilk: np.ndarray,
        TI_eff_ilk: np.ndarray,
        dw_ijlk: np.ndarray,
        D_src_il: np.ndarray,
        ct_ilk: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        """Calculate the dimensional estimate of the 'wake radius'.

        The ``wake_radius`` method is used for PyWake integration, so
        that the wake width parameter can be passed to the turbulence
        model. The EV model has a Gaussian profile in the radial
        direction and therefore not a clearly defined radius. The radius
        is here taken as the measure of the wake width defined by
        Ainslie (1988), which is an approximate measure of the full wake
        width.

        See also the documentation of ``calc_deficit``.
        """
        wake_width_ijlk = self._calc_deficit_terms(
            WS_ilk=WS_ilk,
            WS_eff_ilk=WS_eff_ilk,
            TI_ilk=TI_ilk,
            TI_eff_ilk=TI_eff_ilk,
            dw_ijlk=dw_ijlk,
            D_src_il=D_src_il,
            ct_ilk=ct_ilk,
        )[1]

        # Convert from dimensionless wake width to dimensional wake width in 'm'
        return wake_width_ijlk * D_src_il[:, na, :, na]

    def sigma_ijlk(self, **kwargs: Any) -> np.ndarray:
        """Calculate the dimensional sigma wake width parameter.

        This width parameter is used by the other Gaussian deficit models
        and by the rotor average models for Gaussian wake profiles.
        """
        return self.wake_radius(**kwargs) / np.sqrt(3.56)

    def _calc_deficit_terms(
        self,
        WS_ilk: np.ndarray,
        WS_eff_ilk: np.ndarray,
        TI_ilk: np.ndarray,
        TI_eff_ilk: np.ndarray,
        dw_ijlk: np.ndarray,
        D_src_il: np.ndarray,
        ct_ilk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.min(WS_ilk) < 0.0 or np.min(WS_eff_ilk) < 0.0:
            raise ValueError("Negative wind speed values are not valid.")

        if np.min(ct_ilk) < 0.0:
            raise ValueError("Negative thrust coefficient (Ct) values are not valid.")
        if np.max(ct_ilk) > 1.4:
            raise ValueError("Thrust coefficient (Ct) values higher than 1.4 are not supported.")

        ti0_ilk: np.ndarray
        if self.use_effective_ti:
            ti0_ilk = TI_eff_ilk
        else:
            ti0_ilk = TI_ilk

        if np.min(ti0_ilk) < 0.0:
            raise ValueError("Negative turbulence intensity values are not valid.")

        # Normalise the effective turbulence intensity to the waked wind speed
        if self.normalise_ti_to_waked_ws:
            ti0_ilk = ti0_ilk * (WS_ilk / WS_eff_ilk)

        # Filter the effective turbulence intensity for the calculations to be at most 50%,
        # in order to avoid artificially fast wake decay and wind speed recovery
        ti0_ilk = np.minimum(ti0_ilk, 0.5)

        # Convert downwind and crosswind distance to units of rotor diameters
        dw_norm_ijlk = dw_ijlk / D_src_il[:, na, :, na]

        # Filter the dimensionless downstream distance to the defined range
        dw_norm_ijlk = np.maximum(dw_norm_ijlk, 2.0)
        dw_norm_ijlk = np.minimum(dw_norm_ijlk, self.maximum_wake_distance)

        product_shape = np.ones_like(ti0_ilk[:, na, :, :] * ct_ilk[:, na, :, :] * dw_norm_ijlk)
        matched_ti = ti0_ilk[:, na, :, :] * product_shape
        matched_ct = ct_ilk[:, na, :, :] * product_shape
        matched_dw = dw_norm_ijlk * product_shape

        # Generate input of points for interpolator (ijlk for 'ti0', 'ct' and 'dw')
        interpolator_input = np.stack((matched_ti, matched_ct, matched_dw), axis=-1)
        interpolator_input_shape = interpolator_input.shape
        flat_dim = int(np.prod(interpolator_input_shape[:-1]))
        interpolator_input = interpolator_input.reshape((flat_dim, 3), order="C")

        # Interpolate dimensionless centreline velocity deficit
        centre_frac_deficit = self.interpolator(interpolator_input)
        centre_frac_deficit_ijlk = centre_frac_deficit.reshape(interpolator_input_shape[:-1], order="C")

        wake_width_ijlk = np.sqrt(
            np.divide(
                3.56 * matched_ct,
                4.0 * centre_frac_deficit_ijlk * (2.0 - centre_frac_deficit_ijlk),
                where=~np.isclose(centre_frac_deficit_ijlk, 0.0),
                out=np.zeros_like(product_shape),
            )
        )

        return centre_frac_deficit_ijlk, wake_width_ijlk


class EddyViscosityModel(PropagateDownwind):
    """Pre-defined wind farm model based on the EV deficit model."""

    def __init__(
        self,
        site: XRSite,
        windTurbines: WindTurbines,
        superpositionModel: SuperpositionModel = MaxSum(),
        deflectionModel: DeflectionModel = None,
        turbulenceModel: TurbulenceModel = ModifiedQuartonAndAinslieTurbulenceModel(),
    ) -> None:
        """Initiate an ``EddyViscosityModel``.

        The superposition model should generally be ``MaxSum`` if using
        the EV model in the way it is typically implemented. This
        requires that the wind speeds in the deficit model are scaled by
        the free incident wind speeds rather than the effective (waked)
        wind speeds.

        Note that the wake deficits are scaled by the source turbine
        wind speeds and do not account for any speed-up (or speed-down)
        effects from the source turbines to the target turbines. For
        example if a wake velocity deficit of 1.0 m/s is calculated from
        a turbine with a free wind speed of 10.0 m/s to a turbine with a
        free wind speed 8.0 m/s, the applied deficit is still 1.0 m/s
        and not scaled by the wind speed reduction. In extreme cases,
        this can lead to negative wind speed predictions, which will
        trigger the calculations to exit with an error.

        The EV model has not been tested in conjunction with a
        deflection model.

        The modified version of the Quarton and Ainslie turbulence model
        is used as the default, as that is more widely adopted. The EV
        deficit model is generally used in conjunction with some flavour
        of the Quarton and Ainslie turbulence model, in original or
        modified form.

        :param site: the site object to create the model for
        :param windTurbines: the wind turbines object
        :param superpositionModel: the wake deficit superposition model
            to use for combining the wake impacts from multiple turbines
        :param deflectionModel: the wake deficit deflection model to use
        :param turbulenceModel: the wake added turbulence model to use
        """
        super().__init__(
            site=site,
            windTurbines=windTurbines,
            wake_deficitModel=EddyViscosityDeficitModel(),
            superpositionModel=superpositionModel,
            deflectionModel=deflectionModel,
            turbulenceModel=turbulenceModel,
        )
