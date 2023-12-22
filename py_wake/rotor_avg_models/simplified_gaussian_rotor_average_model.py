"""Module for the simplified Gaussian rotor average model.

"""

from typing import Any, Callable, Final, NamedTuple

import xarray as xr
from numpy import newaxis as na
from scipy.interpolate import RegularGridInterpolator

from py_wake import np
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel


class LookupTableCoordinates(NamedTuple):
    """Named tuple of lookup table coordinates for the rotor averaging."""

    # Dimensionless transverse (combined horizontal and vertical cross-
    # wind) offset of the destination rotor centre relative to the wake
    # centreline, normalised by the destination turbine rotor diameter
    dimensionless_offset: np.ndarray

    # Dimensionless width parameter of the Gaussian wake profile,
    # normalised by the destination turbine rotor diameter
    dimensionless_sigma: np.ndarray


DEFAULT_LOOKUP_TABLE_COORDINATES: Final[LookupTableCoordinates] = LookupTableCoordinates(
    dimensionless_offset=np.arange(0.0, 75.0, 0.1),
    dimensionless_sigma=np.arange(0.0, 50.0, 0.1),
)


class SimplifiedGaussianRotorAverageModel(RotorAvgModel):
    """A simplified rotor average model for a Gaussian wake profile.

    This model calculates the rotor average as the mean value along the
    line through the rotor diameter that is aligned with the direction
    of the wake centre. It is included to produce results comparable to
    other tools such as WindFarmer and Openwind. The use of a lookup
    table avoids the need to include multiple points across the rotor in
    the computations.
    """

    def __init__(self, coordinates: LookupTableCoordinates = DEFAULT_LOOKUP_TABLE_COORDINATES) -> None:
        self._lookup_table = self._generate_lookup_table(coordinates=coordinates)
        self._interpolator = RegularGridInterpolator(
            points=(
                self._lookup_table.coords["dimensionless_offset"],
                self._lookup_table.coords["dimensionless_sigma"],
            ),
            values=self._lookup_table.data,
            method="linear",
            bounds_error=True,
        )

    def _calc_layout_terms(
        self,
        func: Callable[..., Any],
        hcw_ijlk: np.ndarray,
        dh_ijlk: np.ndarray,
        cw_ijlk: np.ndarray,
        **kwargs: Any,
    ) -> None:
        func(hcw_ijlk=hcw_ijlk * 0.0, dh_ijlk=dh_ijlk * 0.0, cw_ijlk=cw_ijlk * 0.0, **kwargs)

    def __call__(
        self,
        func: Callable[..., np.ndarray],
        hcw_ijlk: np.ndarray,
        dh_ijlk: np.ndarray,
        cw_ijlk: np.ndarray,
        D_dst_ijl: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        res_ijlk = func(hcw_ijlk=hcw_ijlk * 0.0, dh_ijlk=dh_ijlk * 0.0, cw_ijlk=cw_ijlk * 0.0, **kwargs)

        if not hasattr(func.__self__, "sigma_ijlk"):  # type: ignore
            raise AttributeError(
                f"'{func.__self__.__class__.__name__}' has no attribute 'sigma_ijlk', "  # type: ignore
                f"which is needed by the 'SimplifiedGaussianRotorAverageModel'"
            )
        sigma_ijlk = func.__self__.sigma_ijlk(**kwargs)  # type: ignore

        rotor_average_factor_ijlk = self._interpolate_rotor_average_factor(
            cw_ijlk=cw_ijlk,
            D_dst_ijl=D_dst_ijl,
            sigma_ijlk=sigma_ijlk,
        )

        return res_ijlk * rotor_average_factor_ijlk

    def _interpolate_rotor_average_factor(
        self,
        cw_ijlk: np.ndarray,
        D_dst_ijl: np.ndarray,
        sigma_ijlk: np.ndarray,
    ) -> np.ndarray:
        d_dst_ijlk = D_dst_ijl[:, :, :, na]

        dimensionless_offset = np.minimum(
            cw_ijlk / d_dst_ijlk,
            self._lookup_table.coords["dimensionless_offset"].values.max(),
        )
        dimensionless_sigma = np.minimum(
            sigma_ijlk / d_dst_ijlk,
            self._lookup_table.coords["dimensionless_sigma"].values.max(),
        )

        product_shape = np.ones_like(dimensionless_offset * dimensionless_sigma)
        interpolator_input = np.stack(
            (dimensionless_offset * product_shape, dimensionless_sigma * product_shape), axis=-1
        )
        interpolator_input_shape = interpolator_input.shape
        flat_dim = int(np.prod(interpolator_input_shape[:-1]))
        interpolator_input = interpolator_input.reshape((flat_dim, 2), order="C")

        rotor_average_factor = self._interpolator(interpolator_input)
        return rotor_average_factor.reshape(interpolator_input_shape[:-1], order="C")

    @staticmethod
    def _generate_lookup_table(coordinates: LookupTableCoordinates) -> xr.DataArray:
        delta = coordinates.dimensionless_offset[:, na, na]
        sigma = coordinates.dimensionless_sigma[na, :, na]
        r = np.arange(-0.495, 0.5, 0.01)[na, na, :]

        mean_fractional_deficit = np.exp(
            -1.0 * np.square(
                np.divide(
                    delta + r,
                    sigma,
                    where=~np.isclose(sigma, 0.0),
                    out=-999999.0 * np.ones_like(delta * sigma * r),
                )
            )
        ).mean(axis=-1)

        return xr.DataArray(
            data=mean_fractional_deficit,
            dims=("dimensionless_offset", "dimensionless_sigma"),
            coords={
                "dimensionless_offset": coordinates.dimensionless_offset,
                "dimensionless_sigma": coordinates.dimensionless_sigma,
            },
        )
