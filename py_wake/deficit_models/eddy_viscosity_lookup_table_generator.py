"""Solver and lookup table generator for the simplified EV model.

"""

from pathlib import Path
from typing import Final, NamedTuple, Optional

import xarray as xr
from numpy import newaxis as na
from py_wake import np
from py_wake.deficit_models.eddy_viscosity_formulations import (
    SimplifiedEddyViscosityDeficitFormulation,
    SimplifiedEddyViscosityFormulationProvider,
)
from scipy.integrate import solve_ivp


class LookupTableCoordinates(NamedTuple):
    """Named tuple of EV lookup table coordinates."""

    ti0: np.ndarray
    ct: np.ndarray
    dw: np.ndarray


DEFAULT_LOOKUP_TABLE_COORDINATES: Final[
    LookupTableCoordinates
] = LookupTableCoordinates(
    ti0=np.arange(0.0, 0.51, 0.01),
    ct=np.arange(0.0, 1.42, 0.02),
    dw=np.exp(np.arange(np.log(2.0), np.log(205.0), 0.0125)),
)


def generate_lookup_table(
    formulation: SimplifiedEddyViscosityFormulationProvider = (
        SimplifiedEddyViscosityDeficitFormulation()
    ),
    use_mixing_function: bool = True,
    coordinates: LookupTableCoordinates = DEFAULT_LOOKUP_TABLE_COORDINATES,
    output_filepath: Optional[Path] = None,
) -> xr.DataArray:
    """Generate an EV model lookup table.

    The lookup table contains three-dimensional gridded data of the
    dimensionless wake wind speed deficit as a function of the
    source turbine thrust coefficient, the source turbine incident
    turbulence intensity and the dimensionless downstream distance
    from the source turbine.

    In the cases where the formulation provides a dimensionless wake
    wind speed instead of a deficit, this is converted to a deficit.
    The return array always contains deficit values.

    :param formulation: the class providing the EV model formulation
        functions (initial value and derivative), which must
        implement the ``SimplifiedEddyViscosityFormulationProvider``
        protocol
    :param use_mixing_function: whether to apply the mixing filter
        function in the near wake region
    :param coordinates: the coordinates for ``ti0``, ``ct`` and ``dw`` to
        compute results for and use in the lookup table
    :param output_filepath: optionally a path to save the lookup
        table to; if set to ``None``, the lookup table is not saved
        to file
    """
    if np.isclose(coordinates.dw[0], 2.0):
        coordinates.dw[0] = 2.0

    if np.any(coordinates.ti0 < 0.0):
        raise ValueError("Turbulence intensity values less than zero are not valid.")
    if np.any(coordinates.ct < 0.0):
        raise ValueError("Thrust coefficient values less than zero are not valid.")
    if np.any(coordinates.dw < 2.0):
        raise ValueError(
            "The Eddy Viscosity (EV) model formulations are not defined "
            "below a dimensionless distance of 2.0."
        )
    for coords in coordinates:
        if np.any(np.diff(coords) <= 0.0):
            raise ValueError("All coordinate values must be monotonic increasing.")

    # Solve only for non-zero values of ct
    ct_non_zero: np.ndarray
    if np.isclose(coordinates.ct[0], 0.0):
        ct_non_zero = coordinates.ct[1:]
    else:
        ct_non_zero = coordinates.ct

    # Create two-dimensional matrices of all combinations of ti0 and ct
    ti0_full = coordinates.ti0[:, na]
    ct_full = ct_non_zero[na, :]
    product_shape = np.ones_like(ti0_full * ct_full)
    ti0_full = ti0_full * product_shape
    ct_full = ct_full * product_shape

    # Reshape into flat one-dimensional array for numerical solver
    ti0_flat = ti0_full.flatten(order="C")
    ct_flat = ct_full.flatten(order="C")

    udc0_flat = formulation.initial_u_value(ct=ct_flat, ti0=ti0_flat)

    sol = solve_ivp(
        fun=formulation.streamwise_derivative,
        t_span=[np.min(coordinates.dw), np.max(coordinates.dw)],
        y0=udc0_flat,
        method="RK45",
        t_eval=coordinates.dw,
        vectorized=True,
        args=(ct_flat[:, na], ti0_flat[:, na], use_mixing_function),
    )

    deficit_result = sol.y
    if not formulation.is_deficit_formulation():
        # The result is in terms of dimensionless wake wind speed, from which
        # the dimensionless wake wind speed deficit needs to be calculated
        deficit_result = 1.0 - deficit_result

    lookup_table = deficit_result.reshape(
        coordinates.ti0.size,
        ct_non_zero.size,
        coordinates.dw.size,
        order="C",
    )

    data_array = xr.DataArray(
        data=lookup_table,
        dims=("ti0", "ct", "dw"),
        coords={
            "ti0": coordinates.ti0,
            "ct": ct_non_zero,
            "dw": coordinates.dw,
        },
    )

    # If zero thrust is included, append zero deficit values
    if np.isclose(coordinates.ct[0], 0.0):
        zero_thrust_data_array = xr.DataArray(
            data=0.0,
            dims=("ti0", "ct", "dw"),
            coords={
                "ti0": coordinates.ti0,
                "ct": np.array([0.0]),
                "dw": coordinates.dw,
            },
        )
        data_array = xr.concat(
            objs=(zero_thrust_data_array, data_array),
            dim="ct",
            compat="broadcast_equals",
            join="outer",
            combine_attrs="drop",
        )

    data_array.attrs["use_mixing_function"] = str(use_mixing_function)
    data_array.attrs["formulation"] = str(formulation)

    if output_filepath is not None:
        data_array.to_netcdf(output_filepath)

    return data_array
