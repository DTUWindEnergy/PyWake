"""Tests for the Eddy Viscosity model implementation.

"""

from pathlib import Path
from typing import Final

import pytest
import xarray as xr
import xarray.testing as xrt
from py_wake import np
from py_wake.deficit_models import eddy_viscosity_lookup_table_generator
from py_wake.deficit_models.eddy_viscosity import EddyViscosityModel, EddyViscosityDeficitModel
from py_wake.deficit_models.eddy_viscosity_formulations import (
    SimplifiedEddyViscosityDeficitFormulation,
    SimplifiedEddyViscositySpeedFormulation,
)
from py_wake.deficit_models.eddy_viscosity_lookup_table_generator import (
    LookupTableCoordinates,
)
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site import UniformSite

SMALL_COORDINATES: Final[LookupTableCoordinates] = LookupTableCoordinates(
    ti0=np.arange(0.0, 0.5, 0.05),
    ct=np.arange(0.0, 1.2, 0.1),
    dw=np.arange(2.0, 50.0, 5.0),
)


@pytest.fixture(scope="module")
def expected_small_table_filepaths() -> dict[bool, Path]:
    """Mapping of configurations to filepath of expected lookup tables."""
    return {
        True: Path(__file__).parent / "test_data" / "small_mixing_func_lut.nc",
        False: Path(__file__).parent / "test_data" / "small_no_mixing_func_lut.nc",
    }


@pytest.fixture
def calc_deficit_kwargs() -> dict[str, np.ndarray]:
    """Test case arguments to the EV model deficit calculation."""
    return {
        "WS_ilk": np.array(
            [
                [
                    [1.0, 3.0, 5.0, 6.0, 7.0],
                    [1.3, 3.3, 5.3, 6.3, 7.3],
                    [1.6, 3.6, 5.6, 6.6, 7.6],
                ]
            ]
        ),
        "WS_eff_ilk": np.array(
            [
                [
                    [1.0, 3.0, 5.0, 6.0, 7.0],
                    [1.3, 3.3, 5.3, 6.3, 7.3],
                    [1.6, 3.6, 5.6, 6.6, 7.6],
                ]
            ]
        ),
        "TI_ilk": np.array(
            [
                [
                    [0.05, 0.31, 0.11, 0.33, 0.09],
                    [0.02, 0.23, 0.12, 0.34, 0.08],
                    [0.08, 0.22, 0.13, 0.35, 0.07],
                ]
            ]
        ),
        "TI_eff_ilk": np.array(
            [
                [
                    [0.05, 0.31, 0.11, 0.33, 0.09],
                    [0.02, 0.23, 0.12, 0.34, 0.08],
                    [0.08, 0.22, 0.13, 0.35, 0.07],
                ]
            ]
        ),
        "dw_ijlk": np.array(
            [
                [
                    [
                        [240.0, 350.0, 719.0, 468.0, 1100.0],
                        [250.0, 360.0, 729.0, 467.0, 1200.0],
                        [260.0, 370.0, 739.0, 466.0, 1300.0],
                    ],
                    [
                        [260.0, 380.0, 799.0, 465.0, 1400.0],
                        [250.0, 390.0, 789.0, 464.0, 1500.0],
                        [240.0, 400.0, 779.0, 463.0, 1600.0],
                    ],
                ]
            ]
        ),
        "cw_ijlk": np.array(
            [
                [
                    [
                        [1.0, 0.0, 10.0, 468.0, 261.0],
                        [2.0, 0.0, 20.0, 467.0, 262.0],
                        [3.0, 0.0, 30.0, 466.0, 263.0],
                    ],
                    [
                        [4.0, 0.0, 40.0, 465.0, 264.0],
                        [5.0, 0.0, 50.0, 464.0, 265.0],
                        [6.0, 0.0, 60.0, 463.0, 266.0],
                    ],
                ]
            ]
        ),
        "D_src_il": np.array([[100.0, 110.0, 120.0]]),
        "ct_ilk": np.array(
            [
                [
                    [0.85, 0.46, 0.99, 0.74, 0.66],
                    [0.82, 0.40, 0.90, 0.72, 0.60],
                    [0.80, 0.44, 0.95, 0.70, 0.64],
                ]
            ]
        ),
    }


@pytest.fixture
def expected_eddy_viscosity_deficit() -> dict[bool, np.ndarray]:
    """Dictionary of expected array of wind speed deficit results."""
    return {
        True: np.array(
            [
                [
                    [
                        [
                            6.96868051e-01,
                            4.67763151e-01,
                            1.07210664e00,
                            1.18777181e-18,
                            8.12234438e-05,
                        ],
                        [
                            9.37181587e-01,
                            5.83580345e-01,
                            1.18303699e00,
                            2.71931934e-16,
                            1.65325731e-04,
                        ],
                        [
                            1.01701033e00,
                            7.53404110e-01,
                            1.40837152e00,
                            3.66858917e-14,
                            5.02598003e-04,
                        ],
                    ],
                    [
                        [
                            6.73933746e-01,
                            4.42911782e-01,
                            7.97521669e-01,
                            1.57014767e-18,
                            4.40164364e-04,
                        ],
                        [
                            9.29889161e-01,
                            5.57092562e-01,
                            7.56823336e-01,
                            3.61638662e-16,
                            7.01415869e-04,
                        ],
                        [
                            1.03117250e00,
                            7.22797673e-01,
                            8.88125522e-01,
                            4.81576789e-14,
                            1.60817674e-03,
                        ],
                    ],
                ]
            ]
        ),
        False: np.array(
            [
                [
                    [
                        [
                            6.92221644e-01,
                            4.52342217e-01,
                            8.12498165e-01,
                            1.56648041e-17,
                            2.99629085e-04,
                        ],
                        [
                            9.33005834e-01,
                            5.67629836e-01,
                            9.58060024e-01,
                            1.49796801e-15,
                            5.38339200e-04,
                        ],
                        [
                            1.01421685e00,
                            7.36641271e-01,
                            1.21832676e00,
                            1.16340321e-13,
                            1.36978213e-03,
                        ],
                    ],
                    [
                        [
                            6.67029458e-01,
                            4.24406660e-01,
                            6.42106111e-01,
                            1.93766930e-17,
                            9.20518779e-04,
                        ],
                        [
                            9.25760009e-01,
                            5.37724087e-01,
                            6.11193049e-01,
                            1.90392913e-15,
                            1.39451661e-03,
                        ],
                        [
                            1.03117250e00,
                            7.02166487e-01,
                            7.82234849e-01,
                            1.47943640e-13,
                            2.92398724e-03,
                        ],
                    ],
                ]
            ]
        ),
    }


@pytest.mark.parametrize("use_effective_ws", [True, False])
@pytest.mark.parametrize("use_effective_ti", [True, False])
@pytest.mark.parametrize("use_mixing_function", [True, False])
def test_calc_deficit_returns_correct_values(
    use_effective_ws: bool,
    use_effective_ti: bool,
    use_mixing_function: bool,
    expected_small_table_filepaths: dict[bool, Path],
    calc_deficit_kwargs: dict[str, np.ndarray],
    expected_eddy_viscosity_deficit: dict[bool, np.ndarray],
) -> None:
    """Assert the deficit model returns the correct values."""
    model = EddyViscosityDeficitModel(
        use_effective_ws=use_effective_ws,
        use_effective_ti=use_effective_ti,
        use_mixing_function=use_mixing_function,
        formulation=None,
        lookup_table_filepath=expected_small_table_filepaths[use_mixing_function],
    )
    deficit = model.calc_deficit(**calc_deficit_kwargs)
    assert np.allclose(deficit, expected_eddy_viscosity_deficit[use_mixing_function])


def test_eddy_viscosity_no_formulation_or_lookup_table_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match=r"either.*formulation.*lookup.*must be provided",
    ):
        EddyViscosityDeficitModel(
            formulation=None,
            lookup_table_filepath=None,
        )


def test_eddy_viscosity_invalid_negative_ws_raises_error() -> None:
    with pytest.raises(ValueError, match=r"negative wind speed.*not valid"):
        model = EddyViscosityDeficitModel()
        model.calc_deficit(
            WS_ilk=np.array([[[-0.01]]]),  # Invalid
            WS_eff_ilk=np.array([[[10.0]]]),
            TI_ilk=np.array([[[0.1]]]),
            TI_eff_ilk=np.array([[[0.1]]]),
            dw_ijlk=np.array([[[[250.0]]]]),
            cw_ijlk=np.array([[[[200.0]]]]),
            D_src_il=np.array([[[100.0]]]),
            ct_ilk=np.array([[[0.1]]]),
        )


def test_eddy_viscosity_invalid_negative_ct_raises_error() -> None:
    with pytest.raises(ValueError, match=r"negative thrust coefficient.*not valid"):
        model = EddyViscosityDeficitModel()
        model.calc_deficit(
            WS_ilk=np.array([[[10.0]]]),
            WS_eff_ilk=np.array([[[10.0]]]),
            TI_ilk=np.array([[[0.1]]]),
            TI_eff_ilk=np.array([[[0.1]]]),
            dw_ijlk=np.array([[[[250.0]]]]),
            cw_ijlk=np.array([[[[200.0]]]]),
            D_src_il=np.array([[[100.0]]]),
            ct_ilk=np.array([[[-0.1]]]),  # Invalid
        )


def test_eddy_viscosity_invalid_high_ct_raises_error() -> None:
    with pytest.raises(ValueError, match=r"higher than .* are not supported"):
        model = EddyViscosityDeficitModel()
        model.calc_deficit(
            WS_ilk=np.array([[[10.0]]]),
            WS_eff_ilk=np.array([[[10.0]]]),
            TI_ilk=np.array([[[0.1]]]),
            TI_eff_ilk=np.array([[[0.1]]]),
            dw_ijlk=np.array([[[[250.0]]]]),
            cw_ijlk=np.array([[[[200.0]]]]),
            D_src_il=np.array([[[100.0]]]),
            ct_ilk=np.array([[[2.0]]]),  # Invalid
        )


def test_eddy_viscosity_invalid_negative_ti_raises_error() -> None:
    with pytest.raises(ValueError, match=r"negative turbulence intensity.*not valid"):
        model = EddyViscosityDeficitModel()
        model.calc_deficit(
            WS_ilk=np.array([[[10.0]]]),
            WS_eff_ilk=np.array([[[10.0]]]),
            TI_ilk=np.array([[[-0.1]]]),  # Invalid
            TI_eff_ilk=np.array([[[-0.1]]]),
            dw_ijlk=np.array([[[[250.0]]]]),
            cw_ijlk=np.array([[[[200.0]]]]),
            D_src_il=np.array([[[100.0]]]),
            ct_ilk=np.array([[[0.8]]]),
        )


def test_eddy_viscosity_wind_farm_model_calculates_correct_aep() -> None:
    model = EddyViscosityModel(
        site=UniformSite([1, 0, 0, 0], ti=0.075),
        windTurbines=HornsrevV80(),
    )
    results = model(
        x=np.array([0.0, 200.0, 300.0]),
        y=np.array([0.0, 0.0, 200.0]),
        h=np.array([60.0, 80.0, 100.0]),
    )
    assert np.isclose(results.aep().sum().values.item(), 46.952268386759414)


@pytest.mark.parametrize("use_mixing_function", [True, False])
def test_small_table_generation(
    tmp_path: Path,
    use_mixing_function: bool,
    expected_small_table_filepaths: dict[bool, Path],
) -> None:
    """Assert generator produces the expected small lookup tables.

    This test is undertaken on the small tables to avoid the larger
    computational time and larger file sizes associated with the full
    tables.
    """
    tmp_result_table_filepath = tmp_path / "test_lookup_table.nc"
    eddy_viscosity_lookup_table_generator.generate_lookup_table(
        formulation=SimplifiedEddyViscosityDeficitFormulation(),
        use_mixing_function=use_mixing_function,
        coordinates=SMALL_COORDINATES,
        output_filepath=tmp_result_table_filepath,
    )
    lookup_table = xr.open_dataarray(tmp_result_table_filepath, engine="h5netcdf")
    expected_lookup_table = xr.open_dataarray(
        expected_small_table_filepaths[use_mixing_function],
        engine="h5netcdf",
    )
    xrt.assert_allclose(lookup_table, expected_lookup_table)


def test_eddy_viscosity_lookup_table_generator_invalid_negative_ti_raises_error() -> None:
    coordinates = LookupTableCoordinates(
        ti0=np.arange(-0.01, 0.3, 0.02),
        ct=np.arange(0.1, 0.9, 0.02),
        dw=np.arange(2.0, 60.0, 0.1),
    )
    with pytest.raises(ValueError, match=r".*turbulence.*not valid.*"):
        eddy_viscosity_lookup_table_generator.generate_lookup_table(
            formulation=SimplifiedEddyViscosityDeficitFormulation(),
            use_mixing_function=True,
            coordinates=coordinates,
        )


def test_eddy_viscosity_lookup_table_generator_invalid_negative_ct_raises_error() -> None:
    coordinates = LookupTableCoordinates(
        ti0=np.arange(0.01, 0.3, 0.02),
        ct=np.arange(-0.1, 0.9, 0.02),
        dw=np.arange(2.0, 60.0, 0.1),
    )
    with pytest.raises(ValueError, match=r".*thrust.*not valid.*"):
        eddy_viscosity_lookup_table_generator.generate_lookup_table(
            formulation=SimplifiedEddyViscosityDeficitFormulation(),
            use_mixing_function=True,
            coordinates=coordinates,
        )


def test_eddy_viscosity_lookup_table_generator_invalid_small_dw_raises_error() -> None:
    coordinates = LookupTableCoordinates(
        ti0=np.arange(0.01, 0.3, 0.02),
        ct=np.arange(0.1, 0.9, 0.02),
        dw=np.arange(1.0, 60.0, 0.1),
    )
    with pytest.raises(ValueError, match=r".*not defined.*below.*2.*"):
        eddy_viscosity_lookup_table_generator.generate_lookup_table(
            formulation=SimplifiedEddyViscosityDeficitFormulation(),
            use_mixing_function=True,
            coordinates=coordinates,
        )


def test_eddy_viscosity_lookup_table_generator_invalid_coordinates_raises_error() -> None:
    coordinates = LookupTableCoordinates(
        ti0=np.arange(0.01, 0.3, 0.02),
        ct=np.arange(0.1, 0.9, 0.02),
        dw=np.array([20.0, 15.0, 10.0]),
    )
    with pytest.raises(ValueError, match=r".*must be monotonic increasing"):
        eddy_viscosity_lookup_table_generator.generate_lookup_table(
            formulation=SimplifiedEddyViscosityDeficitFormulation(),
            use_mixing_function=True,
            coordinates=coordinates,
        )


@pytest.mark.parametrize("use_mixing_function", [True, False])
def test_different_formulations_give_close_results(use_mixing_function: bool) -> None:
    """Assert deficit and speed formulations give close result tables."""
    coordinates = LookupTableCoordinates(
        ti0=np.arange(0.01, 0.3, 0.02),
        ct=np.arange(0.1, 0.9, 0.02),
        dw=np.arange(2.0, 60.0, 0.1),
    )
    deficit_formulation_lookup_table = (
        eddy_viscosity_lookup_table_generator.generate_lookup_table(
            formulation=SimplifiedEddyViscosityDeficitFormulation(),
            use_mixing_function=use_mixing_function,
            coordinates=coordinates,
        )
    )
    speed_formulation_lookup_table = (
        eddy_viscosity_lookup_table_generator.generate_lookup_table(
            formulation=SimplifiedEddyViscositySpeedFormulation(),
            use_mixing_function=use_mixing_function,
            coordinates=coordinates,
        )
    )
    relative = (1.0 - speed_formulation_lookup_table) / (
        1.0 - deficit_formulation_lookup_table
    )
    assert np.allclose(relative.values, 1.0, atol=0.01, rtol=0.01)
