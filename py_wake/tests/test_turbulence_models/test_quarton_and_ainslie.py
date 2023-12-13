"""Tests for the Quarton and Ainslie turbulence model.

"""

import copy

import pytest

from py_wake import np
from py_wake.turbulence_models.quarton_and_ainslie import (
    QuartonAndAinslieTurbulenceModel,
)


@pytest.fixture
def calc_added_turbulence_kwargs() -> dict[str, np.ndarray]:
    """Test case arguments to the wake added turbulence calculation."""
    return {
        "WS_ilk": np.array(
            [
                [
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                ]
            ]
        ),
        "WS_eff_ilk": np.array(
            [
                [
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                    [2.0, 4.0, 6.0, 8.0, 10.0],
                ]
            ]
        ),
        "TI_ilk": np.array(
            [
                [
                    [0.05, 0.31, 0.11, 0.33, 0.09],
                    [0.02, 0.23, 0.11, 0.33, 0.09],
                    [0.08, 0.22, 0.11, 0.31, 0.09],
                ]
            ]
        ),
        "TI_eff_ilk": np.array(
            [
                [
                    [0.05, 0.31, 0.11, 0.33, 0.09],
                    [0.02, 0.23, 0.11, 0.33, 0.09],
                    [0.08, 0.22, 0.11, 0.31, 0.09],
                ]
            ]
        ),
        "dw_ijlk": np.array(
            [
                [
                    [
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                    ],
                    [
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                        [123.0, 456.0, 789.0, 468.0, 246.0],
                    ],
                ]
            ]
        ),
        "cw_ijlk": np.array(
            [
                [
                    [
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                    ],
                    [
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                        [123.0, 456.0, 0.0, 468.0, 246.0],
                    ],
                ]
            ]
        ),
        "ct_ilk": np.array(
            [
                [
                    [0.85, 0.46, 0.99, 0.74, 0.66],
                    [0.80, 0.40, 0.90, 0.70, 0.60],
                    [0.80, 0.46, 0.99, 0.70, 0.66],
                ]
            ]
        ),
        "D_src_il": np.array([[100.0]]),
        "D_dst_ijl": np.array([[[100.0]]]),
        "wake_radius_ijlk": np.array(
            [
                [
                    [
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                    ],
                    [
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                        [150.0, 250.0, 350.0, 300.0, 200.0],
                    ],
                ]
            ]
        ),
    }


@pytest.fixture
def expected_added_turbulence() -> np.ndarray:
    """Expected array of added turbulence results."""
    return np.array(
        [
            [
                [0.07305408, 0.0, 0.19462119, 0.0, 0.0153155],
                [0.03586518, 0.0, 0.09688273, 0.0, 0.01408464],
                [0.09177531, 0.0, 0.19462119, 0.0, 0.0153155],
            ],
            [
                [0.07305408, 0.0, 0.19462119, 0.0, 0.0153155],
                [0.03586518, 0.0, 0.09688273, 0.0, 0.01408464],
                [0.09177531, 0.0, 0.19462119, 0.0, 0.0153155],
            ],
        ]
    )


@pytest.mark.parametrize("use_effective_ws", [True, False])
@pytest.mark.parametrize("use_effective_ti", [True, False])
def test_calc_added_turbulence(
    use_effective_ws: bool,
    use_effective_ti: bool,
    calc_added_turbulence_kwargs: dict[str, np.ndarray],
    expected_added_turbulence: np.ndarray,
) -> None:
    """Assert the turbulence model returns the correct values."""
    model = QuartonAndAinslieTurbulenceModel(
        use_effective_ws=use_effective_ws,
        use_effective_ti=use_effective_ti,
    )
    added_turbulence = model.calc_added_turbulence(**calc_added_turbulence_kwargs)
    assert np.allclose(added_turbulence, expected_added_turbulence)


def test_invalid_negative_wind_speed_raises_error(
    calc_added_turbulence_kwargs: dict[str, np.ndarray],
) -> None:
    model = QuartonAndAinslieTurbulenceModel(
        use_effective_ws=True,
        use_effective_ti=False,
    )
    calc_added_turbulence_invalid_kwargs = copy.deepcopy(calc_added_turbulence_kwargs)
    calc_added_turbulence_invalid_kwargs["WS_eff_ilk"] = np.array(
        [
            [
                [-2.0, 0.0, 2.0, 4.0, 6.0],
                [-2.0, 0.0, 2.0, 4.0, 6.0],
                [-2.0, 0.0, 2.0, 4.0, 6.0],
            ]
        ]
    )

    with pytest.raises(
        expected_exception=ValueError,
        match="Negative wind speed values are not valid",
    ):
        _ = model.calc_added_turbulence(**calc_added_turbulence_invalid_kwargs)


def test_invalid_negative_turbulence_raises_error(
    calc_added_turbulence_kwargs: dict[str, np.ndarray],
) -> None:
    model = QuartonAndAinslieTurbulenceModel(
        use_effective_ws=True,
        use_effective_ti=False,
    )
    calc_added_turbulence_invalid_kwargs = copy.deepcopy(calc_added_turbulence_kwargs)
    calc_added_turbulence_invalid_kwargs["TI_ilk"] = np.array(
        [
            [
                [0.05, 0.31, 0.11, 0.33, 0.09],
                [0.02, 0.23, -0.11, 0.33, 0.09],
                [0.08, 0.22, 0.11, 0.31, 0.09],
            ]
        ]
    )

    with pytest.raises(
        expected_exception=ValueError,
        match="Negative turbulence intensity values are not valid",
    ):
        _ = model.calc_added_turbulence(**calc_added_turbulence_invalid_kwargs)
