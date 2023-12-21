"""Tests for the simplified Gaussian rotor average model.

"""

import pytest

from py_wake.deficit_models.gcl import GCLDeficit
from py_wake.examples.data.hornsrev1 import V80
from py_wake.rotor_avg_models.simplified_gaussian_rotor_average_model import SimplifiedGaussianRotorAverageModel
from py_wake.site import UniformSite
from py_wake.wind_farm_models.engineering_models import PropagateDownwind


def test_missing_sigma_ijlk_raises_error() -> None:
    wind_farm_model = PropagateDownwind(
        site=UniformSite(),
        windTurbines=V80(),
        wake_deficitModel=GCLDeficit(rotorAvgModel=SimplifiedGaussianRotorAverageModel()),
    )
    with pytest.raises(
        expected_exception=AttributeError,
        match="'GCLDeficit' has no attribute 'sigma_ijlk'",
    ):
        _ = wind_farm_model([0, 1000], [0, 0])
