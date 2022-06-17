"""PyWake

An open source wind farm simulation tool capable of calculating wind farm flow fields,
power production and annual energy production (AEP) of wind farms.
"""
import pkg_resources
import numpy
from py_wake.utils.numpy_utils import NumpyWrapper
np = numpy
locals()['np'] = NumpyWrapper()

from .deficit_models.noj import NOJ, NOJLocal  # nopep8
from .deficit_models.fuga import Fuga, FugaBlockage  # nopep8
from .deficit_models.gaussian import BastankhahGaussian, IEA37SimpleBastankhahGaussian  # nopep8
from .deficit_models.gcl import GCL, GCLLocal  # nopep8
from py_wake.flow_map import HorizontalGrid, XYGrid, YZGrid, XZGrid  # nopep8
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel  # nopep8
from py_wake.deficit_models.deficit_model import DeficitModel  # nopep8


plugins = {
    entry_point.name: entry_point.load()
    for entry_point
    in pkg_resources.iter_entry_points('py_wake.plugins')
}

# 'filled_by_setup.py'
__version__ = '2.3.0'
__release__ = '2.3.0'
