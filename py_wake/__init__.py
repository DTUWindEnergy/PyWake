"""PyWake

An open source wind farm simulation tool capable of calculating wind farm flow fields,
power production and annual energy production (AEP) of wind farms.
"""
import pkg_resources
from .deficit_models.noj import NOJ, NOJLocal
from .deficit_models.fuga import Fuga, FugaBlockage
from .deficit_models.gaussian import BastankhahGaussian, IEA37SimpleBastankhahGaussian
from .deficit_models.gcl import GCL, GCLLocal
from py_wake.flow_map import HorizontalGrid, XYGrid, YZGrid
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.deficit_models.deficit_model import DeficitModel


plugins = {
    entry_point.name: entry_point.load()
    for entry_point
    in pkg_resources.iter_entry_points('py_wake.plugins')
}

# 'filled_by_setup.py'
__version__ = '2.1.1'
__release__ = '2.1.1'
