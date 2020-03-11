import pkg_resources
from .deficit_models.noj import NOJ
from .deficit_models.fuga import Fuga, FugaBlockage
from .deficit_models.gaussian import BastankhahGaussian, IEA37SimpleBastankhahGaussian
from py_wake.flow_map import HorizontalGrid

plugins = {
    entry_point.name: entry_point.load()
    for entry_point
    in pkg_resources.iter_entry_points('py_wake.plugins')
}

# 'filled_by_setup.py'
__version__ = '1.0.2'
__release__ = '1.0.2'
