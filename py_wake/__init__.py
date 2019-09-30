import pkg_resources

plugins = {
    entry_point.name: entry_point.load()
    for entry_point
    in pkg_resources.iter_entry_points('topfarm.plugins')
}

# 'filled_by_setup.py'
__version__ = '1.0.2'
__release__ = '1.0.2'
