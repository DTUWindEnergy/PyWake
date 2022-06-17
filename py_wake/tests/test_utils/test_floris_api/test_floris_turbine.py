from py_wake.utils.floris_wrapper import floris_yaml_to_pywake_turbine
import os


def test_rotor_diameter():
    path_to_iea10mw_yaml = os.path.dirname(__file__) + os.sep + 'iea_10MW.yaml'
    wt = floris_yaml_to_pywake_turbine(path_to_iea10mw_yaml)
    assert wt.diameter() == 198
    assert wt.hub_height() == 119
