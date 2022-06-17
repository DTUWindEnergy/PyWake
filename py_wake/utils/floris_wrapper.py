import autograd.numpy as np
import yaml
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine


def floris_yaml_to_pywake_turbine(yamlFile, interpolation_method='linear',
                                  loadFunctions=None):

    # read floris file
    with open(yamlFile) as file:
        floris_data = yaml.load(file, Loader=yaml.FullLoader)

    # parse floris file
    wind_speed = np.array(floris_data['power_thrust_table']['wind_speed'])
    cp = np.array(floris_data['power_thrust_table']['power'])
    ct = np.array(floris_data['power_thrust_table']['thrust'])
    turbine_name = floris_data['turbine_type']
    rotor_diameter = floris_data['rotor_diameter']
    hub_height = floris_data['hub_height']

    # return pywake turbine
    power_ct_table = PowerCtTabular(wind_speed,
                                    cp * 0.5 * 1.225 *
                                    np.pi * (rotor_diameter // 2) ** 2 *
                                    wind_speed ** 3,
                                    'w', ct, ws_cutin=None, ws_cutout=None,
                                    method=interpolation_method)

    return WindTurbine(turbine_name, diameter=rotor_diameter,
                       hub_height=hub_height,
                       powerCtFunction=power_ct_table,
                       loadFunctions=loadFunctions)
