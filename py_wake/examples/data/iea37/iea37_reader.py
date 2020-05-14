import yaml
import numpy as np


def read_iea37_windrose(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    prop = data['definitions']['wind_inflow']['properties']
    wdir = prop['direction']['bins']
    wsp = prop['speed']['default']
    p_wdir = prop['probability']['default']
    return map(np.atleast_1d, (wdir, wsp, p_wdir))


def read_iea37_windturbine(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    definition = data['definitions']
    wt = definition['wind_turbine']
    wt_id = wt['id']['description']
    hubheight = definition['hub']['properties']['height']['default']
    diameter = definition['rotor']['properties']['radius']['default'] * 2
    operation = definition['operating_mode']['properties']
    wsp_cut_in = operation['cut_in_wind_speed']['default']
    wsp_cut_out = operation['cut_out_wind_speed']['default']
    wsp_rated = operation['rated_wind_speed']['default']
    power = operation['power_curve']['items']['items']
    power_rated = definition['wind_turbine_lookup']['properties']['power']['maximum']
    constant_ct = 4.0 * 1. / 3. * (1.0 - 1. / 3.)

    def ct(wsp):
        wsp = np.asarray(wsp)
        ct = np.zeros_like(wsp, dtype=np.float)
        ct[(wsp >= wsp_cut_in) & (wsp <= wsp_cut_out)] = constant_ct
        return ct

    def power(wsp):
        wsp = np.asarray(wsp)
        power = np.where((wsp > wsp_cut_in) & (wsp <= wsp_cut_out),
                         np.minimum(power_rated * ((wsp - wsp_cut_in) / (wsp_rated - wsp_cut_in))**3, power_rated), 0)

        return power

    def dpower(wsp):
        return np.where((wsp > wsp_cut_in) & (wsp <= wsp_rated),
                        3 * power_rated * (wsp - wsp_cut_in)**2 / (wsp_rated - wsp_cut_in)**3,
                        0)

    def dct(wsp):
        return wsp * 0  # constant ct

    return wt_id, hubheight, diameter, ct, power, dct, dpower


def read_iea37_windfarm(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    pos = data['definitions']['position']['items']
    x = pos['xc']
    y = pos['yc']
    aep = data['definitions']['plant_energy']['properties']['annual_energy_production']
    aep = aep['default'], aep['binned']
    return x, y, aep
