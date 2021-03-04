import yaml
import numpy as np
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction, CubePowerSimpleCt


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
    ws_cutin = operation['cut_in_wind_speed']['default']
    ws_cutout = operation['cut_out_wind_speed']['default']
    ws_rated = operation['rated_wind_speed']['default']
    power = operation['power_curve']['items']['items']
    power_rated = definition['wind_turbine_lookup']['properties']['power']['maximum']
    constant_ct = 4.0 * 1. / 3. * (1.0 - 1. / 3.)

    power_ct_func = CubePowerSimpleCt(ws_cutin, ws_cutout, ws_rated, power_rated, 'W', constant_ct, ct_idle=None,
                                      # additional_models=[]
                                      )

#     def ct(wsp):
#         wsp = np.asarray(wsp)
#         ct = np.zeros_like(wsp, dtype=float)
#         ct[(wsp >= ws_cutin) & (wsp <= ws_cutout)] = constant_ct
#         return ct
#
#     def power(wsp):
#         wsp = np.asarray(wsp)
#         power = np.where((wsp > ws_cutin) & (wsp <= ws_cutout),
#                          np.minimum(power_rated * ((wsp - ws_cutin) / (ws_rated - ws_cutin))**3, power_rated), 0)
#
#         return power
#
#     def dpower(wsp):
#         return np.where((wsp > ws_cutin) & (wsp <= ws_rated),
#                         3 * power_rated * (wsp - ws_cutin)**2 / (ws_rated - ws_cutin)**3,
#                         0)
#
#     def dct(wsp):
#         return wsp * 0  # constant ct

    return wt_id, hubheight, diameter, power_ct_func


class T():
    def __init__(self, ws_cutin, ws_rated, ws_cutout, constant_ct, power_rated):
        self.ws_cutin = ws_cutin
        self.ws_rated = ws_rated
        self.ws_cutout = ws_cutout
        self.constant_ct = constant_ct
        self.power_rated = power_rated

    def ct(self, wsp):
        wsp = np.asarray(wsp)
        ct = np.zeros_like(wsp, dtype=float)
        ct[(wsp >= self.ws_cutin) & (wsp <= self.ws_cutout)] = self.constant_ct
        return ct

    def power(self, wsp):
        wsp = np.asarray(wsp)
        power = np.where((wsp > self.ws_cutin) & (wsp <= self.ws_cutout),
                         np.minimum(self.power_rated * ((wsp - self.ws_cutin) / (self.ws_rated - self.ws_cutin))**3,
                                    self.power_rated), 0)

        return power

    def dpower(self, wsp):
        return np.where((wsp > self.ws_cutin) & (wsp <= self.ws_rated),
                        3 * self.power_rated * (wsp - self.ws_cutin)**2 / (self.ws_rated - self.ws_cutin)**3,
                        0)

    def dct(self, wsp):
        return wsp * 0  # constant ct


def read_iea37_windturbine_deprecated(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    definition = data['definitions']
    wt = definition['wind_turbine']
    wt_id = wt['id']['description']
    hubheight = definition['hub']['properties']['height']['default']
    diameter = definition['rotor']['properties']['radius']['default'] * 2
    operation = definition['operating_mode']['properties']
    ws_cutin = operation['cut_in_wind_speed']['default']
    ws_cutout = operation['cut_out_wind_speed']['default']
    ws_rated = operation['rated_wind_speed']['default']
    power = operation['power_curve']['items']['items']
    power_rated = definition['wind_turbine_lookup']['properties']['power']['maximum']
    constant_ct = 4.0 * 1. / 3. * (1.0 - 1. / 3.)

    t = T(ws_cutin, ws_rated, ws_cutout, constant_ct, power_rated)
    return wt_id, hubheight, diameter, t.power, t.ct, t.dpower, t.dct


def read_iea37_windfarm(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    pos = data['definitions']['position']['items']
    x = pos['xc']
    y = pos['yc']
    aep = data['definitions']['plant_energy']['properties']['annual_energy_production']
    aep = aep['default'], aep['binned']
    return x, y, aep
