import numpy as np

from py_wake.examples.data.iea37.iea37_reader import read_iea37_windturbine, read_iea37_windturbine_deprecated
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37.iea37_aepcalc import getTurbLocYAML,\
    getWindRoseYAML, getTurbAtrbtYAML, calcAEP
from py_wake.site._site import UniformSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.wind_turbines_deprecated import DeprecatedOneTypeWindTurbines


class IEA37_WindTurbines(WindTurbine):
    def __init__(self, yaml_filename=iea37_path + 'iea37-335mw.yaml'):
        name, hub_height, diameter, power_ct_func = read_iea37_windturbine(yaml_filename)
        super().__init__(name, diameter, hub_height, power_ct_func)


class IEA37WindTurbinesDeprecated(DeprecatedOneTypeWindTurbines):
    def __init__(self, yaml_filename=iea37_path + 'iea37-335mw.yaml', gradient_functions=True):
        name, hub_height, diameter, power_func, ct_func, dpower_func, dct_func = read_iea37_windturbine_deprecated(
            yaml_filename)
        super().__init__(name, diameter, hub_height, ct_func, power_func, 'w')
        if gradient_functions:
            self.set_gradient_funcs(dpower_func, dct_func)


IEA37WindTurbines = IEA37_WindTurbines


class IEA37Site(UniformSite):
    def __init__(self, n_wt, ti=.075, shear=None):
        assert n_wt in [9, 16, 36, 64]

        from py_wake.examples.data.iea37.iea37_reader import \
            read_iea37_windfarm, read_iea37_windrose

        _, wsp, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
        UniformSite.__init__(self, freq, ti, ws=wsp, shear=shear)
        self.initial_position = np.array(read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)[:2]).T


class IEA37AEPCalc():
    """Run the AEP calculator provided by IEA Task 37"""

    def __init__(self, n_wt):
        assert n_wt in [9, 16, 36, 64]
        self.n_wt = n_wt

    def get_aep(self):
        turb_coords, _, _ = \
            getTurbLocYAML(iea37_path + 'iea37-ex%d.yaml' % self.n_wt)
        wind_dir, wind_freq, wind_speed = \
            getWindRoseYAML(iea37_path + 'iea37-windrose.yaml')
        turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = \
            getTurbAtrbtYAML(iea37_path + 'iea37-335mw.yaml')
        AEP = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                      turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
        return AEP


def main():
    wt = IEA37_WindTurbines()
    print(wt.diameter(0))
    print(wt.hub_height(0))

    site = IEA37Site(16)
    x, y = site.initial_position.T
    lw = site.local_wind(x, y)
    print(lw.WS)


if __name__ == '__main__':
    main()
