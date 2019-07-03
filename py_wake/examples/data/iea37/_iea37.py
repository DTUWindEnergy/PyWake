import numpy as np

from py_wake.examples.data.iea37.iea37_reader import read_iea37_windturbine
from py_wake.wind_turbines import OneTypeWindTurbines
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37.iea37_aepcalc import getTurbLocYAML,\
    getWindRoseYAML, getTurbAtrbtYAML, calcAEP
from py_wake.site._site import UniformSite


class IEA37_WindTurbines(OneTypeWindTurbines):
    def __init__(self, yaml_filename=iea37_path + 'iea37-335mw.yaml'):
        name, hub_height, diameter, ct_func, power_func = read_iea37_windturbine(yaml_filename)
        super().__init__(name, diameter, hub_height, ct_func, power_func, power_unit='W')


class IEA37Site(UniformSite):
    def __init__(self, n_wt, ti=.075):
        assert n_wt in [9, 16, 36, 64]

        from py_wake.examples.data.iea37.iea37_reader import \
            read_iea37_windfarm, read_iea37_windrose

        _, wsp, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
        self.initial_position = np.array(
            read_iea37_windfarm(iea37_path +
                                'iea37-ex%d.yaml' % n_wt)[:2]).T

        UniformSite.__init__(self, freq, ti, ws=wsp)


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
    dw, cw, dh, dw_order = site.wt2wt_distances(x, y, 70, np.array([[0]]))
    print(dw.shape)
    WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x, y)
    print(WS_ilk.shape)


if __name__ == '__main__':
    main()
