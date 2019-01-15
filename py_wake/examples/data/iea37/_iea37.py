from py_wake.examples.data.iea37.iea37_reader import read_iea37_windturbine
from py_wake.wind_turbines import OneTypeWindTurbines
from py_wake.examples.data.iea37 import iea37_path
from py_wake.site._site import UniformSite
import numpy as np


class IEA37_WindTurbines(OneTypeWindTurbines):
    def __init__(self, yaml_filename=iea37_path + 'iea37-335mw.yaml'):
        name, hub_height, diameter, ct_func, power_func = read_iea37_windturbine(yaml_filename)
        super().__init__(name, diameter, hub_height, ct_func, power_func, power_unit='W')


class IEA37Site(UniformSite):
    def __init__(self, n_wt, ti=.75):
        assert n_wt in [16, 36, 64]

        from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm,\
            read_iea37_windrose

        _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
        self.initial_position = np.array(read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)[:2]).T

        UniformSite.__init__(self, freq, ti)


def main():
    if __name__ == '__main__':
        wt = IEA37_WindTurbines()
        print(wt.diameter(0))
        print(wt.hub_height(0))

        site = IEA37Site(16)
        x, y = site.initial_position.T
        dw, cw, dh, dw_order = site.wt2wt_distances(x, y, 70, np.array([[0]]))
        print(dw.shape)
        WD_ilk, WS_ilk, TI_ilk, P_lk = site.local_wind(x, y)
        print(WS_ilk.shape)


main()
