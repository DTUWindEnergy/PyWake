from py_wake.site.wasp_grid_site import WaspGridSite
from py_wake import NOJ
import numpy as np
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite


def main():
    if __name__ == '__main__':
        site = ParqueFicticioSite()
        windTurbines = IEA37_WindTurbines()
        wf_model = NOJ(site, windTurbines)
        x, y = site.initial_position.T
        print(wf_model(x, y).aep())


main()
