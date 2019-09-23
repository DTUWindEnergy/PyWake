from py_wake.aep_calculator import AEPCalculator
from py_wake.site.wasp_grid_site import WaspGridSiteBase
from py_wake.wake_models.noj import NOJ
import numpy as np
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path


def main():
    if __name__ == '__main__':
        site = WaspGridSiteBase.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=False)
        site.initial_position = np.array([
            [263655.0, 6506601.0],
            [263891.1, 6506394.0],
            [264022.2, 6506124.0],
            [264058.9, 6505891.0],
            [264095.6, 6505585.0],
            [264022.2, 6505365.0],
            [264022.2, 6505145.0],
            [263936.5, 6504802.0],
        ])
        windTurbines = IEA37_WindTurbines()
        wake_model = NOJ(site, windTurbines)
        x, y = site.initial_position.T
        aep_calculator = AEPCalculator(wake_model)
        print(aep_calculator.calculate_AEP(x, y).sum())


main()
