# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:39:03 2024

@author: mikf
"""

from pathlib import Path

import numpy as np
import pandas as pd
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

DATA_PATH = Path(__file__).parent

# IEA 22 MW reference turbine
# Based on tabular data found here: https://github.com/IEAWindTask37/IEA-22-280-RWT/blob/main/outputs/01_steady_states/HAWC2/iea-22-280-rwt-steady-states-hawc2.yaml
IEA_22MW_280_RWT_data = pd.read_csv(DATA_PATH / "IEA-22-280-RWT_tabular.csv", sep=';')
IEA_22MW_280_RWT_power_curve = np.array([IEA_22MW_280_RWT_data["Wind [m/s]"], IEA_22MW_280_RWT_data["Power [MW]"]]).T
IEA_22MW_280_RWT_ct_curve = np.array([IEA_22MW_280_RWT_data["Wind [m/s]"], IEA_22MW_280_RWT_data["Thrust Coefficient [-]"]]).T


class IEA_22MW_280_RWT(WindTurbine):
    def __init__(self, method="linear"):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(
            self,
            name="IEA_22MW_280_RWT",
            diameter=284,
            hub_height=170,
            powerCtFunction=PowerCtTabular(
                IEA_22MW_280_RWT_power_curve[:, 0],
                IEA_22MW_280_RWT_power_curve[:, 1],
                "MW",
                IEA_22MW_280_RWT_ct_curve[:, 1],
                method=method,
            ),
        )


def main():
    wt = IEA_22MW_280_RWT()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    plt.plot(ws, wt.power(ws), '.-', label='power [W]')
    c = plt.plot([], label='ct')[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), '.-', color=c)
    plt.show()


if __name__ == '__main__':
    main()
