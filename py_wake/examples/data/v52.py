import numpy as np
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

ws = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
      16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
p = [0.0, 25.5, 67.4, 125.0, 203.0, 304.0, 425.0, 554.0, 671.0, 759.0, 811.0, 836.0, 846.0, 849.0, 850.0, 850.0, 850.0,
     850.0, 850.0, 850.0, 850.0, 850.0, 850.0]
ct = [0.058, 0.82, 0.82, 0.82, 0.82, 0.82, 0.79, 0.751, 0.675, 0.602, 0.418, 0.323, 0.256, 0.209, 0.174, 0.147, 0.126,
      0.108, 0.094, 0.083, 0.073, 0.065, 0.058]


class V52(WindTurbine):
    def __init__(self, hub_height=44):
        WindTurbine.__init__(self, name='V52', diameter=52, hub_height=hub_height,
                             powerCtFunction=PowerCtTabular(ws=ws, power=p, power_unit='kw',
                                                            ct=ct, ws_cutin=4, ws_cutout=25))


if __name__ == '__main__':
    # data copied from VindPro
    # 18m/s manually insert
    power_str = """Vindhastighed [m/s]    Effekt [kW]    Cp
    3.00    0.00    0
    4.00    25.50    0.306
    5.00    67.40    0.415
    6.00    125.00    0.445
    7.00    203.00    0.455
    8.00    304.00    0.456
    9.00    425.00    0.448
    10.00    554.00    0.426
    11.00    671.00    0.388
    12.00    759.00    0.338
    13.00    811.00    0.284
    14.00    836.00    0.234
    15.00    846.00    0.193
    16.00    849.00    0.159
    17.00    850.00    0.133
    18.00    850.00    0.0
    19.00    850.00    0.095
    20.00    850.00    0.082
    21.00    850.00    0.071
    22.00    850.00    0.061
    23.00    850.00    0.054
    24.00    850.00    0.047
    25.00    850.00    0.042"""

    # ct at 3m/s set to ct at 25m/s
    ct_str = """Vindhastighed [m/s]    Ct
    3.00    0.058
    4.00    0.820
    5.00    0.820
    6.00    0.820
    7.00    0.820
    8.00    0.820
    9.00    0.790
    10.00    0.751
    11.00    0.675
    12.00    0.602
    13.00    0.418
    14.00    0.323
    15.00    0.256
    16.00    0.209
    17.00    0.174
    18.00    0.147
    19.00    0.126
    20.00    0.108
    21.00    0.094
    22.00    0.083
    23.00    0.073
    24.00    0.065
    25.00    0.058"""

    ws_p, p, cp = np.array([l.split() for l in power_str.split("\n")[1:]], dtype=float).T
    ws_ct, ct = np.array([l.split() for l in ct_str.split("\n")[1:]], dtype=float).T
    assert np.all(ws_p == ws_ct)
    print(p)
    print(ct)
    print(np.array([ws_p, p, ct]).tolist())
