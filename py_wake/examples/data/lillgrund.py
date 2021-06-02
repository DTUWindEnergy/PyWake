import numpy as np
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines import OneTypeWindTurbines

wt_x = [361469, 361203, 360936, 360670, 360404, 360137,
        359871, 361203, 360936, 360670, 360404, 360137,
        359871, 359604, 359338, 360936, 360670, 360404,
        360137, 359871, 359604, 359338, 359071, 360670,
        360404, 360137, 359871, 359338, 359071, 358805,
        360390, 360137, 359871, 359604, 359071, 358805,
        359871, 359604, 359338, 359071, 358805, 359604,
        359338, 359071, 358805, 359338, 359071, 358805]
wt_y = [6154543, 6154244, 6153946, 6153648, 6153349, 6153051,
        6152753, 6154695, 6154396, 6154098, 6153800, 6153501,
        6153203, 6152905, 6152606, 6154847, 6154548, 6154250,
        6153952, 6153653, 6153355, 6153057, 6152758, 6154999,
        6154701, 6154402, 6154104, 6153507, 6153209, 6152910,
        6155136, 6154853, 6154554, 6154256, 6153659, 6153361,
        6155005, 6154706, 6154408, 6154110, 6153811, 6155157,
        6154858, 6154560, 6154262, 6155309, 6155010, 6154712]


power_curve = np.array([[3.0, 0.0],
                        [4.0, 65.0],
                        [5.0, 180.0],
                        [6.0, 352.0],
                        [7.0, 590.0],
                        [8.0, 906.0],
                        [9.0, 1308.0],
                        [10.0, 1767.0],
                        [11.0, 2085.0],
                        [12.0, 2234.0],
                        [13.0, 2283.0],
                        [14.0, 2296.0],
                        [15.0, 2299.0],
                        [16.0, 2300.0],
                        [17.0, 2300.0],
                        [18.0, 2300.0],
                        [19.0, 2300.0],
                        [20.0, 2300.0],
                        [21.0, 2300.0],
                        [22.0, 2300.0],
                        [23.0, 2300.0],
                        [24.0, 2300.0],
                        [25.0, 2300.0]])
ct_curve = np.array([[3.0, 0.0],
                     [4.0, 0.81],
                     [5.0, 0.84],
                     [6.0, 0.83],
                     [7.0, 0.85],
                     [8.0, 0.86],
                     [9.0, 0.87],
                     [10.0, 0.79],
                     [11.0, 0.67],
                     [12.0, 0.45],
                     [13.0, 0.34],
                     [14.0, 0.26],
                     [15.0, 0.21],
                     [16.0, 0.17],
                     [17.0, 0.14],
                     [18.0, 0.12],
                     [19.0, 0.10],
                     [20.0, 0.09],
                     [21.0, 0.07],
                     [22.0, 0.07],
                     [23.0, 0.06],
                     [24.0, 0.05],
                     [25.0, 0.05]])


class SWT23(OneTypeWindTurbines):   # Siemens 2.3 MW
    def __init__(self):
        OneTypeWindTurbines.__init__(self, 'SWT23', diameter=93, hub_height=65,
                                     ct_func=self._ct, power_func=self._power, power_unit='kW')

    def _ct(self, u):
        return np.interp(u, ct_curve[:, 0], ct_curve[:, 1])

    def _power(self, u):
        return np.interp(u, power_curve[:, 0], power_curve[:, 1])


LillgrundSWT23 = SWT23

# IMPORTANT NOTE #
# The Weibull parameters are based on 7-months of measurements between 06/2012 - 01/2013 #
# Lillgrund 61m met-mast#
# for more details on the measurements, see https://doi.org/10.1016/j.renene.2016.07.038 #
# to be updated when more data is available...#


class LillgrundSite(UniformWeibullSite):
    def __init__(self, ti=0.1, shear=None):
        f = np.array([3.8, 4.5, 0.4, 2.8, 8.3, 7.5, 9.9, 14.8, 14.3, 17.0, 12.6, 4.1])
        f /= f.sum()
        a = [4.5, 4.7, 3.0, 7.2, 8.8, 8.2, 8.4, 9.5, 9.2, 9.9, 10.3, 6.7]
        k = [1.69, 1.78, 1.82, 1.70, 1.97, 2.49, 2.72, 2.70, 2.88, 3.34, 2.84, 2.23]
        UniformWeibullSite.__init__(self, f, a, k, ti, shear=shear)
        self.initial_position = np.array([wt_x, wt_y]).T


def main():
    wt = SWT23()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    plt.plot(ws, wt.power(ws), '.-')
    plt.show()


if __name__ == '__main__':
    main()
