import numpy as np
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines import OneTypeWindTurbines

wt_x = [134205, 134509, 134813, 135118, 135423]

wt_y = [538122, 538095, 538067, 538037, 538012]

power_curve = np.array([[3.0, 0.0],
                        [4.0, 15.0],
                        [5.0, 121.0],
                        [6.0, 251.0],
                        [7.0, 433.0],
                        [8.0, 667.0],
                        [9.0, 974.0],
                        [10.0, 1319.0],
                        [11.0, 1675.0],
                        [12.0, 2004.0],
                        [13.0, 2281.0],
                        [14.0, 2463.0],
                        [15.0, 2500.0],
                        [16.0, 2500.0],
                        [17.0, 2500.0],
                        [18.0, 2500.0],
                        [19.0, 2500.0],
                        [20.0, 2500.0],
                        [21.0, 2500.0],
                        [22.0, 2500.0],
                        [23.0, 2500.0],
                        [24.0, 2500.0],
                        [25.0, 2500.0]])

# Calculated ct curve using PHATAS (BEM code from ECN)
ct_curve = np.array([[3.0, 0.0],
                     [4.0, 0.85199],
                     [5.0, 0.85199],
                     [6.0, 0.80717],
                     [7.0, 0.78455],
                     [8.0, 0.76444],
                     [9.0, 0.72347],
                     [10.0, 0.66721],
                     [11.0, 0.62187],
                     [12.0, 0.57274],
                     [13.0, 0.50807],
                     [14.0, 0.42737],
                     [15.0, 0.33182],
                     [16.0, 0.26268],
                     [17.0, 0.21476],
                     [18.0, 0.18003],
                     [19.0, 0.15264],
                     [20.0, 0.13089],
                     [21.0, 0.11374],
                     [22.0, 0.09945],
                     [23.0, 0.08766],
                     [24.0, 0.07796],
                     [25.0, 0.06971]])


class N80(OneTypeWindTurbines):
    def __init__(self):
        OneTypeWindTurbines.__init__(self, 'N80', diameter=80.0, hub_height=80.0,
                                     ct_func=self._ct, power_func=self._power, power_unit='kW')

    def _ct(self, u):
        return np.interp(u, ct_curve[:, 0], ct_curve[:, 1])

    def _power(self, u):
        return np.interp(u, power_curve[:, 0], power_curve[:, 1])


def main():
    if __name__ == '__main__':
        wt = N80()
        print('Diameter', wt.diameter())
        print('Hub height', wt.hub_height())
        ws = np.arange(3, 25)
        import matplotlib.pyplot as plt
        plt.plot(ws, wt.power(ws), '.-')
        plt.show()


main()
