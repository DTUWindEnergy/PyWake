import numpy as np


class WindTurbines():

    def __init__(self, names, diameters, hub_heights, ct_funcs, power_funcs, power_unit):
        """Set of wind turbines

        Parameters
        ----------
        names : array_like
            Wind turbine names
        diameters : array_like
            Diameter of wind turbines
        hub_heights : array_like
            Hub height of wind turbines
        ct_func : function
            func(ws, wt_type) -> ct
        power_func : function
            func(ws, wt_type) -> power
        power_unit : {'W', 'kW', 'MW', 'GW'}
            Unit of power_func output (case insensitive)
        """
        self._names = names
        self._diameters = np.array(diameters)
        self._hub_heights = np.array(hub_heights)
        self.ct_funcs = ct_funcs
        self.power_scale = {'w': 1, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}[power_unit.lower()]
        if self.power_scale != 1:
            self.power_funcs = [lambda ws: f(ws) * self.power_scale for f in power_funcs]
        else:
            self.power_funcs = power_funcs

    def info(self, var, types):
        return var[np.asarray(types, np.int)]

    def hub_height(self, types=0):
        return self.info(self._hub_heights, types)

    def diameter(self, types=0):
        return self.info(self._diameters, types)

    def name(self, types=0):
        return self.info(self._names, types)

    def power(self, ws_i, type_i):
        return self.ct_power(ws_i, type_i)[1]

    def ct(self, ws_i, type_i):
        return self.ct_power(ws_i, type_i)[0]

    def ct_power(self, ws_i, type_i=0):
        if np.any(type_i != 0):
            CT = np.zeros_like(ws_i)
            P = np.zeros_like(ws_i)
            for t in np.unique(type_i):
                m = type_i == t
                CT[m] = self.ct_funcs[t](ws_i[m])
                P[m] = self.power_funcs[t](ws_i[m])
            return CT, P
        else:
            return self.ct_funcs[0](ws_i), self.power_funcs[0](ws_i)

    def plot(self, x, y, types=None, ax=None):
        import matplotlib.pyplot as plt
        if types is None:
            types = np.zeros_like(x)
        if ax is None:
            ax = plt.gca()
        markers = np.array(list("213v^<>o48spP*hH+xXDd|_"))

        from matplotlib.patches import Circle
        for i, (x_, y_, d) in enumerate(zip(x, y, self.diameter(types))):
            circle = Circle((x_, y_), d / 2, color='gray', alpha=.5)
            ax.add_artist(circle)
        for t, m in zip(np.unique(types), markers):
            ax.plot(np.asarray(x)[types == t], np.asarray(y)[types == t], '%sk' % m, label=self._names[int(t)])

        for i, (x_, y_, d) in enumerate(zip(x, y, self.diameter(types))):
            ax.annotate(i, (x_ + d / 2, y_ + d / 2), fontsize=7)
        ax.legend(loc=1)
        ax.axis('equal')


class OneTypeWindTurbines(WindTurbines):

    def __init__(self, name, diameter, hub_height, ct_func, power_func, power_unit):
        WindTurbines.__init__(self, [name], [diameter], [hub_height],
                              [lambda ws: ct_func(ws)],
                              [lambda ws: power_func(ws)],
                              power_unit)

    def power(self, ws_i, type_i=0):
        return self.ct_power(ws_i, type_i)[1]

    def ct(self, ws_i, type_i=0):
        return self.ct_power(ws_i, type_i)[0]


def cube_power(ws_cut_in, ws_cut_out, ws_rated, power_rated):
    def power_func(ws):
        ws = np.asarray(ws)
        power = np.zeros_like(ws, dtype=np.float)
        m = (ws > ws_cut_in) & (ws < ws_rated)
        power[m] = power_rated * ((ws[m] - ws_cut_in) / (ws_rated - ws_cut_in))**3
        power[(ws >= ws_rated) & (ws <= ws_cut_out)] = power_rated
        return power
    return power_func


def main():
    if __name__ == '__main__':

        wts = WindTurbines(names=['tb1', 'tb2'],
                           diameters=[80, 120],
                           hub_heights=[70, 110],
                           ct_funcs=[lambda ws: 8 / 9,
                                     lambda ws: 8 / 9],
                           power_funcs=[cube_power(ws_cut_in=3, ws_cut_out=25, ws_rated=12, power_rated=2000),
                                        cube_power(ws_cut_in=3, ws_cut_out=25, ws_rated=12, power_rated=3000)],
                           power_unit='kW')

        ws = np.arange(25)
        import matplotlib.pyplot as plt
        ct, power = wts.ct_power(ws)
        plt.plot(ws, power, label=wts.name(0))
        plt.legend()
        plt.show()

        wts.plot([0, 100], [0, 100], [0, 1])
        plt.xlim([-50, 150])
        plt.ylim([-50, 150])
        plt.show()


main()
