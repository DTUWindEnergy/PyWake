import numpy as np


class WindTurbines():

    def __init__(self, names, diameters, hub_heights, ct_func, power_func, power_unit):
        self._names = names
        self._diameters = np.array(diameters)
        self._hub_heights = np.array(hub_heights)
        self.ct_func = ct_func

        power_scale = {'w': 1, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}[power_unit.lower()]
        if power_scale != 1:
            self.power_func = lambda type_i, ws_i: power_func(type_i, ws_i) * power_scale
        else:
            self.power_func = power_func

    def info(self, var, types):
        return var[np.asarray(types, np.int)]

    def hub_height(self, types):
        return self.info(self._hub_heights, types)

    def diameter(self, types):
        return self.info(self._diameters, types)

    def name(self, types):
        return self.info(self._names, types)

    def ct_power(self, ws_i__, type_i):
        return self.ct_func(type_i, ws_i__), self.power_func(type_i, ws_i__)

    def plot(self, x, y, types=None, ax=None):
        import matplotlib.pyplot as plt
        if types is None:
            types = np.zeros_like(x)
        if ax is None:
            ax = plt.gca()
        markers = np.array(list("213v^<>o48spP*hH+xXDd|_"))

        from matplotlib.patches import Circle
        for i, (x_, y_, d) in enumerate(zip(x, y, self.diameter(types))):
            circle = Circle((x_, y_), d / 2, color='w', alpha=.5)
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
                              lambda _, ws: ct_func(ws),
                              lambda _, ws: power_func(ws),
                              power_unit)


def main():
    if __name__ == '__main__':
        def power(types, ws):
            """Calculate power in kW"""
            rated = 2000 + (1000 * types)
            return np.minimum(np.maximum((ws - 4)**3, 0), rated)

        def ct(types, ws):
            return 8 / 9

        wts = WindTurbines(names=['tb1', 'tb2'],
                           diameters=[80, 120],
                           hub_heights=[70, 110],
                           ct_func=ct,
                           power_func=power,
                           power_unit='kW')

        ws = np.arange(25)
        import matplotlib.pyplot as plt
        ct, power = wts.ct_power(ws, 0)
        plt.plot(ws, power, label=wts.name(0))
        plt.legend()
        plt.show()

        wts.plot([0, 100], [0, 100], [0, 1])
        plt.xlim([-50, 150])
        plt.ylim([-50, 150])
        plt.show()


main()
