import numpy as np
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windturbine


class WindTurbines():

    def __init__(self, names, diameters, hub_heights, ct_func, power_func):
        self._names = names
        self._diameters = np.array(diameters)
        self._hub_heights = np.array(hub_heights)
        self.ct_func = ct_func
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
        markers = np.array(list("123v^<>.o48spP*hH+xXDd|_"))
        for t, m in zip(np.unique(types), markers):
            ax.plot(np.asarray(x)[types == t], np.asarray(y)[types == t], '%sk' % m, label=self._names[int(t)])

        for i, (x_, y_) in enumerate(zip(x, y)):
            ax.annotate(i, (x_ + 1, y_ + 1))
        plt.legend()
        plt.axis('equal')


class OneTypeWindTurbines(WindTurbines):

    def __init__(self, name, diameter, hub_height, ct_func, power_func):
        WindTurbines.__init__(self, [name], [diameter], [hub_height],
                              lambda _, ws: ct_func(ws),
                              lambda _, ws: power_func(ws))


def main():
    if __name__ == '__main__':
        def power(types, ws):
            rated = 2000 + (1000 * types)
            return np.minimum(np.maximum((ws - 4)**3, 0), rated)

        def ct(types, ws):
            return 8 / 9

        wts = WindTurbines(names=['tb1', 'tb2'],
                           diameters=[80, 120],
                           hub_heights=[70, 110],
                           ct_func=ct,
                           power_func=power)

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
