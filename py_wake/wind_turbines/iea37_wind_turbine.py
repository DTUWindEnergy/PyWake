from py_wake.examples.data.iea37.iea37_reader import read_iea37_windturbine
from py_wake.wind_turbines import OneTypeWindTurbines
from py_wake.examples.data.iea37 import iea37_path


class IEA37_WindTurbines(OneTypeWindTurbines):
    def __init__(self, yaml_filename):
        name, hub_height, diameter, ct_func, power_func = read_iea37_windturbine(yaml_filename)
        super().__init__(name, diameter, hub_height, ct_func, power_func)


def main():
    if __name__ == '__main__':
        wt = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
        print(wt.diameter(0))
        print(wt.hub_height(0))


main()
