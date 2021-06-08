def main():
    if __name__ == '__main__':
        from py_wake import NOJ
        from py_wake import IEA37SimpleBastankhahGaussian, Fuga
        import py_wake
        import os
        import matplotlib.pyplot as plt
        from py_wake.examples.data.hornsrev1 import HornsrevV80, Hornsrev1Site
        from py_wake.examples.data.iea37._iea37 import IEA37Site

        LUT_path = os.path.dirname(py_wake.__file__) + '/tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'

        wt_x, wt_y = IEA37Site(16).initial_position.T

        windTurbines = HornsrevV80()
        site = Hornsrev1Site()

        wake_models = [NOJ(site, windTurbines), IEA37SimpleBastankhahGaussian(
            site, windTurbines), Fuga(LUT_path, site, windTurbines)]

        for wake_model in wake_models:

            # Calculate AEP
            sim_res = wake_model(wt_x, wt_y)

            # Plot wake map
            plt.figure(wake_model.__class__.__name__)
            plt.title('AEP: %.2f GWh' % sim_res.aep().sum())

            flow_map = sim_res.flow_map(wd=[0], ws=[9])
            flow_map.plot_wake_map()
            flow_map.plot_windturbines()

        plt.show()


main()
