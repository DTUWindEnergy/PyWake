from py_wake.wind_turbines import WindTurbine
import numpy as np
from pathlib import Path
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
import inspect
from py_wake.wind_turbines.power_ct_functions import PowerCtSurrogate
from py_wake.wind_turbines.wind_turbine_functions import FunctionSurrogates
from py_wake.examples.data import example_data_path


class IEA34_130_PowerCtSurrogate(PowerCtSurrogate):
    def __init__(self, name, input_parser):
        surrogate_path = Path(__file__).parent / name
        PowerCtSurrogate.__init__(
            self,
            power_surrogate=TensorflowSurrogate(surrogate_path / "electrical_power", 'operating'),
            power_unit='W',
            ct_surrogate=TensorflowSurrogate(surrogate_path / 'thrust', 'operating'),
            input_parser=input_parser)

        ws_idx = self.function_surrogate_lst[0].input_channel_names.index('ws')
        self.ws_cutin = self.function_surrogate_lst[0].input_scaler.data_min_[ws_idx]  # .wind_speed_cut_in
        self.ws_cutout = self.function_surrogate_lst[0].input_scaler.data_max_[ws_idx]  # .wind_speed_cut_out
        ti_key = [k for k in list(inspect.signature(input_parser).parameters) if k[:2] == 'TI'][0]
        thrust_idle = PowerCtSurrogate._power_ct(self, np.array([self.ws_cutout]), **{ti_key: .1})[1] * 1000
        self.ct_idle = thrust_idle / (1 / 2 * 1.225 * (65**2 * np.pi) * self.ws_cutout**2)

    def _power_ct(self, ws, **kwargs):
        power, thrust = PowerCtSurrogate._power_ct(self, ws, **kwargs)
        ct = thrust * 1000 / (1 / 2 * 1.225 * (65**2 * np.pi) * ws**2)
        power[(ws < self.ws_cutin) | (ws > self.ws_cutout)] = 0
        ct[(ws < self.ws_cutin) | (ws > self.ws_cutout)] = self.ct_idle
        return power, ct


class TreeRegionLoadSurrogates(FunctionSurrogates):
    def __init__(self, function_surrogate_lst, input_parser):
        FunctionSurrogates.__init__(self, function_surrogate_lst, input_parser)
        self.ws_cutin = function_surrogate_lst[0][0].wind_speed_cut_in
        self.ws_cutout = function_surrogate_lst[0][0].wind_speed_cut_out

    def __call__(self, ws, run_only=slice(None), **kwargs):
        ws_flat = ws.ravel()
        x = self.get_input(ws=ws, **kwargs)
        x = np.array([self.fix_shape(v, ws).ravel() for v in x]).T

        def predict(fs):
            output = np.empty(len(x))
            for fs_, m in zip(fs, [ws_flat < self.ws_cutin,
                                   (self.ws_cutin <= ws_flat) & (ws_flat <= self.ws_cutout),
                                   ws_flat > self.ws_cutout]):
                if m.sum():
                    output[m] = fs_.predict_output(x[m], bounds='ignore')[:, 0]
            return output
        return [predict(fs).reshape(ws.shape) for fs in np.asarray(self.function_surrogate_lst)[run_only]]

    @property
    def output_keys(self):
        return [fs[0].output_channel_name for fs in self.function_surrogate_lst]

    @property
    def wohler_exponents(self):
        return [fs[0].wohler_exponent for fs in self.function_surrogate_lst]


class IEA34_130_Base(WindTurbine):
    def __init__(self, powerCtFunction, loadFunction):
        WindTurbine.__init__(self, 'IEA 3.4MW', diameter=130, hub_height=110,
                             powerCtFunction=powerCtFunction,
                             loadFunction=loadFunction)


class IEA34_130_1WT_Surrogate(IEA34_130_Base):

    def __init__(self):
        sensors = ['del_blade_flap', 'del_blade_edge', 'del_tower_bottom_fa', 'del_tower_bottom_ss',
                   'del_tower_top_torsion']
        surrogate_path = Path(example_data_path) / 'iea34_130rwt' / 'one_turbine'
        set_names = ['below_cut_in', 'operating', 'above_cut_out']
        loadFunction = TreeRegionLoadSurrogates(
            [[TensorflowSurrogate(surrogate_path / s, n) for n in set_names] for s in sensors],
            input_parser=lambda ws, TI_eff=.1, Alpha=0: [TI_eff, ws, Alpha])
        powerCtFunction = IEA34_130_PowerCtSurrogate(
            'one_turbine',
            input_parser=lambda ws, TI_eff, Alpha=0: [TI_eff, ws, Alpha])
        IEA34_130_Base.__init__(self, powerCtFunction=powerCtFunction, loadFunction=loadFunction)


class IEA34_130_2WT_Surrogate(IEA34_130_Base):
    def __init__(self):
        sensors = ['del_blade_flap']
        surrogate_path = Path(example_data_path) / 'iea34_130rwt' / 'two_turbines'
        set_names = ['below_cut_in', 'operating', 'above_cut_out']
        loadFunction = TreeRegionLoadSurrogates(
            [[TensorflowSurrogate(surrogate_path / s, n) for n in set_names] for s in sensors],
            input_parser=self.get_input)
        self.max_dist = loadFunction.function_surrogate_lst[0][0].input_scaler.data_max_[4]
        self.max_angle = loadFunction.function_surrogate_lst[0][0].input_scaler.data_max_[3]

        powerCtFunction = IEA34_130_PowerCtSurrogate(
            'two_turbines',
            input_parser=(lambda ws, TI=.1, Alpha=0, get_input=self.get_input:
                          get_input(ws, TI=TI, Alpha=Alpha, dw_ijl=np.array([0]), hcw_ijl=0)))
        IEA34_130_Base.__init__(self, powerCtFunction=powerCtFunction, loadFunction=loadFunction)

    def get_input(self, ws, dw_ijl, hcw_ijl, TI, Alpha=0):
        # ['ws','ti', 'shear', 'wdir', 'dist']
        dist = np.atleast_1d((np.hypot(dw_ijl, hcw_ijl) / 130))
        wd = np.atleast_1d(np.rad2deg(np.arctan2(hcw_ijl, dw_ijl)))
        unwaked = (dist == 0) | (dist > self.max_dist) | (np.abs(wd) > self.max_angle)
        dist[unwaked] = 20
        wd[unwaked] = 20
        return [ws, TI, Alpha, wd, dist]


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.hornsrev1 import Hornsrev1Site
        from py_wake.turbulence_models.stf import STF2017TurbulenceModel
        from py_wake import NOJ
        import matplotlib.pyplot as plt

        u = np.arange(3, 28, .1)
        # ===============================================================================================================
        # IEA34_130_1WT_Surrogate
        # ===============================================================================================================
        wt = IEA34_130_1WT_Surrogate()
        # plot power/ct curves
        ax1 = plt.gca()
        ax2 = plt.twinx()
        for ti in [0.01, .05, .1, .3]:
            power, ct = wt.power_ct(u, TI_eff=ti)
            ax1.plot(u, power / 1000, label=f'TI={ti}')
            ax2.plot(u, ct, '--')
        ax1.legend()
        ax1.set_ylabel('Power [kW]')
        ax2.set_ylabel('Ct')

        plt.figure()
        ax1 = plt.gca()
        ax2 = plt.twinx()
        for alpha in [-0.09, .1, .3, .49]:
            power, ct = wt.power_ct(u, TI_eff=.1, Alpha=alpha)
            ax1.plot(u, power / 1000, label=f'Alpha={alpha}')
            ax2.plot(u, ct, '--')
        ax1.legend()
        ax1.set_ylabel('Power [kW]')
        ax2.set_ylabel('Ct')

        # plot load curves
        sensors = wt.loadFunction.output_keys
        axes = [plt.figure().gca() for _ in sensors]
        for ti in [0.01, .05, .1, .3]:
            loads = wt.loads(u, TI_eff=ti)
            for ax, l in zip(axes, loads):
                ax.plot(u, l, label=f'TI={ti}')
        for alpha in [-0.09, .1, .3, .49]:
            loads = wt.loads(u, TI_eff=.1, Alpha=alpha)
            for ax, l in zip(axes, loads):
                ax.plot(u, l, '--', label=f'Alpha={alpha}')
        for ax, s in zip(axes, sensors):
            ax.set_title(s)
            ax.legend()

        # plot loads as function of wd and ws
        plt.figure()
        site = Hornsrev1Site()
        x, y = [0, 1000], [0, 0]
        sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())(x, y, ws=np.arange(6, 25), Alpha=.12)
        load_wd_averaged = sim_res.loads(normalize_probabilities=True, method='OneWT_WDAvg')
        loads = sim_res.loads(normalize_probabilities=True, method='OneWT')
        loads.DEL.isel(sensor=0, wt=0).plot()

        for s in load_wd_averaged.sensor:
            print(s.item(), load_wd_averaged.LDEL.sel(sensor=s, wt=0).item(), loads.LDEL.sel(sensor=s, wt=0).item())
        plt.show()

        # =======================================================================================================================
        # IEA34_130_2WTSurrogate
        # =======================================================================================================================
        wt = IEA34_130_2WT_Surrogate()
        # plot power/ct curves
        plt.figure()
        ax1 = plt.gca()
        ax2 = plt.twinx()
        for ti in [0.01, .05, .1, .3]:
            power, ct = wt.power_ct(u, TI=ti)
            ax1.plot(u, power, label=f'TI={ti}')
            ax2.plot(u, ct, '--')
        ax1.legend()
        ax1.set_ylabel('Power [kW]')
        ax2.set_ylabel('Ct')

        plt.figure()
        ax1 = plt.gca()
        ax2 = plt.twinx()
        for alpha in [-0.09, .1, .3, .49]:
            power, ct = wt.power_ct(u, TI=.1, Alpha=alpha)
            ax1.plot(u, power / 1000, label=f'Alpha={alpha}')
            ax2.plot(u, ct, '--')
        ax1.set_ylabel('Power [kW]')
        ax2.set_ylabel('Ct')
        ax1.legend()

        # plot load curves
        sensors = wt.loadFunction.output_keys
        axes = [plt.figure().gca() for _ in sensors]
        for ti in [0.01, .05, .1, .3]:
            loads = wt.loads(u, TI=ti, Alpha=.12, dw_ijl=0, hcw_ijl=0)
            for ax, l in zip(axes, loads):
                ax.plot(u, l, label=f'TI={ti}')
        for alpha in [-0.09, .1, .3, .49]:
            loads = wt.loads(u, TI=.1, Alpha=alpha, dw_ijl=np.array([1000]), hcw_ijl=0)
            for ax, l in zip(axes, loads):
                ax.plot(u, l, '--', label=f'Alpha={alpha}')
        for ax, s in zip(axes, sensors):
            ax.set_title(s)
            ax.legend()

        # plot loads as function of wd and ws
        plt.figure()
        site = Hornsrev1Site()
        x, y = [0, 1000], [0, 0]
        sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())(x, y, ws=np.arange(3, 28), Alpha=.12)
        loads = sim_res.loads(normalize_probabilities=True, method='TwoWT')
        loads.DEL.isel(sensor=0, wt=0).plot()

        for s in loads.sensor:
            print(s.item(), loads.LDEL.sel(sensor=s, wt=0).item())
        plt.show()


main()
