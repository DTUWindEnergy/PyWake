from py_wake.wind_turbines import WindTurbine
import numpy as np
from pathlib import Path
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
import inspect
from py_wake.wind_turbines.power_ct_functions import PowerCtSurrogate
from py_wake.wind_turbines.wind_turbine_functions import FunctionSurrogates
from py_wake.examples.data import example_data_path
from py_wake.utils.model_utils import fix_shape
from autograd.numpy.numpy_boxes import ArrayBox
from py_wake.examples.data.dtu10mw import DTU10MW


class DTU10MW_PowerCtSurrogate(PowerCtSurrogate):
    def __init__(self, surrogate_path, input_parser):
        PowerCtSurrogate.__init__(
            self,
            power_surrogate=TensorflowSurrogate(surrogate_path / "Power", 'operating'),
            power_unit='kW',
            ct_surrogate=TensorflowSurrogate(surrogate_path / 'Ct', 'operating'),
            input_parser=input_parser)

        ws_idx = self.function_surrogate_lst[0].input_channel_names.index('U')
        self.ws_cutin = self.function_surrogate_lst[0].input_scaler.data_min_[ws_idx]  # .wind_speed_cut_in
        self.ws_cutout = self.function_surrogate_lst[0].input_scaler.data_max_[ws_idx]  # .wind_speed_cut_out
        ti_key = [k for k in list(inspect.signature(input_parser).parameters) if k[:2] == 'TI'][0]
        self.ct_idle = PowerCtSurrogate._power_ct(self, np.array([self.ws_cutout]), run_only=1, **{ti_key: .1})

    def _power_ct(self, ws, run_only, **kwargs):
        ws = np.atleast_1d(ws)
        m = (ws > self.ws_cutin) & (ws < self.ws_cutout)
        if any([isinstance(v, ArrayBox) for v in [ws] + list(kwargs.values())]):
            # look up all values to avoid item assignment which is not supported by autograd
            arr = PowerCtSurrogate._power_ct(self, ws, run_only=run_only, **kwargs)
            if run_only == 0:
                return np.where(m, arr, 0)
            else:
                return np.where(m, arr, self.ct_idle)
        else:
            # look up only needed values
            kwargs = {k: fix_shape(v, ws)[m] for k, v in kwargs.items()}
            arr_m = PowerCtSurrogate._power_ct(self, ws[m], run_only=run_only, **kwargs)
            if run_only == 0:
                power = np.zeros_like(ws, dtype=arr_m.dtype)
                power[m] = arr_m
                return power
            else:
                ct = np.full(ws.shape, self.ct_idle, dtype=arr_m.dtype)
                ct_m = arr_m
                ct[m] = ct_m
                return ct


class DTU10MW_Base(WindTurbine):
    load_sensors = ['Blade_root_edgewise_M_y', 'Blade_root_flapwise_M_x', 'Tower_top_tilt_M_x', 'Tower_top_yaw_M_z']
    set_names = ['operating']

    def __init__(self, powerCtFunction, loadFunction):
        WindTurbine.__init__(self, 'DTU 10MW', diameter=178.3, hub_height=119,
                             powerCtFunction=powerCtFunction,
                             loadFunction=loadFunction)
        self.loadFunction.output_keys = self.load_sensors


class DTU10MW_1WT_Surrogate(DTU10MW_Base):
    def __init__(self):
        surrogate_path = Path(example_data_path) / 'dtu10mw_surrogate' / 'one_turbine'
        function_surrogate_lst = [TensorflowSurrogate(surrogate_path / n, 'operating') for n in self.load_sensors]
        loadFunction = FunctionSurrogates(function_surrogate_lst=function_surrogate_lst,
                                          input_parser=lambda ws, TI_eff=.1, Alpha=0.2, yaw=0: [ws, TI_eff * 100, Alpha, yaw],
                                          )
        powerCtFunction = DTU10MW_PowerCtSurrogate(
            surrogate_path,
            input_parser=lambda ws, TI_eff=.1, Alpha=0.2, yaw=0: [ws, TI_eff * 100, Alpha, yaw])
        DTU10MW_Base.__init__(self, powerCtFunction=powerCtFunction, loadFunction=loadFunction)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.hornsrev1 import Hornsrev1Site
        from py_wake.turbulence_models.stf import STF2017TurbulenceModel
        from py_wake import NOJ
        import matplotlib.pyplot as plt
        plt.close('all')
        u = np.arange(3, 28, 1)
        # u = np.array([10, 13])
        # ===============================================================================================================
        # DTU10MW_1WT_Surrogate
        # ===============================================================================================================
        wt = DTU10MW_1WT_Surrogate()
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
            loads = wt.loads(u, TI_eff=.1, Alpha=alpha, yaw=0)
            for ax, l in zip(axes, loads):
                ax.plot(u, l, '--', label=f'Alpha={alpha}')
        for ax, s in zip(axes, sensors):
            ax.set_title(s)
            ax.legend()

        # plot loads as function of wd and ws
        plt.figure()
        site = Hornsrev1Site()
        x, y = [0, 1000], [0, 0]
        sim_res = NOJ(site, wt, turbulenceModel=STF2017TurbulenceModel())(x, y, ws=np.arange(3, 28), Alpha=.12, yaw=0)
        load_wd_averaged = sim_res.loads(normalize_probabilities=True, method='OneWT_WDAvg')
        loads = sim_res.loads(normalize_probabilities=True, method='OneWT')
        loads.DEL.isel(sensor=0, wt=0).plot()

        for s in load_wd_averaged.sensor:
            print(s.item(), load_wd_averaged.LDEL.sel(sensor=s, wt=0).item(), loads.LDEL.sel(sensor=s, wt=0).item())
        plt.show()


main()
