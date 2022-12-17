import pandas as pd
from py_wake.wind_turbines.power_ct_functions import PowerCtXr
from py_wake.wind_turbines._wind_turbines import WindTurbine
import xarray as xr
import numpy as np
from py_wake.examples.data import swt_dd_142_4100_noise
import os
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.utils.plotting import setup_plot


class SWT_DD_142_4100(WindTurbine):
    """Siemens SWT-DD-142, 4.1MW, including sound power levels

    Data extracted from Windpro and stored as netcdf in SWT-DD-142_4100.nc
    """

    def __init__(self):
        self.ds = ds = xr.load_dataset(os.path.dirname(swt_dd_142_4100_noise.__file__) + '/SWT-DD-142_4100.nc')
        power_ct = PowerCtXr(ds, 'kW')
        ip = GridInterpolator([ds.mode.values, ds.ws.values], ds.SoundPower.transpose('mode', 'ws', 'freq').values,
                              method=['nearest', 'linear'])

        def sound_power_level(ws, mode, **_):
            ws = np.atleast_1d(ws)
            mode = np.zeros_like(ws) + mode
            return ds.freq.values, ip(np.array([np.round(mode).flatten(), ws.flatten()]).T).reshape(ws.shape + (-1,))

        WindTurbine.__init__(self, name='SWT-DD-142', diameter=142, hub_height=109,
                             powerCtFunction=power_ct, sound_power_level=sound_power_level)


def make_netcdf():
    # temp function used to convert the data from excel to netcdf

    # Read in the turbine power, ct and noise curves retrieved from Windpro
    SWT = pd.read_excel(r'../../../../SWT-DD-142_4100.xlsx')  # Siemens turbine with different modes
    modes = np.arange(7)
    ws = SWT[SWT.Mode == 0]['WindSpeed'].values.astype(float)
    freqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

    SWT = SWT.rename(columns={'f8000Hz ': 'f8000Hz'})

    ds = xr.Dataset({'SoundPower': (('mode', 'freq', 'ws'),
                                    [[SWT[SWT.Mode == mode][f'f{freq}Hz'].values.astype(float) for freq in freqs]
                                     for mode in modes]),
                     **{k: (('mode', 'ws'),
                            [SWT[SWT.Mode == mode][k].values.astype(float) for mode in modes])
                        for k in ['LwaRef', 'Power', 'Ct']}},
                    coords={'ws': ws, 'freq': freqs, 'mode': modes})
    ds.to_netcdf('SWT-DD-142_4100.nc')


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        wt = SWT_DD_142_4100()

        ws = np.arange(3, 26)
        for m in range(7):
            plt.plot(ws, wt.power(ws, mode=m) / 1000, label=f'mode: {m}')
        setup_plot(xlabel='Wind speed [m/s]', ylabel='Power [kW]')

        plt.figure()
        for m in range(7):
            freq, sound_power = wt.sound_power_level(ws=10, mode=m)
            plt.plot(freq, sound_power[0], label=f'mode: {m}')
        setup_plot(xlabel='Frequency [Hz]', ylabel='Sound power [dB]')
        plt.show()


main()
