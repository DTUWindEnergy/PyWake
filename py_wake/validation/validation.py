from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.site._site import UniformSite
from py_wake.wind_turbines import OneTypeWindTurbines
from py_wake.flow_map import XYGrid
from matplotlib.pyplot import cm
import numpy as np
from py_wake.validation.validation_lib import data_path, integrate_velocity_deficit_arc, GaussianFilter, sigma_hornsrev
import os
import matplotlib.pyplot as plt
from py_wake.deficit_models.noj import NOJDeficit
import xarray as xr
from py_wake.validation.lillgrund import SWT2p3_93_65
from py_wake.validation.ecn_wieringermeer import N80
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction, PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine


class ValidationSite(UniformSite):
    """Dummy site, used when instantiating the WindFarmModels to validadate.
    Will be replaced during the validation"""

    def __init__(self):
        UniformSite.__init__(self, p_wd=[1], ti=0.075)


class ValidationWindTurbines(OneTypeWindTurbines):
    """Dummy wind turbine, used when instantiating the WindFarmModels to validadate.
    Will be replaced during the validation"""

    def __init__(self):
        WindTurbine.__init__(self, name='ValidationWindTurbines',
                             diameter=0,
                             hub_height=0,
                             powerCtFunction=PowerCtFunction(input_keys=['ws'], power_ct_func=None,
                                                             power_unit='w')
                             )


class ValidationCase():
    def __init__(self, case_name, site, windTurbines):
        self.case_name = case_name
        self.site = site
        self.windTurbines = windTurbines
        self.U0 = site.ds.ws.item()
        self.model_result_dict = {}

    def get_result(self, model):
        if model not in self.model_result_dict:
            self.model_result_dict[model] = self.run(model)
        return self.model_result_dict[model]


class SingleWakeValidationCase(ValidationCase):
    def __init__(self, case_name, site, windTurbines, xD, x):
        super().__init__(case_name, site, windTurbines)
        self.xD = xD
        self.x = x
        self.U0 = site.ds.ws.item()

        # LES, based on EllipSys3D AD
        # RANS,  based on EllipSys3D AD k-epsilon-fP
        def load_dat(m, x):
            if x % 1 == 0:
                x = int(x)
            f = data_path + case_name + '_%s_%sD.dat' % (m, str(x).replace('.', 'p'))
            if os.path.isfile(f):
                return np.genfromtxt(f)
            return np.zeros((0, 4))
        self.meas, self.LES, self.RANS = [[load_dat(m, x) for x in xD] for m in ['data', 'LES', 'RANS']]

        if case_name == 'Wieringermeer-West':
            for m in self.meas:
                m[:, 0] -= 315  # subtract middle wind direction
                m[:, 1:3] /= self.U0  # normalize with U0
        elif case_name == 'Wieringermeer-East':
            for m in self.meas:
                m[:, 0] -= 31  # subtract middle wind direction
                m[:, 1:3] /= self.U0  # normalize with U0
        elif case_name == 'Nordtank-500':
            # remove 2nd column and 4th column with 74
            self.meas = [np.concatenate([m[:, :1], m[:, 2:4], m[:, :1] * 0 + 74], 1) for m in self.meas]
        elif case_name == 'Nibe':
            self.meas = [m[np.searchsorted(m[:, 0], -30):np.searchsorted(m[:, 0], 30)] for m in self.meas]

    def run(self, windFarmModel):
        windFarmModel.site = self.site
        windFarmModel.windTurbines = self.windTurbines
        fm = windFarmModel([0], [0]).flow_map(XYGrid(y=-self.x, x=0))
        wd = (fm.wd.values + 180) % 360 - 180
        ws = (fm.WS_eff / fm.ws).squeeze()
        return wd, ws

    @staticmethod
    def from_case_dict(name, case_dict):
        site = UniformSite(p_wd=[1],
                           ti=case_dict['TItot'] / 0.8,
                           ws=case_dict['U0'])
        site.default_wd = np.linspace(-30, 30, 61) % 360
        windTurbines = WindTurbine(name="",
                                   diameter=case_dict['D'],
                                   hub_height=case_dict['zH'],
                                   powerCtFunction=PowerCtFunction(
                                       input_keys=['ws'],
                                       power_ct_func=lambda ws, run_only, ct=case_dict['CT']: ws * 0 + ct,
                                       power_unit='w'))
        xD = case_dict['xDown']
        x = xD * case_dict['sDown']
        return SingleWakeValidationCase(name, site, windTurbines, xD, x)

    def plot_ref(self, axes, cLES='b', cRANS='g'):
        for ax, LES, RANS, meas in zip(axes, self.LES, self.RANS, self.meas):
            ax.fill_between(LES[:, 0], LES[:, 1] - LES[:, 2] / np.sqrt(LES[:, 3]),
                            LES[:, 1] + LES[:, 2] / np.sqrt(LES[:, 3]),
                            color=cLES, alpha=0.2, label='LES')
            ax.plot(RANS[:, 0], RANS[:, 1], color=cRANS, linewidth=2, label='RANS', linestyle='dashdot')
            if meas.shape[1] == 4:
                ax.errorbar(meas[:, 0], meas[:, 1], yerr=meas[:, 2] / np.sqrt(meas[:, 3]),
                            color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4,
                            capsize=3,
                            label='Measurements')
            else:
                ax.scatter(meas[:, 0], meas[:, 1], color='k', marker='o', zorder=0, s=10)


class WindRosePlot():
    def load_data(self, data_type, case_name):
        file_path = data_path + case_name + '_%s_%s.dat' % (data_type, 'WFeff')
        if os.path.isfile(file_path):
            return np.genfromtxt(file_path, skip_header=True)

    def __call__(self, case, result_dict, cLES='b', cRANS='g', lw=2):
        '''Plot wind farm efficiency as function of wind direction'''
        ax = plt.figure(figsize=(15, 5)).gca()
        colors = cm.tab10(np.linspace(0, 1, len(result_dict)))  # @UndefinedVariable
        ref_data = {m: self.load_data(m, case.case_name) for m in ['WFdata', 'RANS', 'LES']}

        # Plot reference data
        for key, dat in ref_data.items():
            if dat is None:
                continue
            elif key == 'WFdata':
                ax.fill_between(dat[:, 0], dat[:, 1] - dat[:, 2], dat[:, 1] + dat[:, 2],
                                color='k', alpha=0.3, label='Measurements')
            elif key == 'RANS':
                ax.plot(dat[:, 0], dat[:, 1], color=cRANS, linewidth=lw, label='RANS')
                if dat.shape[1] == 2:
                    ax.plot(dat[:, 0], dat[:, 2], color=cRANS,
                            dashes=[5, 2], linewidth=lw, label='RANS (gaus avg)')
            elif key == 'LES':
                ax.plot(dat[:, 0], dat[:, 1], color=cLES, linewidth=lw, label='LES')

            for co, (wfm_name, (sim_res, ls)) in zip(colors, result_dict.items()):
                norm = len(sim_res.wt) * sim_res.windFarmModel.windTurbines.power(sim_res.ws)
                ax.plot(sim_res.wd, sim_res.Power.squeeze().sum('wt') / norm, color=co, lw=lw, label=wfm_name)
                # ax.plot(sim_res.wd, sim_res.PowerGA.squeeze().sum('wt') / norm, color=co, dashes=[5, 2], label=wfm_name)

        ax.grid(True)
        ax.set_ylabel('$P/P_{max}$')
        ax.set_xlabel('Wind direction [deg]')
        ax.set_title(case)
        ax.legend()


class RowPlot():
    def __init__(self, name, wd, wts):
        self.name = name
        self.wd = wd
        self.wts = wts

    def load_data(self, data_type, case_name):
        file_path = data_path + case_name + '_%s_wd%s_%s.dat' % (data_type, int(self.wd), self.name)
        if os.path.isfile(file_path):
            return np.genfromtxt(file_path, skip_header=True)

    def __call__(self, case, result_dict, cLES='b', cRANS='g', lw=2):
        '''Plot comparison along one row of turbines'''
        ax = plt.figure(figsize=(15, 5)).gca()
        colors = cm.tab10(np.linspace(0, 1, len(result_dict)))  # @UndefinedVariable
        ref_data = {m: self.load_data(m, case.case_name) for m in ['WFdata', 'RANS', 'LES']}
        for key, dat in ref_data.items():
            if key == 'WFdata':
                ax.errorbar(dat[:, 0], dat[:, 1] / dat[0, 1], yerr=dat[:, 2] / np.sqrt(dat[:, 3]),
                            color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4,
                            capsize=3, label='Measurements')
            elif key == 'RANS':
                ax.plot(dat[:, 0], dat[:, 1], color=cRANS, linewidth=lw,
                        label='RANS', marker='.')
                ax.plot(dat[:, 0], dat[:, 2], color=cRANS, dashes=[5, 2], linewidth=lw,
                        label='RANS (gaus avg)', marker='.')
        for co, (wfm_name, (sim_res, ls)) in zip(colors, result_dict.items()):
            wt_i = np.arange(len(self.wts)) + 1
            sim_res = sim_res[['Power', 'PowerGA']].sel(wd=(np.arange(-3, 4) + self.wd) % 360).mean('wd').squeeze()

            if case.case_name == 'Hornsrev1':
                Power, PowerGa = sim_res.to_array().values.reshape(2, 10, 8)[:, :, 1:6].sum(2)
                # Power, PowerGa = [P.reshape((10, 6)).mean(1) for P in [Power, PowerGa]]
                wt_i = np.arange(1, 11)
            else:
                sim_res = sim_res.sel(wt=[(v, 0)[int(np.isnan(v))] for v in self.wts])
                Power, PowerGa = [np.where(~np.isnan(np.array(self.wts)), arr, np.nan) for arr in sim_res.to_array()]

            ax.plot(wt_i, Power / Power[0], color=co, lw=lw, label=wfm_name, marker='.')

        ax.grid(axis='y')
        ax.set_ylabel('$P_i/P_1$ [-]')
        ax.set_xlabel('WT nr. [-]')
        ax.set_xticks(wt_i)
        ax.set_title("%s %s WD=%.1fdeg" % (case.case_name, self.name, self.wd))
        ax.legend()


class MultiWakeValidationCase(ValidationCase):
    def __init__(self, case_name, site, windTurbines, sigma, plots):
        super().__init__(case_name, site, windTurbines)
        self.sigma = sigma
        self.plots = plots

    @staticmethod
    def from_case_dict(name, case_dict):
        site = case_dict['site']
        site.initial
        return MultiWakeValidationCase(name, case_dict['site'], case_dict['wt'])

    def run(self, windFarmModel):
        windFarmModel.site = self.site
        windFarmModel.windTurbines = self.windTurbines
        x, y = self.site.initial_position.T
        sim_res = windFarmModel(x, y)

        powerGA = np.zeros(sim_res.Power.shape)
        for iAD in range(len(x)):
            powerGA[iAD, :, 0] = GaussianFilter(sim_res.Power.values[iAD, :, 0],
                                                np.arange(0, 360.0, 1),
                                                int(np.ceil(3 * self.sigma[iAD])), self.sigma[iAD])
        sim_res['PowerGA'] = xr.DataArray(powerGA, dims=['wt', 'wd', 'ws'])
        sim_res['PowerGA'].attrs['Description'] = 'Gaussian averaged power production [W]'
        return sim_res

    def plot(self, windFarmModel_dict):
        results = {k: (self.get_result(wfm), ls) for k, (wfm, ls) in windFarmModel_dict.items()}
        for plot in self.plots:
            plot(self, results)


class Validation():
    def __init__(self):
        self.windFarmModel_dict = {}

        # single wake cases
        swc = {
            'Wieringermeer-West': {'U0': 10.7, 'CT': 0.63, 'TItot': 0.08, 'D': 80.0, 'zH': 80.0,
                                               'xDown': np.array([2.5, 3.5, 7.5]), 'sDown': 80.0, 'location': 'onshore'},
            'Wieringermeer-East': {'U0': 10.9, 'CT': 0.63, 'TItot': 0.06, 'D': 80.0, 'zH': 80.0,
                                   'xDown': np.array([2.5, 3.5, 7.5]), 'sDown': 80.0, 'location': 'onshore'},
            'Nibe': {'U0': 8.5, 'CT': 0.89, 'TItot': 0.08, 'D': 40.0, 'zH': 45.0,
                     'xDown': np.array([2.5, 4, 7.5]), 'sDown': 40.0, 'location': 'onshore'},
            'Nordtank-500': {'U0': 7.45, 'CT': 0.70, 'TItot': 0.112, 'D': 41.0, 'zH': 36.0,
                                         'xDown': np.array([2, 5, 7.5]), 'sDown': 40.0, 'location': 'onshore'},
            'NREL-5MW_TIlow': {'U0': 8.0, 'CT': 0.79, 'TItot': 0.04, 'D': 126.0, 'zH': 90.0,
                               'xDown': np.array([2.5, 5, 7.5]), 'sDown': 126.0, 'location': 'offshore'},
            'NREL-5MW_TIhigh': {'U0': 8.0, 'CT': 0.79, 'TItot': 0.128, 'D': 126.0, 'zH': 90.0,
                                'xDown': np.array([2.5, 5, 7.5]), 'sDown': 126.0, 'location': 'onshore'}
        }
        self.single_wake_cases = [SingleWakeValidationCase.from_case_dict(k, v) for k, v in swc.items()]

        # multiwake cases
        from py_wake.examples.data.hornsrev1 import wt_x as wt_x_hr
        from py_wake.examples.data.hornsrev1 import wt_y as wt_y_hr
        from py_wake.validation.ecn_wieringermeer import wt_x as wt_x_w
        from py_wake.validation.ecn_wieringermeer import wt_y as wt_y_w
        from py_wake.validation.lillgrund import wt_x as wt_x_l
        from py_wake.validation.lillgrund import wt_y as wt_y_l

        def get_site(ti, ws, wt_x, wt_y):
            return UniformSite(p_wd=[1], ti=ti, ws=ws, initial_position=np.array([wt_x, wt_y]).T)
        hr_inner_rows = np.arange(80).reshape(10, 8)[:, 1:7].flatten().tolist()
        self.multi_wake_cases = [MultiWakeValidationCase('Wieringermeer',
                                                         site=get_site(
                                                             ti=0.096 / 0.8, ws=8.35, wt_x=wt_x_w, wt_y=wt_y_w),
                                                         windTurbines=N80(),
                                                         sigma=2.5 * np.ones(len(wt_x_w)),
                                                         plots=[RowPlot(name='Row', wd=275.0, wts=[0, 1, 2, 3, 4])]),
                                 MultiWakeValidationCase('Lillgrund',
                                                         site=get_site(ti=0.048, ws=9, wt_x=wt_x_l, wt_y=wt_y_l),
                                                         windTurbines=SWT2p3_93_65(),
                                                         sigma=3.3 * np.ones(len(wt_x_l)),
                                                         plots=[RowPlot('RowB', 222.0, [14, 13, 12, 11, 10, 9, 8, 7]),
                                                                RowPlot('RowD', 222.0, [
                                                                    29, 28, 27, np.nan, 26, 25, 24, 23]),
                                                                RowPlot('RowB', 207.0, [14, 13, 12, 11, 10, 9, 8, 7]),
                                                                RowPlot('RowD', 207.0, [
                                                                    29, 28, 27, np.nan, 26, 25, 24, 23]),
                                                                RowPlot('Row6', 120.0, [2, 9, 17, 25, 32, 37, 42, 46]),
                                                                RowPlot('Row4', 120.0, [
                                                                    4, 11, 19, np.nan, np.nan, 39, 44]),
                                                                RowPlot('Row6', 105.0, [2, 9, 17, 25, 32, 37, 42, 46]),
                                                                RowPlot('Row4', 105.0, [
                                                                    4, 11, 19, np.nan, np.nan, 39, 44]),
                                                                WindRosePlot(),
                                                                ]),
                                 MultiWakeValidationCase('Hornsrev1',
                                                         site=get_site(ti=0.056, ws=8, wt_x=wt_x_hr, wt_y=wt_y_hr),
                                                         windTurbines=HornsrevV80(),
                                                         sigma=sigma_hornsrev('vanderLaan', wt_x_hr, wt_y_hr),
                                                         plots=[RowPlot('InnerRowMean', 270.0, hr_inner_rows),
                                                                WindRosePlot(),
                                                                ])

                                 ]

    def add_windFarmModel(self, name, windFarmModel, line_style='-'):
        self.windFarmModel_dict[name] = (windFarmModel, line_style)

    def _init_plot(self, case):
        n_x = len(case.x)
        colors = cm.tab10(np.linspace(0, 1, len(self.windFarmModel_dict)))  # @UndefinedVariable
        fig, axes = plt.subplots(1, n_x, sharey=False, figsize=(5 * n_x, 5))
        fig.suptitle(case.case_name)
        for ax, xD in zip(axes, case.xD):
            ax.set_title('x/D = %.1f' % xD)
        return fig, axes, colors

    def _add_legend(self, fig):
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        # some use legend from ax with most labels
        axes = fig.axes
        ax = axes[np.argmax([len(ax.get_legend_handles_labels()[0]) for ax in axes])]
        fig.legend(*ax.get_legend_handles_labels())

    def plot_deficit_profile(self, cLES='b', cRANS='g'):
        for case in self.single_wake_cases:
            fig, axes, colors = self._init_plot(case)
            case.plot_ref(axes, cLES=cLES, cRANS=cRANS)
            for co, (wfm_name, (wfm, ls)) in zip(colors, self.windFarmModel_dict.items()):
                wd, ws_lst = case.get_result(wfm)
                for ax, ws in zip(axes, ws_lst):
                    ax.plot(wd, ws, color=co, linewidth=2, label=wfm_name, linestyle=ls)

            for ax, xD in zip(axes, case.xD):

                ax.set_xticks(np.arange(wd.min(), wd.max() + 10.0, 10.0))
                lim = [30, 20][int(xD > 7)]
                ax.set_xlim(-lim, lim)
                ax.set_ylim(None, ymax=1.1)

            axes[0].set_ylabel('$U/U_0$', rotation=0)
            axes[0].yaxis.labelpad = 20
            axes[1].set_xlabel('Relative wind direction [deg]')
            fig.tight_layout(rect=[0, 0, 1, 0.9])

            # some use legend from ax with most labels
            ax = axes[np.argmax([len(ax.get_legend_handles_labels()[0]) for ax in axes])]
            fig.legend(*ax.get_legend_handles_labels())

    def plot_integrated_deficit(self, cLES='b', cRANS='g'):
        '''
            Bar plot comparison of integrated momentum deficit predicted by models
            and reference data at different downstream locations
        '''
        for case in self.single_wake_cases:
            fig, axes, colors = self._init_plot(case)

            for ax, xD, LES, RANS, meas in zip(axes, case.xD, case.LES, case.RANS, case.meas):
                for ibar, (data, co, label) in enumerate(
                        [(meas, 'k', 'Measurements'), (LES, cLES, 'LES'), (RANS, cRANS, 'RANS')]):
                    if len(data) > 20:
                        int_vel_def = integrate_velocity_deficit_arc(data[:, 0], data[:, 1] * case.U0, xD, case.U0)
                        ax.bar(ibar, int_vel_def, width=0.5, color=co, edgecolor=co, label=label)
            for ibar, (co, (wfm_name, (wfm, ls))) in enumerate(zip(colors, self.windFarmModel_dict.items()), 3):
                wd, ws_lst = case.get_result(wfm)
                for ax, ws, xD in zip(axes, ws_lst, case.xD):
                    int_vel_def = integrate_velocity_deficit_arc(wd, ws * case.U0, xD, case.U0)
                    ax.bar(ibar, int_vel_def, width=0.5, color=co, edgecolor=co, label=wfm_name)
            for ax, xD in zip(axes, case.xD):
                ax.set_ylabel('Integrated velocity deficit')
                ax.get_xaxis().set_visible(False)
            self._add_legend(fig)

    def plot_multiwake_power(self):
        for case in self.multi_wake_cases:
            case.plot(self.windFarmModel_dict)


if __name__ == '__main__':
    import matplotlib
    matplotlib.rcParams.update({'figure.max_open_warning': 0})
    validation = Validation()
    site, windTurbines = ValidationSite(), ValidationWindTurbines()
    validation.add_windFarmModel("NOJ", PropagateDownwind(site, windTurbines, wake_deficitModel=NOJDeficit()), ':')
    validation.add_windFarmModel(
        "NOJ(k=0.04)",
        PropagateDownwind(site, windTurbines, wake_deficitModel=NOJDeficit(k=0.04)), ':')

    validation.plot_deficit_profile()
#     validation.plot_integrated_deficit()
#     plt.close('all')
    validation.plot_multiwake_power()

    plt.show()
