import numpy as np
import os
from numpy import newaxis as na
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from scipy.interpolate import interp1d
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.superposition_models import SquaredSum
from py_wake.rotor_avg_models import RotorCenter
import xarray as xr

# -----------------------------------------------------
# Default values
# -----------------------------------------------------
data_path = os.path.dirname(__file__) + '/data/'   # path to reference data
cLES = 'b'            # line color for LES results
cRANS = 'g'           # line color for RANS
lw = 2.0              # line width for most resutls

# -----------------------------------------------------
# General functions
# -----------------------------------------------------


def integrate_velocity_deficit_arc(wd, U, R, U0):
    # Integrate velocity deficit
    # Input: velocity deficit defined on an arc
    #        as a function of the relative wd
    # Output: Estimate of a projected stream-wise
    #         velocity deficit, normalized by U0*Ly
    wd = wd / 180.0 * np.pi
    dy = R * (np.sin(wd[1:]) - np.sin(wd[:-1]))
    return np.sum((1 - U[:-1] / U0) * dy) / np.sum(dy)


def uniqueIndexes(l):
    seen = set()
    res = []
    for i, n in enumerate(l):
        if n not in seen:
            res.append(i)
            seen.add(n)
    return res


def uniqueLabels(all_handles, all_labels):
    uI = uniqueIndexes(all_labels)
    labels = [all_labels[i] for i in uI]
    handles = [all_handles[i] for i in uI]
    return handles, labels


def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return([start + step * i for i in range(n + 1)])
    else:
        return([start, stop])


def gauss(mu, sigma, x):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))


def sigmaVarDist(x, y, xref, yref):
    dist = np.hypot(x - xref, y - yref)
    sigma = np.hypot(0.00035 * dist + 2.1, 2.5)
    return sigma


# def GaussianFilter(y, wd, nwdGA, sigma):
#     # Gaussian filter for cyclic data, i.e. as function of wind direction
#     # y is a vector and a function of wd
#     # It is assumed that the spacing in wd is uniform
#     # yGA is the returned filtered y
#     # nwdGA is the amount of points that are used for the smoothing
#     wdDelta = wd[1] - wd[0]
#     # wdGArel is the smoothing operator
#     wdGArel = seq(- nwdGA * wdDelta, nwdGA * wdDelta, wdDelta)
#     ny = len(y)
#     nwd = len(wd)
#     assert ny == nwd, 'length of wd does not correspond with length of y'
#     yGA = np.zeros((1, ny))
#     int_gauss = 0
#     if(nwd * wdDelta == 360):
#         # Full wind rose
#         i1 = 0
#         i2 = nwd
#     else:
#         # Partial wind rose: cannot GA the first and last wind directions
#         i1 = nwdGA
#         i2 = nwd - nwdGA
#     for i in range(i1, i2):
#         j = 0
#         int_gauss = 0
#         for wdrel in wdGArel:
#             wdrelcor = wd[i] + wdrel
#             if (wdrelcor) < 0:
#                 wdrelcor = wdrelcor + 360.0
#             elif (wdrelcor) > 359.999:
#                 wdrelcor = wdrelcor - 360.0
#             int_gauss = int_gauss + gauss(0, sigma, wdrel)
#             yGA[0, i] = yGA[0, i] + y[wd == wdrelcor] * gauss(0, sigma, wdrel)
#             j = j + 1
#         yGA[0, i] = yGA[0, i] / int_gauss
#     return yGA


def GaussianFilter(y, wd, nwdGA, sigma):
    # Gaussian filter for cyclic data, i.e. as function of wind direction
    # y is a vector and a function of wd
    # It is assumed that the spacing in wd is uniform
    # yGA is the returned filtered y
    # nwdGA is the amount of points that are used for the smoothing
    wdDelta = wd[1] - wd[0]
    # wdGArel is the smoothing operator
    wdGArel = np.arange(-nwdGA, nwdGA + 1) * wdDelta  # seq(- nwdGA * wdDelta, nwdGA * wdDelta, wdDelta)
    ny = len(y)
    nwd = len(wd)
    assert ny == nwd, 'length of wd does not correspond with length of y'
    yGA = np.zeros((1, ny))
    if(nwd * wdDelta == 360):
        # Full wind rose
        i1 = 0
        i2 = nwd
    else:
        # Partial wind rose: cannot GA the first and last wind directions
        i1 = nwdGA
        i2 = nwd - nwdGA

    gauss_arr = gauss(0, sigma, wdGArel)
    for i, wdrelcor_arr in zip(range(i1, i2), (wd[range(i1, i2), na] + wdGArel) % 360):
        yGA[:, i] = np.sum(y[np.searchsorted(wd, wdrelcor_arr)] * gauss_arr)
    yGA[0, :] = yGA[0, :] / gauss_arr.sum()
    return yGA


def sigma_hornsrev(method, wt_x, wt_y, sigma_cnst=5.0):
    # A general standard deviation (std) is used to Gaussian average results of Power
    # for the Horns Rev 1 wind farm.
    # Three methods can be used:
    if method == 'constant':
        # Constant sigma of 5 deg
        sigma = np.zeros((80))
        sigma[:] = sigma_cnst
    elif method == 'Gaumond':
        # Gaumond et el. (2013): DOI: 10.1002/we.1625
        # Fitted a std per row using the power deficit at WT 2 from Fuga.
        sigma = np.zeros((10, 8))
        sigma[:, 0] = 7.4
        sigma[:, 1] = 7.0
        sigma[:, 2] = 6.2
        sigma[:, 3] = 5.8
        sigma[:, 4] = 5.4
        sigma[:, 5] = 5.0
        sigma[:, 6] = 4.5
        sigma[:, 7] = 4.8
        sigma = sigma.flatten()
    elif method == 'vanderLaan':
        # van der Laan et al. (2014): DOI: 10.1002/we.1804
        # Using empirical data to define a std based on the distance between
        # a wt and the location where the reference wind direction was measured (WT G2)
        sigma = np.zeros((80))
        WTref = 7  # WT G2
        for iAD in range(80):
            sigma[iAD] = sigmaVarDist(wt_x[iAD], wt_y[iAD], wt_x[int(WTref - 1)], wt_y[int(WTref - 1)])
    return sigma


def name(o):
    return o.__class__.__name__


def deficit_linestyle(deficit_name):
    ls = 'solid'
    if deficit_name[:3] == 'NOJ':
        ls = 'dotted'
    elif deficit_name[:3] == 'GCL':
        ls = 'dashed'
    return ls

# -----------------------------------------------------
# Single wake plotting funcions
# -----------------------------------------------------


def plot_refdata(case_name, case, j, ax, linewidth=lw, cLES=cLES, cRANS=cRANS):
    '''
        Plot the single wake reference data for the specific case
    '''
    if case['xDown'][j] % 2 == 0 or (case['xDown'][j] + 1) % 2 == 0:
        xDownlabel = str(int(case['xDown'][j]))
    else:
        xDownlabel = str(case['xDown'][j]).replace('.', 'p')
    # References, based on field measurements
    if case_name == 'Wieringermeer-West' and j == 1:
        data = np.genfromtxt(data_path + case_name + '_data_' + xDownlabel + 'D.dat')
        ax.errorbar(data[:, 0] - 315, data[:, 1] / case['U0'], yerr=data[:, 2] / case['U0'] /
                    np.sqrt(data[:, 3]), color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4, label='Measurements')
    elif case_name == 'Wieringermeer-East' and j == 0:
        data = np.genfromtxt(data_path + case_name + '_data_' + xDownlabel + 'D.dat')
        ax.errorbar(data[:, 0] - 31, data[:, 1] / case['U0'], yerr=data[:, 2] / case['U0'] /
                    np.sqrt(data[:, 3]), color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4, label='Measurements')
    elif case_name == 'Nibe':
        # No standard deviation of the 10 min. available.
        data = np.genfromtxt(data_path + case_name + '_data_' + xDownlabel + 'D.dat')
        ax.scatter(data[:, 0], data[:, 1], color='k', marker='o', zorder=0, s=10)
    elif case_name == 'Nordtank-500' and j < 2:
        data = np.genfromtxt(data_path + case_name + '_data_' + xDownlabel + 'D.dat')
        ax.errorbar(data[:, 0], data[:, 2], yerr=data[:, 3] / np.sqrt(74.0),
                    color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4, label='Measurements')
    # LES, based on EllipSys3D AD
    LES = np.genfromtxt(data_path + case_name + '_LES_' + xDownlabel + 'D.dat')
    # Shaded area represent the standard error of the mean
    ax.fill_between(LES[:, 0], LES[:, 1] - LES[:, 2] / np.sqrt(LES[:, 3]),
                    LES[:, 1] + LES[:, 2] / np.sqrt(LES[:, 3]),
                    color=cLES, alpha=0.2, label='LES')
    # RANS,  based on EllipSys3D AD k-epsilon-fP
    RANS = np.genfromtxt(data_path + case_name + '_RANS_' + xDownlabel + 'D.dat')
    ax.plot(RANS[:, 0], RANS[:, 1], color=cRANS, linewidth=linewidth, label='RANS', linestyle='dashdot')
    return


def udef_refdata(case_name, case):
    '''
        Compute momentum deficit for the reference data
    '''
    nD = len(case['xDown'])
    Udef = np.zeros((nD, 3))
    Udef[:, :] = np.nan
    wds = case['wds']
    for j in range(nD):
        if case['xDown'][j] % 2 == 0 or (case['xDown'][j] + 1) % 2 == 0:
            xDownlabel = str(int(case['xDown'][j]))
        else:
            xDownlabel = str(case['xDown'][j]).replace('.', 'p')
        # References, based on field measurements
        if case_name == 'Wieringermeer-West' and j == 1:
            data = np.genfromtxt(data_path + case_name + '_data_' + xDownlabel + 'D.dat')
            fdata = interp1d(data[:, 0] - 315, data[:, 1])
            Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds), case['xDown'][j], case['U0'])
        elif case_name == 'Wieringermeer-East' and j == 0:
            data = np.genfromtxt(data_path + case_name + '_data_' + xDownlabel + 'D.dat')
            fdata = interp1d(data[:, 0] - 31, data[:, 1])
            Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds), case['xDown'][j], case['U0'])
        elif case_name == 'Nibe':
            # No standard deviation of the 10 min. available.
            data = np.genfromtxt(data_path + case_name + '_data_' + xDownlabel + 'D.dat')
            fdata = interp1d(data[:, 0], data[:, 1])
            Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds) * case['U0'], case['xDown'][j], case['U0'])
        elif case_name == 'Nordtank-500' and j < 2:
            data = np.genfromtxt(data_path + case_name + '_data_' + xDownlabel + 'D.dat')
        # LES, based on EllipSys3D AD
        LES = np.genfromtxt(data_path + case_name + '_LES_' + xDownlabel + 'D.dat')
        # RANS,  based on EllipSys3D AD k-epsilon-fP
        RANS = np.genfromtxt(data_path + case_name + '_RANS_' + xDownlabel + 'D.dat')
        Udef[j, 1] = integrate_velocity_deficit_arc(LES[:, 0], LES[:, 1] * case['U0'], case['xDown'][j], case['U0'])
        Udef[j, 2] = integrate_velocity_deficit_arc(RANS[:, 0], RANS[:, 1] * case['U0'], case['xDown'][j], case['U0'])
    return Udef


def modify_deficit_name(deficit_model):
    raw_name = "%s" % name(deficit_model)
    deficit_name = raw_name[:-7]
    if raw_name == 'BastankhahGaussianDeficit':
        k = deficit_model.k_ilk()[0, 0, 0]
        deficit_name += "(k={:.3f})".format(k)
    elif raw_name == 'NOJDeficit':
        deficit_name += "(k={:.3f})".format(deficit_model.a[1])
    return deficit_name


def modify_deficit_name_sw(deficit_model, TI):
    raw_name = "%s" % name(deficit_model)
    deficit_name = raw_name[:-7]

    if raw_name == 'BastankhahGaussianDeficit':
        k = deficit_model.k_ilk()[0, 0, 0]
        deficit_name += "(k={:.3f})".format(k)
    elif raw_name[-15:] == 'GaussianDeficit':
        k = deficit_model.k_ilk(TI)[0, 0, 0]
        deficit_name += "(k={:.3f})".format(k)
    elif raw_name == 'NOJDeficit':
        deficit_name += "(k={:.3f})".format(deficit_model.a[1])
    elif raw_name == 'NOJLocalDeficit':
        k = deficit_model.k_ilk(TI)[0, 0, 0]
        deficit_name += "(k={:.3f})".format(k)

    return deficit_name


def run_wms(swc, test_cases=['Wieringermeer-West',
                             'Wieringermeer-East',
                             'Nibe',
                             'Nordtank-500',
                             'NREL-5MW_TIlow',
                             'NREL-5MW_TIhigh'], deficit_models=[NOJDeficit(), BastankhahGaussianDeficit()],
            wds=np.linspace(-30, 30, 61)):
    '''
        Run the different wake models for the specified sites and output simulation results
    '''
    swc_out = {}
    for case in test_cases:
        swc_out[case] = swc[case]
        x_j = swc[case]['sDown'] * (np.dot(swc[case]['xDown'][:, na], np.cos(wds / 180.0 * np.pi)[na, :])).flatten()
        y_j = swc[case]['sDown'] * (np.dot(swc[case]['xDown'][:, na], np.sin(wds / 180.0 * np.pi)[na, :])).flatten()
        ii, jj = len(swc[case]['xDown']), len(wds)
        swc_out[case]['x'] = x_j.reshape(ii, jj)
        swc_out[case]['y'] = y_j.reshape(ii, jj)
        swc_out[case]['wds'] = wds
        swc_out[case]['deficit_models'] = []
        for model in deficit_models:
            # set up model
            tmp = {}
            wfm = PropagateDownwind(swc[case]['site'], swc[case]['wt'],
                                    model,
                                    superpositionModel=SquaredSum(),
                                    rotorAvgModel=RotorCenter(),
                                    turbulenceModel=STF2017TurbulenceModel())
            # simulation
            sim_res = wfm([0], [0], h=[100], wd=[270])
            lw_j, WS_eff_jlk, TI_eff_jlk = wfm._flow_map(x_j, y_j, np.ones_like(x_j) * 100, sim_res)
            deficit_name = modify_deficit_name_sw(wfm.wake_deficitModel, TI_eff_jlk)
            tmp['bar_label'] = modify_deficit_name(wfm.wake_deficitModel)
            tmp['WS_eff'] = WS_eff_jlk[:, 0, 0].reshape(ii, jj)
            tmp['TI_eff'] = TI_eff_jlk[:, 0, 0].reshape(ii, jj)
            swc_out[case][deficit_name] = tmp
            swc_out[case]['deficit_models'].append(deficit_name)

    return swc_out


def plot_single_wake(swc_out, lw=lw):
    '''
        Plot reference data and wake results for each test case
    '''
    for case in swc_out.keys():
        jj = len(swc_out[case]['xDown'])
        color = cm.tab10(np.linspace(0, 1, len(swc_out[case]['deficit_models'])))  # @UndefinedVariable

        fig, ax = plt.subplots(1, jj, sharey=False, figsize=(5 * jj, 5))
        fig.suptitle(case)
        handles, labels = [], []
        for j in range(jj):
            plot_refdata(case, swc_out[case], j, ax[j])
            co = 0
            for deficit_name in swc_out[case]['deficit_models']:
                ls = deficit_linestyle(deficit_name)
                ll, = ax[j].plot(swc_out[case]['wds'], swc_out[case][deficit_name]['WS_eff'][j] / swc_out[case]['U0'],
                                 color=color[co], linewidth=lw, label=deficit_name, linestyle=ls)
                co += 1
            ax[j].set_title('x/D = {:.1f}'.format(swc_out[case]['xDown'][j]))
            ax[j].set_xticks(np.arange(min(swc_out[case]['wds']), max(swc_out[case]['wds']) + 10.0, 10.0))
            if swc_out[case]['xDown'][j] < 7.0:
                ax[j].set_xlim(-30, 30)
            else:
                ax[j].set_xlim(-20, 20)
            ax[j].grid(True)
            ax[j].set_ylim(None, ymax=1.1)
            handles_tmp, labels_tmp = ax[j].get_legend_handles_labels()
            if len(handles_tmp) > len(handles):
                handles = handles_tmp
                labels = labels_tmp
        ax[0].set_ylabel('$U/U_0$', rotation=0)
        ax[0].yaxis.labelpad = 20
        ax[1].set_xlabel('Relative wind direction [deg]')
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(handles, labels)

    return


def plotbar_single_wake(swc_out, cLES=cLES, cRANS=cRANS):
    '''
        Bar plot comparison of integrated momentum deficit predicted by models
        and reference data at different downstream locations
    '''

    # Compute momentum deficit
    for case in swc_out.keys():
        swc_out[case]['udef_ref'] = udef_refdata(case, swc_out[case])
        jj = len(swc_out[case]['xDown'])
        for defict_name in swc_out[case]['deficit_models']:
            tmp = np.zeros(jj)
            for j in range(jj):
                tmp[j] = integrate_velocity_deficit_arc(swc_out[case]['wds'], swc_out[case][defict_name]['WS_eff'][j],
                                                        swc_out[case]['xDown'][j], swc_out[case]['U0'])
            swc_out[case][defict_name]['udef'] = tmp

    # Create bar plot of the integrated velocity deficit for all single wake cases
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ibar = 1
    names = []
    subnames = []
    lines = []
    color = cm.tab10(np.linspace(0, 1, len(swc_out[case]['deficit_models'])))  # @UndefinedVariable
    i = 0
    ymax = 0
    for case in swc_out.keys():
        ymax = max(ymax, np.nanmax(swc_out[case]['udef_ref']))
        jj = len(swc_out[case]['xDown'])
        for j in range(jj):
            ldata, = ax.bar(ibar, swc_out[case]['udef_ref'][j, 0], width=0.5,
                            color='k', edgecolor='k', label='Measurements')  # Data
            ibar += 1
            lLES, = ax.bar(ibar, swc_out[case]['udef_ref'][j, 1], width=0.5,
                           color=cLES, edgecolor=cLES, label='LES')  # LES
            ibar += 1
            lRANS, = ax.bar(ibar, swc_out[case]['udef_ref'][j, 2], width=0.5,
                            color=cRANS, edgecolor=cRANS, label='RANS')  # RANS
            ibar += 1

            co = 0
            for deficit_name in swc_out[case]['deficit_models']:
                ymax = max(ymax, swc_out[case][deficit_name]['udef'][j])
                l1, = ax.bar(ibar, swc_out[case][deficit_name]['udef'][j], width=0.5, color=color[co],
                             edgecolor=color[co], label=swc_out[case][deficit_name]['bar_label'])
                ibar += 1
                lines.append(l1)
                co += 1

            ibar += 1
            if j < jj - 1:
                ax.plot([ibar, ibar], [0, 0.35], ':k')
            else:
                ax.plot([ibar, ibar], [0, 0.35], '--k', dashes=[5, 2])
            ibar += 1
            subnames.append(str(swc_out[case]['xDown'][j]))
        # ibar = ibar +1
        names.append('Case ' + str(i + 1))
        i += 1
    ibar = ibar - 1
    ax.set_xticks(np.linspace(0.5 * ibar / i, ibar - 0.5 * ibar / i, i))
    ax.set_xticklabels(names)
    ax.set_xlim(0, ibar)
    ax.set_ylim(0, 1.05 * ymax)
    ax.tick_params(axis='x', direction='out')
    ax2 = ax.twiny()
    ax2.set_xticks(np.linspace(0.5 * ibar / (jj * i), ibar -
                               0.5 * ibar / (jj * i), jj * i))
    ax2.set_xticklabels(subnames)
    ax2.set_xlim(0, ibar)
    ax2.tick_params(axis='x', direction='in', pad=-15)
    ax.set_title('$x/D$')
    ax.set_ylabel('Integrated velocity deficit')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    all_handles, all_labels = ax.get_legend_handles_labels()
    handles, labels = uniqueLabels(all_handles, all_labels)
    fig.legend(handles, labels)
    return


# -----------------------------------------------------
# Wind farm plotting funcions
# -----------------------------------------------------

def get_setup_name(deficit_setup):
    setup_name = []
    i = 0
    for method in deficit_setup.keys():
        if method == 'deficit_model':
            setup_name.append(modify_deficit_name(deficit_setup[method]))
        else:
            setup_name.append(name(deficit_setup[method]))
        i += 1
    return "%s" % ("-".join(setup_name))


def run_wfm(mwc, test_cases=['Wieringermeer', 'Lillgrund', 'Hornsrev1'],
            deficit_setups=[{'deficit_model': NOJDeficit(),
                             'superpositionModel': SquaredSum(),
                             'rotorAvgModel': RotorCenter(),
                             'turbulenceModel': STF2017TurbulenceModel()}],
            gaussian_filter=True):
    '''
        Evaluate wake models for the different test cases
    '''
    mwc_out = {}

    for i in range(len(deficit_setups)):
        deficit_setups[i]['setup_name'] = get_setup_name(deficit_setups[i])

    for case in test_cases:
        mwc_out[case] = mwc[case]
        mwc_out[case]['deficit_setups'] = deficit_setups
        # mwc_out[case]['deficit_models'] = []
        if case == 'Wieringermeer':
            sigma = 2.5 * np.ones((len(mwc[case]['wt_x'])))
        elif case == 'Lillgrund':
            sigma = 3.3 * np.ones((len(mwc[case]['wt_x'])))
        elif case == 'Hornsrev1':
            sigma = sigma_hornsrev('vanderLaan', mwc[case]['wt_x'], mwc[case]['wt_y'])

        for i in range(len(deficit_setups)):
            # set up model
            wfm = PropagateDownwind(mwc[case]['site'], mwc[case]['wt'],
                                    deficit_setups[i]['deficit_model'],
                                    superpositionModel=deficit_setups[i]['superpositionModel'],
                                    rotorAvgModel=deficit_setups[i]['rotorAvgModel'],
                                    turbulenceModel=deficit_setups[i]['turbulenceModel'])
            # simulation
            sim_res = wfm(mwc[case]['wt_x'], mwc[case]['wt_y'], ws=mwc[case]['U0'])
            # Gaussian averaging
            if gaussian_filter:
                powerGA = np.zeros(sim_res.Power.shape)
                for iAD in range(len(mwc[case]['wt_x'])):
                    powerGA[iAD, :, 0] = GaussianFilter(sim_res.Power.values[iAD, :, 0],
                                                        np.arange(0, 360.0, 1),
                                                        int(np.ceil(3 * sigma[iAD])), sigma[iAD])
                sim_res['PowerGA'] = xr.DataArray(powerGA, dims=['wt', 'wd', 'ws'])
                sim_res['PowerGA'].attrs['Description'] = 'Gaussian averaged power production [W]'

            mwc_out[case][deficit_setups[i]['setup_name']] = sim_res
            mwc_out[case][deficit_setups[i]['setup_name']]['gaussian_filter'] = gaussian_filter

    return mwc_out


def load_data(data_path, keys, case, name):
    dat = {}
    for key in keys:
        file_path = data_path + case + '_' + key + '_' + name + '.dat'
        if os.path.isfile(file_path):
            dat[key] = np.genfromtxt(file_path, skip_header=True)
    return dat


def plot_windrose(case, mwc_out, cLES=cLES, cRANS=cRANS, lw=lw):
    '''
        Plot wind farm efficiency as function of wind direction
    '''
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    color = cm.tab10(np.linspace(0, 1, len(mwc_out[case]['deficit_setups'])))
    # Get reference data
    keys = ['WFdata', 'RANS', 'LES']
    dat = load_data(data_path, keys, case, 'WFeff')
    # Plot reference data
    for key in dat.keys():
        if key == 'WFdata':
            ldata = ax.fill_between(dat[key][:, 0], dat[key][:, 1] -
                                    dat[key][:, 2], dat[key][:, 1] + dat[key][:, 2],
                                    color='k', alpha=0.3, label='Measurements')
        if key == 'RANS':
            lRANS1, = ax.plot(dat[key][:, 0], dat[key][:, 1], color=cRANS, linewidth=lw, label='RANS')
            if dat[key].shape[1] == 2:
                lRANS2, = ax.plot(dat[key][:, 0], dat[key][:, 2], color=cRANS, dashes=[5, 2], linewidth=lw, label='GA')

        if key == 'LES':
            ax.plot(dat[key][:, 0], dat[key][:, 1], color=cLES, linewidth=lw, label='LES')

    norm = len(mwc_out[case]['wt_x']) * mwc_out[case]['wt'].power(mwc_out[case]['U0'])
    for i in range(len(mwc_out[case]['deficit_setups'])):
        name = mwc_out[case]['deficit_setups'][i]['setup_name']
        WFeff = mwc_out[case][name].Power.values[:, :, 0].sum(axis=0) / norm
        l1, = ax.plot(mwc_out[case][name].wd, WFeff, lw=lw, color=color[i], label=name)
        if mwc_out[case][name]['gaussian_filter']:
            WFeffGA = mwc_out[case][name].PowerGA.values[:, :, 0].sum(axis=0) / norm
            l1, = ax.plot(mwc_out[case][name].wd, WFeffGA, lw=lw, color=color[i], dashes=[5, 2])

    ax.grid(True)
    ax.set_ylabel('$P/P_{max}$')
    ax.set_xlabel('Wind direction [deg]')
    ax.set_title(case)
    ax.legend()


def plot_rows(case, mwc_out, plot, cLES=cLES, cRANS=cRANS, lw=lw):
    '''
        Plot comparison along one row of turbines
    '''
    # Get reference data
    keys = ['WFdata', 'RANS', 'LES']
    for i in range(len(keys)):
        keys[i] += '_wd' + str(int(plot['wd']))
    dat = load_data(data_path, keys, case, plot['name'])

    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    color = cm.tab10(np.linspace(0, 1, len(mwc_out[case]['deficit_setups'])))
    for key in dat.keys():
        if key[:6] == 'WFdata':
            ldata = ax.errorbar(dat[key][:, 0], dat[key][:, 1] / dat[key][0, 1], yerr=dat[key][:, 2] / np.sqrt(dat[key][:, 3]),
                                color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4,
                                label='Measurements')
        elif key[:4] == 'RANS':
            lRANS1, = ax.plot(dat[key][:, 0], dat[key][:, 1], color=cRANS, linewidth=lw,
                              label='RANS')
            lRANS2, = ax.plot(dat[key][:, 0], dat[key][:, 2], color=cRANS, dashes=[5, 2], linewidth=lw,
                              label='GA')

    for i in range(len(mwc_out[case]['deficit_setups'])):
        name = mwc_out[case]['deficit_setups'][i]['setup_name']
        if case == 'Hornsrev1':
            # Linear average for 267-273 deg and reshape in WT rows and columns
            power_matrix = mwc_out[case][name].Power.values[:, 267:274, 0].mean(axis=1).reshape(10, 8)
            # Sum the innner rows
            py = np.linspace(1, 10, 10)
            pdat = power_matrix[:, 1:6].sum(axis=1)
            if mwc_out[case][name]['gaussian_filter']:
                powerGA_matrix = mwc_out[case][name].PowerGA.values[:, 267:274, 0].mean(axis=1).reshape(10, 8)
                pdatGA = powerGA_matrix[:, 1:6].sum(axis=1)
        else:
            wd = plot['wd']
            power_row = np.zeros((len(plot['wts']), 2))
            for j in range(len(plot['wts'])):
                if np.isnan(plot['wts'][j]):
                    power_row[j, :] = np.nan
                else:
                    power_row[j, 0] = mwc_out[case][name].Power.values[plot['wts']
                                                                       [j], int(wd) - 3:int(wd) + 4, 0].mean()
                    if mwc_out[case][name]['gaussian_filter']:
                        power_row[j, 1] = mwc_out[case][name].PowerGA.values[plot['wts']
                                                                             [j], int(wd) - 3:int(wd) + 4, 0].mean()
            py = np.linspace(1, len(plot['wts']), len(plot['wts']))
            pdat = power_row[:, 0]
        l1, = ax.plot(py, pdat / pdat[0], color=color[i], linewidth=lw, label=name)
        if mwc_out[case][name]['gaussian_filter']:
            pdatGA = power_row[:, 1]
            l2, = ax.plot(py, pdatGA / pdatGA[0], color=color[i], dashes=[5, 2], linewidth=lw)

    ax.grid(True)
    ax.set_ylabel('$P_i/P_1$ [-]')
    ax.set_xlabel('WT nr. [-]')
    ax.set_title("{} {} WD={:.0f}deg".format(case, plot['name'], plot['wd']))
    ax.legend()


def plot_wind_farm(mwc_out):
    '''
        Plot comparison for the wind farm
    '''
    for case in mwc_out.keys():
        for plot in mwc_out[case]['plots']:
            if plot['name'] == 'WFeff':
                plot_windrose(case, mwc_out)
            else:
                plot_rows(case, mwc_out, plot)
