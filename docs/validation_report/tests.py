import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from py_wake.aep_calculator import AEPCalculator
from py_wake.wind_turbines import OneTypeWindTurbines
from py_wake.wake_models import NOJ
from py_wake.wake_models import BastankhahGaussian
from py_wake.site import UniformSite
from py_wake.examples.data.hornsrev1 import wt_x as wt_x_hr
from py_wake.examples.data.hornsrev1 import wt_y as wt_y_hr
from py_wake.examples.data.hornsrev1 import HornsrevV80, Hornsrev1Site
from data.ecn_wieringermeer import wt_x as wt_x_w
from data.ecn_wieringermeer import wt_y as wt_y_w
from data.ecn_wieringermeer import N80
from data.lillgrund import wt_x as wt_x_l
from data.lillgrund import wt_y as wt_y_l
from data.lillgrund import SWT2p3_93_65, LillgrundSite
from copy import copy
from matplotlib.ticker import MaxNLocator
import os
import matplotlib.style
import matplotlib as mpl
from scipy.interpolate import interp1d


def p(ws):
    # Dummy power function for wind turbine
    p = 1.0
    return p


def Gau_k_from_Iu(Iu):
    # Calculate Gaussian wake model expansion parameter k from a fit with LES
    # doi:10.1088/1742-6596/625/1/012039
    if Iu < 0.065:
        k = 0.4 * 0.065
    elif 0.065 <= Iu and Iu <= 0.15:
        k = 0.4 * Iu + 0.004
    elif Iu > 0.15:
        k = 0.4 * 0.15
    # Remove rounding errors
    k = int(k * 10 ** 6) / 10 ** 6
    return k


def NOJ_k_from_location(location):
    # Calculate NOJ wake model expansion parameter k based on onshore and offshore location
    if location == 'offshore':
        k = 0.04
    elif location == 'onshore':
        k = 0.1
    else:
        print('Location', location, 'is undefined. Choose onshore of offshore.')
    return k


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


def GaussianFilter(y, wd, nwdGA, sigma):
    # Gaussian filter for cyclic data, i.e. as function of wind direction
    # y is a vector and a function of wd
    # It is assumed that the spacing in wd is uniform
    # yGA is the returned filtered y
    # nwdGA is the amount of points that are used for the smoothing
    wdDelta = wd[1] - wd[0]
    # wdGArel is the smoothing operator
    wdGArel = seq(- nwdGA * wdDelta, nwdGA * wdDelta, wdDelta)
    ny = len(y)
    nwd = len(wd)
    if(ny != nwd):
        print('###############################################################')
        print('ERROR: length of wd does not correspond with length of y')
        print('###############################################################')
        sys.exit()
    yGA = np.zeros((1, ny))
    int_gauss = 0
    if(nwd * wdDelta == 360):
        # Full wind rose
        i1 = 0
        i2 = nwd
    else:
        # Partial wind rose: cannot GA the first and last wind directions
        i1 = nwdGA
        i2 = nwd - nwdGA
    for i in range(i1, i2):
        j = 0
        int_gauss = 0
        for wdrel in wdGArel:
            wdrelcor = wd[i] + wdrel
            if (wdrelcor) < 0:
                wdrelcor = wdrelcor + 360.0
            elif (wdrelcor) > 359.999:
                wdrelcor = wdrelcor - 360.0
            int_gauss = int_gauss + gauss(0, sigma, wdrel)
            yGA[0, i] = yGA[0, i] + y[wd.index(wdrelcor)] * gauss(0, sigma, wdrel)
            j = j + 1
        yGA[0, i] = yGA[0, i] / int_gauss
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


def integrate_velocity_deficit_arc(wd, U, R, U0):
    # Integrate velocity deficit
    # Input: velocity deficit defined on an arc
    #        as a function of the relative wd
    # Output: Estimate of a projected stream-wise
    #         velocity deficit, normalized by U0*Ly
    UInt = 0
    wd = wd / 180.0 * np.pi
    Ly = 0
    for i in range(len(wd) - 1):
        dy = R * (np.sin(wd[i + 1]) - np.sin(wd[i]))
        Ly = Ly + dy
        UInt = UInt + (1 - U[i] / U0) * dy
    return UInt / Ly


def deficitPlotSingleWakeCases(SingleWakecases, site, linewidth, cLES, cRANS, colors):
    # Plot velocity deficit for each single wake case

    UdefCases = []
    for case in SingleWakecases:

        def ct(ws):
            # ct function for wind turbine
            ct = case['CT']
            return ct

        wt = OneTypeWindTurbines(name=case['name'],
                                 diameter=case['D'],
                                 hub_height=case['zH'],
                                 ct_func=ct,
                                 power_func=p,
                                 power_unit='W')

        if case['name'] == 'Nordtank-500':
            xDown = case['xDown'] * 40.0
        else:
            xDown = case['xDown'] * wt._diameters
        wds = np.linspace(-30, 30, 61)

        kNOJ = NOJ_k_from_location(case['location'])
        kGAU = Gau_k_from_Iu(case['TItot'] / 0.8)
        print(case['name'], case['TItot'], kNOJ, kGAU)
        wakemodels = [NOJ(site, wt, k=kGAU), BastankhahGaussian(site, wt, k=kGAU)]
        wake_ws = np.zeros((len(wakemodels), len(wds), len(xDown)))
        for k in range(len(wakemodels)):
            aep_calc = AEPCalculator(wakemodels[k])
            # The velocity deficit at an arc is calculated by running the iwake_map for each point.
            # because it is only possible to provide a x and y array that define a rectangular plane.
            # This should be improved, where one can use a list of points and run wake_map once.
            for i in range(len(wds)):
                for j in range(len(xDown)):
                    x = xDown[j] * np.cos(wds[i] / 180.0 * np.pi)
                    y = xDown[j] * np.sin(wds[i] / 180.0 * np.pi)
                    X, Y, WS_eff = aep_calc.wake_map(x_j=x,
                                                     y_j=y,
                                                     wt_x=[0.0],
                                                     wt_y=[0.0],
                                                     wd=[270.0],
                                                     ws=case['U0'])
                    wake_ws[k, i, j] = WS_eff
        lines = []
        fig, ax = plt.subplots(1, len(xDown), sharey=False, figsize=(3 * len(xDown), 3))

        Udef = np.zeros((len(xDown), len(wakemodels) + 3))
        Udef[:, :] = np.nan
        for j in range(len(xDown)):
            if case['xDown'][j] % 2 == 0 or (case['xDown'][j] + 1) % 2 == 0:
                xDownlabel = str(int(case['xDown'][j]))
            else:
                xDownlabel = str(case['xDown'][j]).replace('.', 'p')
            # References, based on field measurements
            if case['name'] == 'Wieringermeer-West' and j == 1:
                data = np.genfromtxt('data/' + case['name'] + '_data_' + xDownlabel + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0] - 315, data[:, 1] / case['U0'], yerr=data[:, 2] / case['U0'] /
                                       np.sqrt(data[:, 3]), color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4)
                fdata = interp1d(data[:, 0] - 315, data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds), case['xDown'][j], case['U0'])
            elif case['name'] == 'Wieringermeer-East' and j == 0:
                data = np.genfromtxt('data/' + case['name'] + '_data_' + xDownlabel + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0] - 31, data[:, 1] / case['U0'], yerr=data[:, 2] / case['U0'] /
                                       np.sqrt(data[:, 3]), color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4)
                fdata = interp1d(data[:, 0] - 31, data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds), case['xDown'][j], case['U0'])
            elif case['name'] == 'Nibe':
                # No standard deviation of the 10 min. available.
                data = np.genfromtxt('data/' + case['name'] + '_data_' + xDownlabel + 'D.dat')
                ldata = ax[j].scatter(data[:, 0], data[:, 1], color='k', marker='o', zorder=0, s=10)
                fdata = interp1d(data[:, 0], data[:, 1])
                Udef[j, 0] = integrate_velocity_deficit_arc(wds, fdata(wds) * case['U0'], case['xDown'][j], case['U0'])
            elif case['name'] == 'Nordtank-500' and j < 2:
                data = np.genfromtxt('data/' + case['name'] + '_data_' + xDownlabel + 'D.dat')
                ldata = ax[j].errorbar(data[:, 0], data[:, 2], yerr=data[:, 3] / np.sqrt(74.0),
                                       color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4)
            # LES, based on EllipSys3D AD
            LES = np.genfromtxt('data/' + case['name'] + '_LES_' + xDownlabel + 'D.dat')
            # Shaded area represent the standard error of the mean
            lLES = ax[j].fill_between(LES[:, 0], LES[:, 1] - LES[:, 2] / np.sqrt(LES[:, 3]),
                                      LES[:, 1] + LES[:, 2] / np.sqrt(LES[:, 3]), color=cLES, alpha=0.5)
            # RANS,  based on EllipSys3D AD k-epsilon-fP
            RANS = np.genfromtxt('data/' + case['name'] + '_RANS_' + xDownlabel + 'D.dat')
            lRANS, = ax[j].plot(RANS[:, 0], RANS[:, 1], color=cRANS, linewidth=linewidth)
            for k in range(len(wakemodels)):
                l1, = ax[j].plot(wds, wake_ws[k, :, j] / case['U0'], color=colors[k], linewidth=linewidth)
                lines.append(l1)
                if case['name'] == 'Nordtank-500':
                    title = '%s %g %s' % ('$x=', case['xDown'][j], 'D^*$')
                else:
                    title = '%s %g %s' % ('$x=', case['xDown'][j], 'D$')
                ax[j].set_title(title)
                ax[j].set_xticks(np.arange(min(wds), max(wds) + 10.0, 10.0))
            if case['xDown'][j] < 7.0:
                ax[j].set_xlim(-30, 30)
            else:
                ax[j].set_xlim(-20, 20)
            ax[j].grid(True)
            ax[j].set_ylim(None, ymax=1.1)
            # Momemtum deficit wake models
            for k in range(len(wakemodels)):
                Udef[j, 3 + k] = integrate_velocity_deficit_arc(wds, wake_ws[k, :, j], case['xDown'][j], case['U0'])
            Udef[j, 1] = integrate_velocity_deficit_arc(LES[:, 0], LES[:, 1] * case['U0'], case['xDown'][j], case['U0'])
            Udef[j, 2] = integrate_velocity_deficit_arc(
                RANS[:, 0], RANS[:, 1] * case['U0'], case['xDown'][j], case['U0'])
        ax[0].set_ylabel('$U/U_0$', rotation=0)
        ax[0].yaxis.labelpad = 20
        ax[1].set_xlabel('Relative wind direction [deg]')
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        NOJlabel = 'NOJ ($k=%g$)' % (kNOJ)
        GAUlabel = 'GAU ($k=%g$)' % (kGAU)
        if case['name'] in ['Nibe', 'Nordtank-500', 'Wieringermeer-West', 'Wieringermeer-East']:
            fig.legend((ldata, lLES, lRANS, lines[0], lines[1]), ('Data', 'LES', 'RANS', NOJlabel, GAUlabel),
                       ncol=5, loc='upper center', bbox_to_anchor=(0, 0, 1, 1), numpoints=1, scatterpoints=1)
        else:
            fig.legend((lLES, lRANS, lines[0], lines[1]), ('LES', 'RANS', NOJlabel, GAUlabel),
                       ncol=4, loc='upper center', bbox_to_anchor=(0, 0, 1, 1), numpoints=1, scatterpoints=1)
        filename = case['name'] + '.pdf'
        fig.savefig(filename)
        os.system('mv ' + filename + ' report/figures/')
        UdefCases.append(Udef)
    return UdefCases


def barPlotSingleWakeCases(SingleWakecases, UdefCases, cLES, cRANS, colors):
    # Create bar plot of the integrated velocity deficit for all single wake cases
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ibar = 1
    names = []
    subnames = []
    lines = []
    nDown = 3
    nWakeModels = 2
    for i in range(len(UdefCases)):
        for j in range(nDown):
            ldata, = ax.bar(ibar, UdefCases[i][j, 0], width=0.5, color='k', edgecolor='k')  # Data
            ibar = ibar + 1
            lLES, = ax.bar(ibar, UdefCases[i][j, 1], width=0.5, color=cLES, edgecolor=cLES)  # LES
            ibar = ibar + 1
            lRANS, = ax.bar(ibar, UdefCases[i][j, 2], width=0.5, color=cRANS, edgecolor=cRANS)  # RANS
            ibar = ibar + 1
            for k in range(nWakeModels):
                l1, = ax.bar(ibar, UdefCases[i][j, 3 + k], width=0.5, color=colors[k], edgecolor=colors[k])
                lines.append(l1)
                ibar = ibar + 1
            ibar = ibar + 1
            if j < nDown - 1:
                ax.plot([ibar, ibar], [0, 0.35], ':k')
            else:
                ax.plot([ibar, ibar], [0, 0.35], '--k', dashes=[5, 2])
            ibar = ibar + 1
            subnames.append(str(SingleWakecases[i]['xDown'][j]))
        #ibar = ibar +1
        names.append('Case ' + str(i + 1))
    ibar = ibar - 1
    ax.set_xticks(np.linspace(0.5 * ibar / len(UdefCases), ibar - 0.5 * ibar / len(UdefCases), len(UdefCases)))
    ax.set_xticklabels(names)
    ax.set_xlim(0, ibar)
    ax.tick_params(axis='x', direction='out')
    ax2 = ax.twiny()
    ax2.set_xticks(np.linspace(0.5 * ibar / (nDown * len(UdefCases)), ibar -
                               0.5 * ibar / (nDown * len(UdefCases)), nDown * len(UdefCases)))
    ax2.set_xticklabels(subnames)
    ax2.set_xlim(0, ibar)
    ax2.tick_params(axis='x', direction='in', pad=-15)
    ax.set_title('$x/D$')
    ax.set_ylabel('Integrated velocity deficit')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend((ldata, lLES, lRANS, lines[0], lines[1]), ('Data', 'LES', 'RANS', 'NOJ', 'GAU'),
               ncol=5, loc='upper center', bbox_to_anchor=(0, 0, 1, 1), numpoints=1, scatterpoints=1)
    filename = 'VelocityDeficit.pdf'
    fig.savefig(filename)
    os.system('mv ' + filename + ' report/figures/')


def deficitPlotWFCases(WFcases, linewidth, cLES, cRANS, colors):
    # Plot power deficit in wind turbine row cases or plot the WF efficiency
    for case in WFcases:

        kNOJ = NOJ_k_from_location(case['location'])
        kGAU = Gau_k_from_Iu(case['TItot'] / 0.8)
        print(case['name'], case['TItot'], kNOJ, kGAU)
        site = case['site']
        windTurbines = case['wt']
        wakemodels = [NOJ(site, windTurbines, k=kGAU), BastankhahGaussian(site, windTurbines, k=kGAU)]

        if case['name'] == 'Wieringermeer':
            sigma = np.zeros((len(case['wt_x'])))
            sigma[:] = 2.5
        elif case['name'] == 'Lillgrund':
            sigma = np.zeros((len(case['wt_x'])))
            sigma[:] = 3.3
        elif case['name'] == 'Hornsrev1':
            sigma = sigma_hornsrev('vanderLaan', case['wt_x'], case['wt_y'])  # Standard deviation in hornsrev

        power_models = []
        powerGA_models = []
        for k in range(len(wakemodels)):
            aep_calc = AEPCalculator(wakemodels[k])
            aep_calc._run_wake_model(case['wt_x'], case['wt_y'], ws=case['U0'])
            power_models.append(aep_calc.power_ilk)

            # Gaussian averaging
            powerGA = np.zeros((len(aep_calc.power_ilk[:, 0, 0]), len(
                aep_calc.power_ilk[0, :, 0]), len(aep_calc.power_ilk[0, 0, :])))
            for iAD in range(len(case['wt_x'])):
                powerGA[iAD, :, 0] = GaussianFilter(aep_calc.power_ilk[iAD, :, 0], np.arange(
                    0, 360.0, 1).tolist(), int(np.ceil(3 * sigma[iAD])), sigma[iAD])
            powerGA_models.append(powerGA)

        # Make a figure per wind direction and WT row or wind farm efficiency
        for plot in case['plots']:
            lines = []
            linesGA = []
            if plot['name'] == 'WFeff':
                data = np.genfromtxt('data/' + case['name'] + '_WFdata_' + plot['name'] + '.dat', skip_header=True)
                RANS = np.genfromtxt('data/' + case['name'] + '_RANS_' + plot['name'] + '.dat', skip_header=True)
                fig = plt.figure(figsize=(9, 6))
                rect = [0.1, 0.1, 0.8, 0.7]
                ax_polar = fig.add_axes(rect, polar=True)
                ax_layout = fig.add_axes(rect, frameon=False)
                norm = len(case['wt_x']) * case['wt'].power(case['U0'])
                WFeffWakeModels = []
                for k in range(len(wakemodels)):
                    WFeff = power_models[k][:, :, 0].sum(axis=0) / norm
                    WFeffGA = powerGA_models[k][:, :, 0].sum(axis=0) / norm
                    l1, = ax_polar.plot(np.append(np.linspace(0, 359, 360), 0) / 180.0 * np.pi,
                                        np.append(WFeff, WFeff[0]), color=colors[k], linewidth=linewidth)
                    l2, = ax_polar.plot(np.append(np.linspace(0, 359, 360), 0) / 180.0 * np.pi,
                                        np.append(WFeffGA, WFeffGA[0]), color=colors[k], dashes=[5, 2], linewidth=linewidth)
                    lines.append(l1)
                    linesGA.append(l2)
                    print('WFeff', 'k', WFeff.mean())
                    WFeffWakeModels.append(WFeff.mean())
                datap = np.append(data[:, :], [data[0, :]], axis=0)
                ldata = ax_polar.fill_between(datap[:, 0] / 180.0 * np.pi, datap[:, 1] -
                                              datap[:, 2], datap[:, 1] + datap[:, 2], color='k', alpha=0.3)
                lRANS1, = ax_polar.plot(np.append(RANS[:, 0], RANS[0, 0]) / 180.0 * np.pi,
                                        np.append(RANS[:, 1], RANS[0, 1]), color=cRANS, linewidth=linewidth)
                lRANS2, = ax_polar.plot(np.append(RANS[:, 0], RANS[0, 0]) / 180.0 * np.pi,
                                        np.append(RANS[:, 2], RANS[0, 2]), color=cRANS, dashes=[5, 2], linewidth=linewidth)
                print('WFeff', 'RANS', RANS[:, 1].mean(), 'DATA', data[:, 1].mean())

                ax_polar.set_theta_zero_location("N")
                ax_polar.set_theta_direction(-1)
                ax_polar.set_rmin(0.0)
                ax_polar.set_rmax(1.0)
                ax_polar.set_rlabel_position(340.0)

                scale = 0.15
                xWTc = 0.5 * (np.array(case['wt_x']).max() + np.array(case['wt_x']).min())
                yWTc = 0.5 * (np.array(case['wt_y']).max() + np.array(case['wt_y']).min())
                xWT = (np.array(case['wt_x']) - xWTc) / case['wt'].diameter() * scale
                yWT = (np.array(case['wt_y']) - yWTc) / case['wt'].diameter() * scale

                ax_layout.set_xlim(xWT.min() / scale, xWT.max() / scale)
                ax_layout.set_ylim(yWT.min() / scale, yWT.max() / scale)
                ax_layout.set_xlim(xWT.min() / scale, xWT.max() / scale)
                ax_layout.set_ylim(yWT.min() / scale, yWT.max() / scale)

                ax_layout.set_aspect(1)
                ax_layout.scatter(xWT, yWT, s=2, color='r')
                ax_layout.set_xticks([])
                ax_layout.set_yticks([])

                filename = case['name'] + '_' + plot['name'] + '.pdf'

                # Create a latex table
                f = open('report/' + case['name'] + '_' + plot['name'] + '.tex', 'w')
                f.write('\\begin{tabular}{lcccc}\n')
                f.write('\hline\n')
                f.write('		     & Measurement data & RANS & NOJ  &   GAU  \\\\ \n')
                f.write('\hline\n')
                # Estimate uncertainty in measured wind farm efficiency assuming that we
                # have the same number of samples per wd
                dataWFunc = np.sqrt(sum(data[:, 2] ** 2) / len(data[:, 1]))
                f.write('%s %4.2f %s %4.3f %s %4.2f' % ('Wind farm efficiency & $',
                                                        data[:, 1].mean(), '\\pm', dataWFunc, '$ & ', RANS[:, 1].mean()))
                for k in range(len(wakemodels)):
                    f.write('%s %4.2f' % (' & ', WFeffWakeModels[k]))
                f.write('\\\\ \n')
                f.write('%s %4.0f' % ('Relative error [\\%] & - & ', 100.0 *
                                      (RANS[:, 1].mean() - data[:, 1].mean()) / data[:, 1].mean()))
                for k in range(len(wakemodels)):
                    f.write('%s %4.0f' % (' & ', 100.0 * (WFeffWakeModels[k] - data[:, 1].mean()) / data[:, 1].mean()))
                f.write('\\\\ \n')
                f.write('\hline\n')
                f.write('\end{tabular}\n')
                f.close()
            else:
                wd = plot['wd']
                data = np.genfromtxt('data/' + case['name'] + '_WFdata_wd' +
                                     str(int(wd)) + '_' + plot['name'] + '.dat', skip_header=True)
                RANS = np.genfromtxt('data/' + case['name'] + '_RANS_wd' +
                                     str(int(wd)) + '_' + plot['name'] + '.dat', skip_header=True)
                fig, ax = plt.subplots(1, 1, sharey=False, figsize=(9, 4))
                ax_layout = fig.add_axes([0.7, 0.3, 0.25, 0.25], frameon=True)
                for k in range(len(wakemodels)):
                    if case['name'] == 'Hornsrev1':
                        # Linear average for 267-273 deg and reshape in WT rows and columns
                        power_matrix = power_models[k][:, 267:274, 0].mean(axis=1).reshape(10, 8)
                        powerGA_matrix = powerGA_models[k][:, 267:274, 0].mean(axis=1).reshape(10, 8)

                        # Sum the innner rows
                        powersum_inner_rows = power_matrix[:, 1:6].sum(axis=1)
                        powerGAsum_inner_rows = powerGA_matrix[:, 1:6].sum(axis=1)

                        l1, = ax.plot(np.linspace(1, 10, 10), powersum_inner_rows /
                                      powersum_inner_rows[0], color=colors[k], linewidth=linewidth)
                        l2, = ax.plot(np.linspace(1, 10, 10), powerGAsum_inner_rows /
                                      powerGAsum_inner_rows[0], color=colors[k], dashes=[5, 2], linewidth=linewidth)
                        ax.set_xlim(0.8, 10.2)
                    else:
                        # Collect data for a row and linear average within wd bin.
                        power_row = np.zeros((len(plot['wts']), 2))
                        for i in range(len(plot['wts'])):
                            if np.isnan(plot['wts'][i]):
                                power_row[i, :] = np.nan
                            else:
                                power_row[i, 0] = power_models[k][plot['wts'][i], int(wd) - 3:int(wd) + 4, 0].mean()
                                power_row[i, 1] = powerGA_models[k][plot['wts'][i], int(wd) - 3:int(wd) + 4, 0].mean()
                        l1, = ax.plot(np.linspace(1, len(plot['wts']), len(
                            plot['wts'])), power_row[:, 0] / power_row[0, 0], color=colors[k], linewidth=linewidth)
                        l2, = ax.plot(np.linspace(1, len(plot['wts']), len(
                            plot['wts'])), power_row[:, 1] / power_row[0, 1], color=colors[k], dashes=[5, 2], linewidth=linewidth)
                        ax.set_xlim(0.8, len(plot['wts']) + 0.2)
                    lines.append(l1)
                    linesGA.append(l2)
                # Data
                ldata = ax.errorbar(data[:, 0], data[:, 1] / data[0, 1], yerr=data[:, 2] / np.sqrt(data[:, 3]),
                                    color='k', elinewidth=1.0, linewidth=0, marker='o', zorder=0, markersize=4)
                lRANS1, = ax.plot(RANS[:, 0], RANS[:, 1], color=cRANS, linewidth=linewidth)
                lRANS2, = ax.plot(RANS[:, 0], RANS[:, 2], color=cRANS, dashes=[5, 2], linewidth=linewidth)
                ax.grid(True)
                ax.set_ylabel('$P_i/P_1$', rotation=0)
                ax.yaxis.labelpad = 20
                ax.set_xlabel('WT nr.')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                fig.tight_layout(rect=[0, 0, 0.75, 0.8])
                # Plot the layout in a small plot
                xWT = (np.array(case['wt_x']) - case['wt_x'][0]) / case['wt'].diameter()
                yWT = (np.array(case['wt_y']) - case['wt_y'][0]) / case['wt'].diameter()
                ax_layout.scatter(xWT, yWT, s=2, color='k')
                for i in range(len(plot['wts'])):
                    if not np.isnan(plot['wts'][i]):
                        ax_layout.scatter(xWT[int(plot['wts'][i])], yWT[int(plot['wts'][i])], s=2, color='r')
                ax_layout.set_xticks([])
                ax_layout.set_yticks([])
                ax_layout.set_aspect(1)
                ax_layout.set_xlim(xWT.min() - 20, xWT.max() + 10)
                ax_layout.set_ylim(yWT.min() - 10, yWT.max() + 20)
                ax_layout.arrow(xWT.min() - 10, yWT.max() + 10, 6 * np.cos((270.0 - wd) / 180.0 * np.pi),
                                6 * np.sin((270.0 - wd) / 180.0 * np.pi), head_width=2, head_length=1, color='r')
                circle1 = plt.Circle((xWT.min() - 10, yWT.max() + 10), 8, color='k', fill=False)
                ax_layout.add_artist(circle1)
                # Move layout plot to the top of the main plot
                pos1 = ax.get_position()
                pos2 = ax_layout.get_position()
                pos3 = [pos2.x0, pos2.y0 + pos1.y0 + pos1.height - (pos2.y0 + pos2.height), pos2.width, pos2.height]
                ax_layout.set_position(pos3)
                filename = case['name'] + '_wd' + str(int(wd)) + '_' + plot['name'] + '.pdf'
            ldummy = copy(lines[0])
            ldummy.set_color('w')
            NOJlabel = 'NOJ ($k=%g$)' % (kNOJ)
            GAUlabel = 'GAU ($k=%g$)' % (kGAU)
            fig.legend((ldata, ldummy, lRANS1, lRANS2, lines[0], linesGA[0], lines[1], linesGA[1]),
                       ('Data', '', 'RANS', 'RANS GA', NOJlabel, NOJlabel + ' GA', GAUlabel, GAUlabel + ' GA'), ncol=4, loc='upper center', bbox_to_anchor=(0, 0, 1, 1), numpoints=1, scatterpoints=1)
            fig.savefig(filename)
            os.system('mv ' + filename + ' report/figures/')


def main():
    if __name__ == '__main__':

        site = UniformSite(p_wd=[1], ti=.1)  # Dummy site (flat and uniform)

        #  Validation cases:
        SingleWakecases = [
            {'name': 'Wieringermeer-West', 'U0': 10.7, 'CT': 0.63, 'TItot': 0.08,  'D': 80.0,
                'zH': 80.0, 'xDown': np.array([2.5, 3.5, 7.5]), 'location': 'onshore'},
            {'name': 'Wieringermeer-East', 'U0': 10.9, 'CT': 0.63, 'TItot': 0.06,  'D': 80.0,
                'zH': 80.0, 'xDown': np.array([2.5, 3.5, 7.5]), 'location': 'onshore'},
            {'name': 'Nibe',               'U0': 8.5,  'CT': 0.89, 'TItot': 0.08,  'D': 40.0,
                'zH': 45.0, 'xDown': np.array([2.5, 4, 7.5]),   'location': 'onshore'},
            {'name': 'Nordtank-500',       'U0': 7.45, 'CT': 0.70, 'TItot': 0.112, 'D': 41.0,
                'zH': 36.0, 'xDown': np.array([2, 5, 7.5]),     'location': 'onshore'},
            {'name': 'NREL-5MW_TIlow',     'U0': 8.0,  'CT': 0.79, 'TItot': 0.04,  'D': 126.0,
                'zH': 90.0, 'xDown': np.array([2.5, 5, 7.5]),   'location': 'offshore'},
            {'name': 'NREL-5MW_TIhigh',    'U0': 8.0,  'CT': 0.79, 'TItot': 0.128, 'D': 126.0,
                'zH': 90.0, 'xDown': np.array([2.5, 5, 7.5]),   'location': 'onshore'}
        ]

        #  If missing wind turbines need to be included in the plot, one should write np.nan in the wts list.
        hr_inner_rows = np.linspace(0, 79, 80).reshape(10, 8)[:, 1:7].flatten(
        ).tolist()  # WTs representing the inner rows of Horns Rev 1
        WFcases = [{'name': 'Wieringermeer', 'U0': 8.35, 'TItot': 0.096, 'wt': N80(),          'wt_x': wt_x_w, 'wt_y': wt_y_w,  'site': site,             'location': 'onshore',
                            'plots': [{'name': 'Row', 'wd': 275.0, 'wts': [0, 1, 2, 3, 4]}]},
                   {'name': 'Lillgrund',     'U0': 9.0,  'TItot': 0.048, 'wt': SWT2p3_93_65(), 'wt_x': wt_x_l, 'wt_y': wt_y_l,  'site': LillgrundSite(),  'location': 'offshore',
                            'plots': [{'name': 'RowB', 'wd': 222.0, 'wts': [14, 13, 12, 11, 10, 9, 8, 7]},
                                      {'name': 'RowD', 'wd': 222.0, 'wts': [29, 28, 27, np.nan, 26, 25, 24, 23]},
                                      {'name': 'RowB', 'wd': 207.0, 'wts': [14, 13, 12, 11, 10, 9, 8, 7]},
                                      {'name': 'RowD', 'wd': 207.0, 'wts': [29, 28, 27, np.nan, 26, 25, 24, 23]},
                                      {'name': 'Row6', 'wd': 120.0, 'wts': [2, 9, 17, 25, 32, 37, 42, 46]},
                                      {'name': 'Row4', 'wd': 120.0, 'wts': [4, 11, 19, np.nan, np.nan, 39, 44]},
                                      {'name': 'Row6', 'wd': 105.0, 'wts': [2, 9, 17, 25, 32, 37, 42, 46]},
                                      {'name': 'Row4', 'wd': 105.0, 'wts': [4, 11, 19, np.nan, np.nan, 39, 44]},
                                      {'name': 'WFeff'}]},
                   {'name': 'Hornsrev1',     'U0': 8.0,  'TItot': 0.056, 'wt': HornsrevV80(),  'wt_x': wt_x_hr, 'wt_y': wt_y_hr, 'site': Hornsrev1Site(), 'location': 'offshore',
                            'plots': [{'name': 'InnerRowMean', 'wd': 270.0, 'wts': hr_inner_rows}]}
                   ]

        # Revert back to old default style:
        mpl.style.use('classic')
        # Latex font
        plt.rcParams['font.family'] = 'STIXGeneral'

        linewidth = 1.5
        cLES = 'c'
        cRANS = 'g'
        colors = ['r', 'b']

        # Plot velocity deficit of single wake cases and calculate integrated velocity deficit
        UdefCases = deficitPlotSingleWakeCases(SingleWakecases, site, linewidth, cLES, cRANS, colors)

        # Plot bar plot of integrated velocity deficit of single wake cases
        barPlotSingleWakeCases(SingleWakecases, UdefCases, cLES, cRANS, colors)

        # Plot power deficit in a row of wind turbine or plot wind farm efficiency
        deficitPlotWFCases(WFcases, linewidth, cLES, cRANS, colors)


main()
