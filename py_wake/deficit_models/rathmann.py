import numpy as np
from numpy import newaxis as na
from py_wake.deficit_models import BlockageDeficitModel


class Rathmann(BlockageDeficitModel):
    """
    Ole Sten Rathmann (DTU) developed in 2020 an approximation [1] to the vortex
    cylinder solution (E. Branlard and M. Gaunaa, 2014). In speed it is
    comparable to the vortex dipole method, whilst giving a flow-field
    nearly identical to the vortex cylinder model for x/R < -1. Its centreline
    deficit is identical to the vortex cylinder model, whilst using a radial shape
    function that depends on the opening of the vortex cylinder seen from
    a point upstream. To simulate the speed-up downstream the deficit is mirrored in the
    rotor plane with a sign change.

    References:
        [1] A Meyer Forsting, OS Rathmann, MP van der Laan, N Troldborg, B Gribben, G Hawkes, E Branlard -
        Verification of induction zone models for wind farm annual energy production estimation -
        Journal of Physics: Conference Series 1934 (2021) 012023
    """

    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, sct=1.0, limiter=1e-10, exclude_wake=True, superpositionModel=None):
        BlockageDeficitModel.__init__(self, superpositionModel=superpositionModel)
        # coefficients for BEM approximation by Madsen (1997)
        self.a0p = np.array([0.2460, 0.0586, 0.0883])
        # limiter to avoid singularities
        self.limiter = limiter
        # coefficient for scaling the effective forcing
        self.sct = sct
        # if used in a wind farm simulation, set deficit in wake region to
        # zero, as here the wake model is active
        self.exclude_wake = exclude_wake

    def a0(self, ct_ilk):
        """
        BEM axial induction approximation by Madsen (1997).
        """
        a0_ilk = self.a0p[2] * ct_ilk**3 + self.a0p[1] * ct_ilk**2 + self.a0p[0] * ct_ilk
        return a0_ilk

    def dmu(self, xi_ijlk):
        """
        Centreline deficit shape function. Same as for the vortex cylinder model.
        """
        dmu_ijlk = 1 + xi_ijlk / np.sqrt(1 + xi_ijlk**2)
        return dmu_ijlk

    def G(self, xi_ijlk, rho_ijlk):
        """
        Radial shape function, that relies on the opening angles of the cylinder seen
        from the point (xi,rho). It is an approximation of the vortex cylinder behaviour.
        """
        # horizontal angle
        sin2a = 2 * xi_ijlk / np.sqrt((xi_ijlk**2 + (rho_ijlk - 1)**2) * (xi_ijlk**2 + (rho_ijlk + 1)**2))
        # get sin(alpha) from sin(2*alpha)
        sina = np.sqrt((1 - np.sqrt(1 - sin2a**2)) / 2)
        # vertical angle
        sinb = 1 / np.sqrt(xi_ijlk**2 + rho_ijlk**2 + 1)
        # normalise by the value along the centreline (rho=0)
        norm = 1 / (1 + xi_ijlk**2)
        G_ijlk = sina * sinb / norm

        return G_ijlk

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        """
        The deficit is determined from a streamwise and radial shape function, whereas
        the strength is given vom vortex and BEM theory.
        """
        # Ensure dw and cw have the correct shape
        if (cw_ijlk.shape[3] != ct_ilk.shape[2]):
            cw_ijlk = np.repeat(cw_ijlk, ct_ilk.shape[2], axis=3)
            dw_ijlk = np.repeat(dw_ijlk, ct_ilk.shape[2], axis=3)
        R_il = (D_src_il / 2)
        # circulation/strength of vortex dipole Eq. (1) in [1]
        gammat_ilk = WS_ilk * 2. * self.a0(ct_ilk * self.sct)
        # determine dimensionless radial and streamwise coordinates
        rho_ijlk = cw_ijlk / R_il[:, na, :, na]
        xi_ijlk = dw_ijlk / R_il[:, na, :, na]
        # mirror the bahaviour in the rotor-plane
        xi_ijlk[xi_ijlk > 0] = -xi_ijlk[xi_ijlk > 0]
        # centerline shape function
        dmu_ijlk = self.dmu(xi_ijlk)
        # radial shape function
        G_ijlk = self.G(xi_ijlk, rho_ijlk)
        # deficit
        deficit_ijlk = gammat_ilk[:, na] / 2. * dmu_ijlk * G_ijlk
        # turn deficiti into speed-up downstream
        deficit_ijlk[dw_ijlk > 0] = -deficit_ijlk[dw_ijlk > 0]

        if self.exclude_wake:
            # indices in wake region
            iw = ((dw_ijlk / R_il[:, na, :, na] >= -self.limiter) &
                  (np.abs(cw_ijlk) <= R_il[:, na, :, na])) * np.full(deficit_ijlk.shape, True)
            deficit_ijlk[iw] = 0.

        return deficit_ijlk


class RathmannScaled(Rathmann):
    """
    Vortex cylinder based models consistently underestimate the induction, due to missing
    wake expansion [1]. A simple fix alleviating this issue is to simply scale the results
    with respect to the thrust coefficient as demonstrated in [1]. There is also a small
    depenancy on the distance fron the rotor.

    References:
        [1] A Meyer Forsting, OS Rathmann, MP van der Laan, N Troldborg, B Gribben, G Hawkes, E Branlard -
        Verification of induction zone models for wind farm annual energy production estimation -
        Journal of Physics: Conference Series 1934 (2021) 012023
    """

    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, sct=1.0, limiter=1e-10, exclude_wake=True):
        BlockageDeficitModel.__init__(self)
        # coefficients for BEM approximation by Madsen (1997)
        self.a0p = np.array([0.2460, 0.0586, 0.0883])
        # limiter to avoid singularities
        self.limiter = limiter
        # coefficient for scaling the effective forcing
        self.sct = sct
        # if used in a wind farm simulation, set deficit in wake region to
        # zero, as here the wake model is active
        self.exclude_wake = exclude_wake
        # scaling coefficients for Eq.11-13 in [1]
        self.sd = np.array([1.02, 0.1554, 0.0005012, 8.45, 0.025])

    def deficit_scaling(self, D_src_il, dw_ijlk, cw_ijlk, ct_ilk):
        """
        Scaling function defined in [1], Eq. 11-13 forcing the output closer to the
        CFD results.
        """
        r = np.hypot(cw_ijlk, dw_ijlk) / D_src_il[:, na, :, na]
        fac = 1. + self.sd[4] * (r - 5.)
        fac[fac > 1.] = 1.
        mval = 1. - 4. * self.sd[4]
        fac[fac < mval] = mval
        boost_ijlk = self.sd[0] * \
            np.exp(self.sd[1] * fac * ct_ilk[:, na]) + \
            self.sd[2] * np.exp(self.sd[3] * fac * ct_ilk[:, na])

        return boost_ijlk

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        boost_ijlk = self.deficit_scaling(D_src_il, dw_ijlk, cw_ijlk, ct_ilk)
        return boost_ijlk * Rathmann.calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.site._site import UniformSite
        from py_wake.wind_turbines import OneTypeWindTurbines
        from py_wake.superposition_models import LinearSum
        from py_wake.wind_farm_models import All2AllIterative
        from py_wake.deficit_models.no_wake import NoWakeDeficit
        from py_wake.rotor_avg_models import RotorCenter
        from py_wake.deficit_models.vortexcylinder import VortexCylinder
        from py_wake.deficit_models.vortexdipole import VortexDipole
        import matplotlib.pyplot as plt
        from py_wake import HorizontalGrid
        from timeit import default_timer as timer
        import time

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()
        d = windTurbines.diameter()
        ra = Rathmann()
        ras = RathmannScaled()
        grid = HorizontalGrid(x=np.linspace(-6, 6, 100) * d, y=np.linspace(0, 4, 100) * d)

        noj_ra = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=ra)
        noj_ras = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                   superpositionModel=LinearSum(), blockage_deficitModel=ras)
        noj_vc = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=VortexCylinder())
        noj_vd = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=VortexDipole())
        t1 = timer()
        flow_map = noj_ra(x=[0], y=[0], wd=[270], ws=[10]).flow_map(grid=grid)
        t2 = timer()
        flow_map_ras = noj_ras(x=[0], y=[0], wd=[270], ws=[10]).flow_map(grid=grid)
        t3 = timer()
        flow_map_vc = noj_vc(x=[0], y=[0], wd=[270], ws=[10]).flow_map(grid=grid)
        t4 = timer()
        flow_map_vd = noj_vd(x=[0], y=[0], wd=[270], ws=[10]).flow_map(grid=grid)
        t5 = timer()
        print(t2 - t1, t3 - t2, t4 - t3, t5 - t4)

        plt.figure()
        clevels = np.array(
            [.6, .7, .8, .9, .95, .98, .99, .995, .998, .999, 1., 1.005, 1.01, 1.02, 1.05]) * 10.
        flow_map.plot_wake_map(levels=clevels)
        plt.contour(flow_map.x, flow_map.y,
                    flow_map.WS_eff[:, :, 0, -1, 0], levels=clevels, colors='k', linewidths=1)
        plt.contour(flow_map.x, flow_map.y, flow_map_ras.WS_eff[:, :, 0, -1, 0],
                    levels=clevels, colors='g', linewidths=1, linestyles='dashed')
        plt.contour(flow_map.x, flow_map.y, flow_map_vc.WS_eff[:, :, 0, -1, 0],
                    levels=clevels, colors='r', linewidths=1, linestyles='dashed')
        plt.contour(flow_map.x, flow_map.y, flow_map_vd.WS_eff[:, :, 0, -1, 0],
                    levels=clevels, colors='b', linewidths=1, linestyles='dotted')
        plt.plot([0, 0], [0, 0], 'k-', label='Rathmann')
        plt.plot([0, 0], [0, 0], 'g--', label='scaled Rathmann')
        plt.plot([0, 0], [0, 0], 'r--', label='vortex cylinder')
        plt.plot([0, 0], [0, 0], 'b:', label='vortex dipole')
        plt.title('Rathmann')
        plt.ylabel("Crosswind distance [y/R]")
        plt.xlabel("Downwind distance [x/R]")
        plt.legend()
        plt.show()

        # run wind farm simulation
        sim_res = noj_ra(x, y, wd=[0, 30, 45, 60, 90], ws=[5, 10, 15])
        # calculate AEP
        aep = sim_res.aep().sum()
        # plot wake map
        plt.figure()
        print(noj_ra)
        flow_map = sim_res.flow_map(wd=0, ws=10)
        flow_map.plot_wake_map(levels=clevels, plot_colorbar=False)
        plt.title('Rathmann model, AEP: %.3f GWh' % aep)
        plt.show()

        # run wind farm simulation
        sim_res = noj_ras(x, y, wd=[0, 30, 45, 60, 90], ws=[5, 10, 15])
        # calculate AEP
        aep = sim_res.aep().sum()
        # plot wake map
        plt.figure()
        print(noj_ras)
        flow_map = sim_res.flow_map(wd=0, ws=10)
        flow_map.plot_wake_map(levels=clevels, plot_colorbar=False)
        plt.title('Rathmann model, AEP: %.3f GWh' % aep)
        plt.show()

        class epfl_model_wt(OneTypeWindTurbines):
            def __init__(self):
                OneTypeWindTurbines.__init__(self, 'NREL 5MW', diameter=2,
                                             hub_height=1, ct_func=self._ct,
                                             power_func=self._power, power_unit='W')

            def _ct(self, u):
                ct = 0.798
                return ct * u / u

            def _power(self, u):
                cp = 0.5
                A = np.pi
                rho = 1.225
                return 0.5 * rho * u**3 * A * cp

        wt = epfl_model_wt()
        d = wt.diameter()
        h = wt.hub_height()
        wt_x = np.array([-6. * d, -3. * d, 0. * d, 3. * d, 6. * d])
        wt_y = np.array([0., 0., 0., 0., 0.])

        class epfl_wt(UniformSite):
            def __init__(self):
                p_wd = [1]
                ws = [1]
                ti = [0.06]
                UniformSite.__init__(self, p_wd=p_wd, ti=ti, ws=ws)
                self.initial_position = np.array([wt_x, wt_y]).T

        site = epfl_wt()
        blockage_models = [
            ('Rathmann', Rathmann()),
            ('RathmannScaled', RathmannScaled())]

        def pywake_run(blockage_models):

            grid = HorizontalGrid(x=np.linspace(-10 * d, 10 * d, 100),
                                  y=np.linspace(-15, 2 * d, 100))

            res = {}
            out = {}
            for nam, blockage_model in blockage_models:
                print(nam, blockage_model)
                # for dum, rotor_avg_model in rotor_avg_models:
                wm = All2AllIterative(site, wt, wake_deficitModel=NoWakeDeficit(),
                                      superpositionModel=LinearSum(), blockage_deficitModel=blockage_model,
                                      rotorAvgModel=RotorCenter())
                res[nam] = wm(wt_x, wt_y, wd=[180., 195., 210., 225.], ws=[1.])
                tic = time.perf_counter()
                flow_map = res[nam].flow_map(grid=grid, ws=[1.], wd=225.)
                toc = time.perf_counter()
                elapsed_time = toc - tic
                out[nam] = flow_map['WS_eff'] / flow_map['WS']
                out[nam]['time'] = elapsed_time
            return out, res

        out, res = pywake_run(blockage_models)
        # CFD data from Meyer Forsting et al. 2017
        p_cfd = np.array([[-1.12511646713429, 0.268977884040651, 0.712062872514373, 1.08033923355738, 1.97378837188847],
                          [-0.610410399845213, 0.355339771667814,
                              0.670255435929930, 0.915154608331424, 1.52808830519513],
                          [-0.0988002822865217, 0.451698279664756,
                              0.636987630794206, 0.751760283763044, 1.03629687168984],
                          [0.477918399858401, 0.628396438496795, 0.663799054750132, 0.628396438496795, 0.479688468006036]])

        plt.figure()
        lines = ['.-', 'o--', '^', '-.', '.-', '.-']
        theta = [0, 15, 30, 45]
        viridis = plt.cm.viridis(np.linspace(0, 0.9, 5))
        jj = 0
        tno = np.arange(1, 6, 1)
        for i in range(4):
            ii = len(theta) - i - 1
            plt.plot(tno, ((p_cfd[ii, :] / 100. + 1.) / (p_cfd[ii, 0] / 100. + 1.) - 1.) * 100, 's:', color=viridis[i], label='CFD: ' + str(theta[i]) + 'deg',
                     lw=2)

        for nam, blockage_model in blockage_models:
            for i in range(4):
                if i == 3:
                    plt.plot(tno, (res[nam].Power[:, i, 0] - res[nam].Power[0, i, 0]) / res[nam].Power[0, i, 0] * 100, lines[jj],
                             label=nam, color=viridis[i],
                             lw=1, alpha=0.8)
                else:
                    plt.plot(tno, (res[nam].Power[:, i, 0] - res[nam].Power[0, i, 0]) / res[nam].Power[0, i, 0] * 100, lines[jj],
                             color=viridis[i],
                             lw=1, alpha=0.8)
            jj += 1

        plt.grid(alpha=0.2)
        plt.xlabel('Turbine no.')
        plt.ylabel('Power change, $(P-P_1)/P_1$ [%]')
        plt.xticks([0, 1, 2, 3, 4, 5])
        plt.xlim([.9, 5.1])
        plt.legend(fontsize=11)
        plt.show()


main()
