import numpy as np
from numpy import newaxis as na
from scipy.special import ellipk
from py_wake.utils.elliptic import ellipticPiCarlson
from py_wake.deficit_models import BlockageDeficitModel


class VortexCylinder(BlockageDeficitModel):
    """
    Induced velocity from a semi infinite cylinder of tangential vorticity,
    extending along the z axis.
    This script is an adapted version of the one published by Emmanuel Branlard:
    https://github.com/ebranlard/wiz/blob/master/wiz/VortexCylinder.py
    References:
        [1] E. Branlard, M. Gaunaa - Cylindrical vortex wake model: right cylinder - Wind Energy, 2014
        [2] E. Branlard - Wind Turbine Aerodynamics and Vorticity Based Method, Springer, 2017
        [3] E. Branlard, A. Meyer Forsting, Using a cylindrical vortex model to assess the induction
            zone in front of aligned and yawed rotors, in Proceedings of EWEA Offshore Conference, 2015
    """

    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, limiter=1e-10, exclude_wake=True, superpositionModel=None):
        BlockageDeficitModel.__init__(self, superpositionModel=superpositionModel)
        # coefficients for BEM approximation by Madsen (1997)
        self.a0p = np.array([0.2460, 0.0586, 0.0883])
        # limiter to avoid singularities
        self.limiter = limiter
        # if used in a wind farm simulation, set deficit in wake region to
        # zero, as here the wake model is active
        self.exclude_wake = exclude_wake

    def a0(self, ct_ilk):
        """
        BEM axial induction approximation by Madsen (1997).
        """
        a0_ilk = self.a0p[2] * ct_ilk**3 + self.a0p[1] * ct_ilk**2 + self.a0p[0] * ct_ilk
        return a0_ilk

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        """
        The analytical relationships can be found in [1,2], in particular equations (7-8) from [1].
        """
        # Ensure dw and cw have the correct shape
        if (cw_ijlk.shape[3] != ct_ilk.shape[2]):
            cw_ijlk = np.repeat(cw_ijlk, ct_ilk.shape[2], axis=3)
            dw_ijlk = np.repeat(dw_ijlk, ct_ilk.shape[2], axis=3)

        R_il = D_src_il / 2
        # radial distance from turbine centre
        r_ijlk = np.hypot(dw_ijlk, cw_ijlk)
        # circulation/strength of vortex cylinder
        gammat_ilk = WS_ilk * 2. * self.a0(ct_ilk)
        # initialize deficit
        deficit_ijlk = np.zeros_like(dw_ijlk)
        # deficit along centreline
        ic = (cw_ijlk / R_il[:, na, :, na] < self.limiter)
        deficit_ijlk = gammat_ilk[:, na] / 2 * (1 + dw_ijlk / np.sqrt(dw_ijlk**2 + R_il[:, na, :, na]**2)) * ic
        # singularity on rotor and close to R
        ir = (np.abs(r_ijlk / R_il[:, na, :, na] - 1.) <
              self.limiter) & (np.abs(dw_ijlk / R_il[:, na, :, na]) < self.limiter)
        deficit_ijlk = deficit_ijlk * (~ir) + gammat_ilk[:, na] / 4. * ir
        # compute deficit everywhere else
        # indices outside any of the previously computed regions
        io = np.logical_not(np.logical_or(ic, ir))
        # elliptic integrals
        k_2 = 4 * cw_ijlk * R_il[:, na, :, na] / ((R_il[:, na, :, na] + cw_ijlk)**2 + dw_ijlk**2)
        k0_2 = 4 * cw_ijlk * R_il[:, na, :, na] / (R_il[:, na, :, na] + cw_ijlk)**2
        k = np.sqrt(k_2)
        KK = ellipk(k_2)
        k_2[k_2 > 1.] = 1.  # Safety purely for numerical precision
        PI = ellipticPiCarlson(k0_2, k_2)
        # --- Special values
        PI[PI == np.inf] = 0
        PI[(cw_ijlk == R_il[:, na, :, na])] = 0  # when r==R, PI=0
        KK[KK == np.inf] = 0  # when r==R, K=0
        # Term 1 has a singularity at r=R, # T1 = (R-r + np.abs(R-r))/(2*np.abs(R-r))
        T1 = np.zeros_like(cw_ijlk)
        T1[cw_ijlk == R_il[:, na, :, na]] = 1 / 2
        T1[cw_ijlk < R_il[:, na, :, na]] = 1
        div = (2 * np.pi * np.sqrt(cw_ijlk * R_il[:, na, :, na]))
        div[div == 0.] = np.inf
        deficit_ijlk[io] = (gammat_ilk[:, na] / 2 * (T1 + dw_ijlk * k / div *
                                                     (KK + (R_il[:, na, :, na] - cw_ijlk) / (R_il[:, na, :, na] + cw_ijlk) * PI)))[io]
        if self.exclude_wake:
            # indices on rotor plane and in wake region
            iw = ((dw_ijlk / R_il[:, na, :, na] >= -self.limiter) &
                  (np.abs(cw_ijlk) <= R_il[:, na, :, na])) * np.full(deficit_ijlk.shape, True)
            deficit_ijlk[iw] = 0.

        return deficit_ijlk


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.superposition_models import LinearSum
        from py_wake.wind_farm_models import All2AllIterative
        from py_wake.deficit_models.no_wake import NoWakeDeficit
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()
        vc = VortexCylinder()

        plt.figure()
        noj_vc = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=vc)
        flow_map = noj_vc(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        clevels = np.array([.6, .7, .8, .9, .95, .98, .99, .995, .998, .999, 1.]) * 10.
        flow_map.plot_wake_map(levels=clevels)
        plt.title('Vortex Cylinder')
        plt.ylabel("Crosswind distance [y/R]")
        plt.xlabel("Downwind distance [x/R]")
        plt.show()

        # run wind farm simulation
        sim_res = noj_vc(x, y, wd=[0, 30, 45, 60, 90], ws=[5, 10, 15])

        # calculate AEP
        aep = sim_res.aep().sum()

        # plot wake map
        plt.figure()
        print(noj_vc)
        flow_map = sim_res.flow_map(wd=0, ws=10)
        flow_map.plot_wake_map(levels=clevels, plot_colorbar=False)
        plt.title('Vortex Cylinder model, AEP: %.3f GWh' % aep)
        plt.show()


main()
