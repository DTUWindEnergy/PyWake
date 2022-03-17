import numpy as np
from numpy import newaxis as na
from scipy.special import ellipk
from py_wake.ground_models.ground_models import NoGround
from py_wake.utils.elliptic import ellipticPiCarlson
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models import BlockageDeficitModel
from py_wake.utils.gradients import hypot, cabs
from py_wake.deficit_models.utils import a0


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

    def __init__(self, limiter=1e-3, exclude_wake=True, superpositionModel=None, groundModel=NoGround(),
                 upstream_only=False):
        DeficitModel.__init__(self, groundModel=groundModel)
        BlockageDeficitModel.__init__(self, upstream_only=upstream_only, superpositionModel=superpositionModel)
        # limiter to avoid singularities
        self.limiter = limiter
        # if used in a wind farm simulation, set deficit in wake region to
        # zero, as here the wake model is active
        self.exclude_wake = exclude_wake

    def _k2(self, xi, rho, eps=1e-1):
        """
        [k(x,r)]^2 function with regularization parameter epsilon to avoid singularites
        """
        return 4. * rho / ((1. + rho)**2 + xi**2 + eps**2)

    def _calc_layout_terms(self, D_src_il, dw_ijlk, cw_ijlk, **_):

        R_ijlk = (D_src_il / 2)[:, na, :, na]
        # determine dimensionless radial and streamwise coordinates
        rho_ijlk = cw_ijlk / R_ijlk
        # formulation is invalid for r==R therefore avoid this condition
        rho_ijlk[abs(rho_ijlk - 1.) < self.limiter] = 1. + self.limiter
        xi_ijlk = dw_ijlk / R_ijlk

        # term 1
        # non-zero for rho < R
        term1_ijlk = np.zeros_like(rho_ijlk)
        term1_ijlk[rho_ijlk < 1.] = 1.
        # term 2
        # zero for xi==0, thus avoid computation
        ic = (abs(xi_ijlk) > self.limiter)
        # compute k(xi, rho)**2 and k(0, rho)**2
        keps2_ijlk = self._k2(xi_ijlk, rho_ijlk, eps=0.0)
        keps20_ijlk = self._k2(0.0, rho_ijlk, eps=0.0)
        # elliptical integrals
        PI = ellipticPiCarlson(keps20_ijlk, keps2_ijlk)
        KK = ellipk(keps2_ijlk)
        # zero everywhere else
        term2_ijlk = np.zeros_like(rho_ijlk)
        # simplified terms by inserting k
        term2_ijlk[ic] = xi_ijlk[ic] / np.pi * np.sqrt(1. / ((1. + rho_ijlk[ic])**2 + xi_ijlk[ic]**2)) * \
            (KK[ic] + (1. - rho_ijlk[ic]) / (1. + rho_ijlk[ic]) * PI[ic])

        # deficit shape function
        self.dmu_ijlk = term1_ijlk + term2_ijlk

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        """
        The analytical relationships can be found in [1,2], in particular equations (7-8) from [1].
        """
        if not self.deficit_initalized:
            # calculate layout term, self.dmu_G_ijlk
            self._calc_layout_terms(D_src_il, dw_ijlk, cw_ijlk)

        # circulation/strength of vortex cylinder
        gammat_ilk = WS_ilk * 2. * a0(ct_ilk)

        deficit_ijlk = gammat_ilk[:, na] / 2. * self.dmu_ijlk

        if self.exclude_wake:
            # indices on rotor plane and in wake region
            R_il = D_src_il / 2
            iw = ((dw_ijlk / R_il[:, na, :, na] >= -self.limiter) &
                  (cabs(cw_ijlk) <= R_il[:, na, :, na])) * np.full(deficit_ijlk.shape, True)
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
