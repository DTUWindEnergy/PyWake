import numpy as np
from numpy import newaxis as na
from py_wake.deficit_models import BlockageDeficitModel


class Rathmann(BlockageDeficitModel):
    """
    Ole Sten Rathmann (DTU) developed in 2020 an approximation to the vortex
    cylinder solution (E. Branlard and M. Gaunaa, 2014). In speed it is
    comparable to the vortex dipole method, whilst giving a flow-field
    nearly identical to the vortex cylinder model for x/R < -1. Its centreline
    deficit is identical to the vortex cylinder model, whilst using a radial shape
    function that depends on the opening of the vortex cylinder seen from
    a point upstream. To simulate the speed-up downstream the deficit is mirrored in the
    rotor plane with a sign change.

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


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.superposition_models import LinearSum
        from py_wake.wind_farm_models import All2AllIterative
        from py_wake.deficit_models.no_wake import NoWakeDeficit
        from py_wake.deficit_models.vortexcylinder import VortexCylinder
        from py_wake.deficit_models.vortexdipole import VortexDipole
        import matplotlib.pyplot as plt
        from py_wake import HorizontalGrid
        from timeit import default_timer as timer

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()
        d = windTurbines.diameter()
        ra = Rathmann()
        grid = HorizontalGrid(x=np.linspace(-6, 6, 100) * d, y=np.linspace(0, 4, 100) * d)

        noj_ra = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=ra)
        noj_vc = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=VortexCylinder())
        noj_vd = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=VortexDipole())
        t1 = timer()
        flow_map = noj_ra(x=[0], y=[0], wd=[270], ws=[10]).flow_map(grid=grid)
        t2 = timer()
        flow_map_vc = noj_vc(x=[0], y=[0], wd=[270], ws=[10]).flow_map(grid=grid)
        t3 = timer()
        flow_map_vd = noj_vd(x=[0], y=[0], wd=[270], ws=[10]).flow_map(grid=grid)
        t4 = timer()
        print(t2 - t1, t3 - t2, t4 - t3)

        plt.figure()
        clevels = np.array([.6, .7, .8, .9, .95, .98, .99, .995, .998, .999, 1., 1.005, 1.01, 1.02, 1.05]) * 10.
        flow_map.plot_wake_map(levels=clevels)
        plt.contour(flow_map.x, flow_map.y, flow_map.WS_eff[:, :, 0, -1, 0], levels=clevels, colors='k', linewidths=1)
        plt.contour(flow_map.x, flow_map.y, flow_map_vc.WS_eff[:, :, 0, -1, 0], levels=clevels, colors='r', linewidths=1, linestyles='dashed')
        plt.contour(flow_map.x, flow_map.y, flow_map_vd.WS_eff[:, :, 0, -1, 0], levels=clevels, colors='b', linewidths=1, linestyles='dotted')
        plt.title('Rathmann')
        plt.ylabel("Crosswind distance [y/R]")
        plt.xlabel("Downwind distance [x/R]")
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


main()
