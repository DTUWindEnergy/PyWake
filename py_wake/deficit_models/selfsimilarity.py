import numpy as np
from numpy import newaxis as na
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deficit_models import BlockageDeficitModel


class SelfSimilarityDeficit(BlockageDeficitModel):
    """References:
        [1] N. Troldborg, A.R. Meyer Forsting, Wind Energy, 2016
    """
    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, ss_gamma=1.1, ss_lambda=0.587, ss_eta=1.32,
                 ss_alpha=8. / 9., ss_beta=np.sqrt(2), limiter=1e-10, superpositionModel=None):
        super().__init__(superpositionModel=superpositionModel)
        # function constants defined in [1]
        self.ss_gamma = ss_gamma
        self.ss_lambda = ss_lambda
        self.ss_eta = ss_eta
        self.ss_alpha = ss_alpha
        self.ss_beta = ss_beta
        # coefficients for BEM approximation by Madsen (1997)
        self.a0p = np.array([0.2460, 0.0586, 0.0883])
        # limiter for singularities
        self.limiter = limiter

    def r12(self, x_ijlk):
        """
        Compute half radius of self-similar profile as function of streamwise
        location (x<0 upstream)
        Eq. (13) from [1]
        """
        r12_ijlk = np.sqrt(self.ss_lambda * (self.ss_eta + x_ijlk ** 2))

        return r12_ijlk

    def gamma(self, x_ijlk, ct_ilk):
        """
        Compute thrust coefficient scaling factor
        Refer to Eq. (8) from [1]
        """
        return self.ss_gamma * np.ones_like(x_ijlk)

    def f_eps(self, x_ijlk, cw_ijlk, R_ijl):
        """
        Radial induction shape function
        Eq. (6) from [1]
        """
        r12_ijlk = self.r12(x_ijlk)
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'overflow encountered in cosh')
            feps_ijlk = (1 / np.cosh(self.ss_beta * cw_ijlk / (R_ijl[..., na] * r12_ijlk))) ** self.ss_alpha

        return feps_ijlk

    def a0f(self, x_ijlk):
        """
        Axial induction shape function along centreline , derived from a
        vortex cylinder. Eq. (7) from [1]
        """
        a0f_ijlk = (1. + x_ijlk / np.sqrt(1. + x_ijlk**2))
        return a0f_ijlk

    def a0(self, x_ijlk, ct_ilk):
        """
        BEM axial induction approximation by Madsen (1997). Here the effective
        CT is used instead, which is gamma*CT as shown in Eq. (8) in [1].
        """
        gamma_ct_ijlk = self.gamma(x_ijlk, ct_ilk) * ct_ilk[:, na]
        a0_ijlk = self.a0p[2] * gamma_ct_ijlk**3 + self.a0p[1] * gamma_ct_ijlk**2 + self.a0p[0] * gamma_ct_ijlk
        return a0_ijlk

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        """
        Deficit as function of axial and radial coordinates.
        Eq. (5) in [1].
        """
        R_ijl = (D_src_il / 2)[:, na]
        x_ijlk = dw_ijlk / R_ijl[..., na]
        # radial shape function
        feps_ijlk = self.f_eps(x_ijlk, cw_ijlk, R_ijl)
        a0x_ijlk = self.a0(x_ijlk, ct_ilk) * self.a0f(x_ijlk)
        # only activate the model upstream of the rotor
        return WS_ilk[:, na] * (x_ijlk < -self.limiter) * a0x_ijlk * feps_ijlk


class SelfSimilarityDeficit2020(SelfSimilarityDeficit):
    """
    This is an updated version of [1]. The new features are found in the radial
    and axial functions:
        1. Radially Eq. (13) is replaced by a linear fit, which ensures the
           induction half width, r12, to continue to diminish approaching the
           rotor. This avoids unphysically large lateral induction tails,
           which could negatively influence wind farm simulations.
        2. The value of gamma in Eq. (8) is revisited. Now gamma is a function
           of CT and axial coordinate to force the axial induction to match
           the simulated results more closely. The fit is valid over a larger
           range of thrust coefficients and the results of the constantly
           loaded rotor are excluded in the fit.
    References:
        [1] N. Troldborg, A.R. Meyer Fortsing, Wind Energy, 2016
    """

    def __init__(self, ss_alpha=8. / 9., ss_beta=np.sqrt(2),
                 r12p=np.array([-0.672, 0.4897]),
                 ngp=np.array([-1.381, 2.627, -1.524, 1.336]),
                 fgp=np.array([-0.06489, 0.4911, 1.116, -0.1577]),
                 limiter=1e-10, superpositionModel=None):
        BlockageDeficitModel.__init__(self, superpositionModel=superpositionModel)
        # original constants from [1]
        self.ss_alpha = ss_alpha
        self.ss_beta = ss_beta
        # coefficients for the half width approximation
        self.r12p = r12p
        # cofficients for the near- and farfield approximations of gamma
        self.ngp = ngp
        self.fgp = fgp
        # coefficients for BEM approximation by Madsen (1997)
        self.a0p = np.array([0.2460, 0.0586, 0.0883])
        # limiter for singularities
        self.limiter = limiter

    def r12(self, x_ijlk):
        """
        Compute half radius of self-similar profile as function of streamwise
        location (x<0 upstream)
        Linear replacement of Eq. (13) [1]
        """
        r12_ijlk = self.r12p[0] * x_ijlk + self.r12p[1]
        return r12_ijlk

    def far_gamma(self, ct_ilk):
        """
        gamma(CT) @ x/R = -6
        """
        fg_ilk = self.fgp[0] * np.sin((ct_ilk - self.fgp[1]) / self.fgp[3]) + self.fgp[2]
        return fg_ilk

    def near_gamma(self, ct_ilk):
        """
        gamma(CT) @ x/R = -1
        """
        fn_ilk = self.ngp[0] * ct_ilk**3 + self.ngp[1] * ct_ilk**2 + self.ngp[2] * ct_ilk + self.ngp[3]
        return fn_ilk

    def inter_gamma_fac(self, x_ijlk):
        """
        Interpolation coefficient between near- and far-field gamma(CT)
        """
        finter_ijlk = np.abs(self.a0f(x_ijlk) - self.a0f(-1.)) / np.ptp(self.a0f(np.array([-6, -1])))
        finter_ijlk[x_ijlk < -6] = 1.
        finter_ijlk[x_ijlk > -1] = 0.
        return finter_ijlk

    def gamma(self, x_ijlk, ct_ilk):
        """
        Two-dimensional scaling function gamma(x,CT)
        """
        ng_ilk = self.near_gamma(ct_ilk)
        fg_ilk = self.far_gamma(ct_ilk)
        finter_ijlk = self.inter_gamma_fac(x_ijlk)
        gamma_ijlk = finter_ijlk * fg_ilk[:, na] + (1. - finter_ijlk) * ng_ilk[:, na]
        return gamma_ijlk


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.examples.data.hornsrev1 import Hornsrev1Site
        from py_wake.examples.data import hornsrev1
        from py_wake.superposition_models import LinearSum
        from py_wake.wind_farm_models import All2AllIterative

        site = Hornsrev1Site()
        windTurbines = hornsrev1.HornsrevV80()
        ws = 10
        D = 80
        R = D / 2
        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[.8]]])
        ss = SelfSimilarityDeficit()
        ss20 = SelfSimilarityDeficit2020()

        x, y = -np.arange(200), np.array([0])
        # original model
        deficit = ss.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                  dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                  cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        # updated method
        deficit20 = ss20.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                      dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                      cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        plt.figure()
        plt.title('Fig 11 from [1]')
        plt.xlabel('x/R')
        plt.ylabel('a')
        plt.plot(x / R, deficit[0, :, 0, 0] / ws, label='original')
        plt.plot(x / R, deficit20[0, :, 0, 0] / ws, '--', label='updated')
        plt.legend()

        plt.figure()
        x, y = np.array([-2 * R]), np.arange(200)
        deficit = ss.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                  dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                  cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        deficit20 = ss20.calc_deficit(WS_ilk=WS_ilk, D_src_il=D_src_il,
                                      dw_ijlk=x.reshape((1, len(x), 1, 1)),
                                      cw_ijlk=y.reshape((1, len(y), 1, 1)), ct_ilk=ct_ilk)
        plt.title('Fig 10 from [1]')
        r12 = ss.r12(x / R)
        r12_20 = ss20.r12(x / R)
        plt.xlabel('y/R12 (epsilon)')
        plt.ylabel('f')
        plt.plot((y / R) / r12, deficit[0, :, 0, 0] / deficit[0, 0, 0, 0], label='original')
        plt.plot((y / R) / r12_20, deficit20[0, :, 0, 0] / deficit20[0, 0, 0, 0], '--', label='updated')
        plt.legend()

        plt.figure()
        noj_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=ss)
        noj_ss20 = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                    superpositionModel=LinearSum(), blockage_deficitModel=ss20)
        flow_map = noj_ss(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        flow_map20 = noj_ss20(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        clevels = [.9, .95, .98, .99, .995, .998, .999, 1., 1.01, 1.02, 1.03]
        flow_map.plot_wake_map()
        plt.contour(flow_map.x, flow_map.y, flow_map.WS_eff[:, :,
                                                            0, -1, 0] / 10, levels=clevels, colors='k', linewidths=0.5)
        plt.contour(flow_map.x, flow_map.y, flow_map20.WS_eff[:, :,
                                                              0, -1, 0] / 10, levels=clevels, colors='r', linewidths=0.5)
        plt.title('Original (black) vs updated (red)')
        plt.show()

        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        noj_ss = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=ss)
        noj_ss20 = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                    superpositionModel=LinearSum(), blockage_deficitModel=ss20)
        # run wind farm simulation
        sim_res = noj_ss(x, y, wd=[0, 30, 45, 60, 90], ws=[5, 10, 15])
        sim_res20 = noj_ss20(x, y, wd=[0, 30, 45, 60, 90], ws=[5, 10, 15])

        # calculate AEP
        aep = sim_res.aep().sum()
        aep20 = sim_res20.aep().sum()

        # plot wake map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        levels = np.array([.9, .95, .98, .99, .995, .998, .999, 1., 1.01, 1.02, 1.03]) * 10.
        print(noj_ss)
        flow_map = sim_res.flow_map(wd=30, ws=10.)
        flow_map.plot_wake_map(levels=levels, ax=ax1, plot_colorbar=False)
        ax1.set_title('Original Self-Similar, AEP: %.3f GWh' % aep)

        # plot wake map
        print(noj_ss20)
        flow_map = sim_res20.flow_map(wd=30, ws=10.)
        flow_map.plot_wake_map(levels=levels, ax=ax2, plot_colorbar=False)
        ax2.set_title('Self-Similar 2020, AEP: %.3f GWh' % aep20)
        plt.show()


main()
