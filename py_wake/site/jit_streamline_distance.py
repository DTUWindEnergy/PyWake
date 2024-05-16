from py_wake.site.distance import StraightDistance
from py_wake import np
from numpy import newaxis as na
import matplotlib.pyplot as plt
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.examples.data.hornsrev1 import V80
from py_wake.deficit_models.noj import NOJ
from py_wake.flow_map import XYGrid
from py_wake.utils.streamline import VectorField3D


class JITStreamlineDistance(StraightDistance):
    """Just-In-Time Streamline Distance
    Calculates downwind crosswind and vertical distance along streamlines.
    Streamlines calculated in each call
    """

    def __init__(self, vectorField, step_size=20):
        """Parameters
        ----------
        vectorField : VectorField3d
        step_size : int for float
            Size of linear streamline steps
        """
        StraightDistance.__init__(self, wind_direction='wd')
        self.vectorField = vectorField
        self.step_size = step_size

    def __call__(self, wd_l, WD_ilk, src_idx=slice(None), dst_idx=slice(None)):
        assert self.src_x_ilk.shape[1:] == (1, 1), 'JITStreamlineDistance does not support flowcase dependent positions'
        start_points_m = np.array([self.src_x_ilk[src_idx], self.src_y_ilk[src_idx],
                                   self.src_h_ilk[src_idx]])[:, :, 0, 0].T

        if len(np.shape(dst_idx)) == 2:
            # dst_idx depends on wind direction
            dw_jl, hcw_jl, dh_jl = [v[0, :, :, 0] for v in StraightDistance.__call__(self, wd_l=wd_l,
                                                                                     src_idx=src_idx, dst_idx=dst_idx)]
            dw_mj, hcw_mj, dh_mj = [np.moveaxis(v, 0, 1) for v in [dw_jl, hcw_jl, dh_jl]]

            i_wd_l = np.arange(np.shape(dst_idx)[1])
            dz_jlk = self.dz_iilk[src_idx, dst_idx, np.minimum(i_wd_l, self.dh_iilk.shape[2] - 1).astype(int)]
            dz_mj = np.moveaxis(dz_jlk[:, :, 0], 0, 1)

            wd_m = wd_l
        else:
            # dst_idx independent of wind direction
            dw_ijlk, hcw_ijlk, dh_ijlk = StraightDistance.__call__(self, wd_l=wd_l,
                                                                   src_idx=src_idx, dst_idx=dst_idx)
            # +0 ~ autograd safe copy (broadcast_to returns readonly array)
            dz_ijlk = np.broadcast_to(self.dz_ijlk[src_idx][:, dst_idx], dw_ijlk.shape) + 0
            I, J, L, K = dw_ijlk.shape
            dw_mj, hcw_mj, dh_mj, dz_mj = [np.moveaxis(v, 1, 2).reshape(I * L, J)
                                           for v in [dw_ijlk, hcw_ijlk, dh_ijlk, dz_ijlk]]

            wd_m = np.tile(wd_l, I)
            start_points_m = np.repeat(start_points_m, L, 0)

        stream_lines = self.vectorField.stream_lines(wd_m, start_points=start_points_m, dw_stop=dw_mj.max(1),
                                                     step_size=self.step_size)

        dxyz = np.diff(np.concatenate([stream_lines[:, :1], stream_lines], 1), 1, -2)
        length_is = np.cumsum(np.sqrt(np.sum(dxyz**2, -1)), -1)
        dist_xyz = stream_lines - start_points_m[:, na]
        t = np.deg2rad(270 - wd_m)[:, na]
        dw_is = dist_xyz[:, :, 0] * np.cos(t) + dist_xyz[:, :, 1] * np.sin(t)
        hcw_is = dist_xyz[:, :, 0] * np.sin(t) - dist_xyz[:, :, 1] * np.cos(t)

        for m, (dw_j, dw_s, hcw_s, dh_s, length_s) in enumerate(
                zip(dw_mj, dw_is, hcw_is, dist_xyz[:, :, 2], length_is)):
            dw = dw_j > 0
            hcw_mj[m, dw] += np.interp(dw_j[dw], dw_s, hcw_s)
            dh_mj[m, dw] -= np.interp(dw_j[dw], dw_s, dh_s)
            dw_mj[m, dw] = np.interp(dw_j[dw], dw_s, length_s)

        # streamline dh contains absolute height different, but pywake needs differences relative to ground, so
        # we need to subtract elevation differences, dz
        dh_mj += dz_mj

        if len(np.shape(dst_idx)) == 2:

            return np.array([np.moveaxis(v, 0, 1)[na, :, :, na] for v in [dw_mj, hcw_mj, dh_mj]])
        else:
            return [v.reshape((I, J, L, K)) for v in [dw_mj, hcw_mj, dh_mj]]


def main():
    if __name__ == '__main__':

        wt = V80()
        vf3d = VectorField3D.from_WaspGridSite(ParqueFicticioSite())
        site = ParqueFicticioSite(distance=JITStreamlineDistance(vf3d))

        x, y = site.initial_position[:].T
        wfm = NOJ(site, wt)
        wd = 330
        sim_res = wfm(x, y, wd=[wd], ws=10)
        fm = sim_res.flow_map(XYGrid(x=np.linspace(site.ds.x[0], site.ds.x[-1], 500),
                                     y=np.linspace(site.ds.y[0], site.ds.y[-1], 500)))
        stream_lines = vf3d.stream_lines(wd=np.full(x.shape, wd), start_points=np.array([x, y, np.full(x.shape, 70)]).T,
                                         dw_stop=y - 6504700)
        fm.plot_wake_map()
        for sl in stream_lines:
            plt.plot(sl[:, 0], sl[:, 1])

        plt.show()


main()
