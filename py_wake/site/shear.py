from py_wake import np
from numpy import newaxis as na
from abc import abstractmethod, ABC


class Shear(ABC):
    @abstractmethod
    def __call__(self, localWind, WS_ilk, h):
        """Get wind speed at height

        Parameters
        ----------
        localWind : LocalWind
            Local wind and coordinates
        WS_ilk : array_like
            wind speed
        h : array_like
            height
        Returns
        -------
        WS_ilk : array_like
            Wind speed at height h
        """


class PowerShear(Shear):
    def __init__(self, h_ref=100, alpha=.1, interp_method='nearest'):
        self.h_ref = h_ref
        from py_wake.site._site import get_sector_xr
        self.alpha = get_sector_xr(alpha, "Power shear coefficient")
        self.interp_method = interp_method

    def __call__(self, localWind, WS_ilk, h):
        alpha = self.alpha.interp_ilk(localWind.coords, interp_method=self.interp_method)
        return (h / self.h_ref)[:, na, na] ** alpha * WS_ilk


class LogShear(Shear):
    def __init__(self, h_ref=100, z0=.03, interp_method='nearest'):
        self.h_ref = h_ref
        from py_wake.site._site import get_sector_xr
        self.z0 = get_sector_xr(z0, "Roughness length")
        self.interp_method = interp_method

    def __call__(self, localWind, WS_ilk, h):
        z0 = self.z0.interp_ilk(localWind.coords, interp_method=self.interp_method)
        return np.log(h[:, na, na] / z0) / np.log(self.h_ref / z0) * WS_ilk


# ======================================================================================================================
# Potentially the code below can be used to implement power/log shear interpolation between grid layers
# ======================================================================================================================
# class InterpolationShear(ABC):
#     @abstractmethod
#     def setup(self, ds):
#         """"""
#
#     @abstractmethod
#     def __call__(self):
#         """"""
#
#
# class LinearInterpolationShear():
#     def setup(self, ds):
#         pass
#
#     def __call__(self, WS_ilk, WD_ilk, h_i):
#         return WS_ilk
#
#
# class PowerInterpolationShear():
#     """Apply wind shear coefficient based on speed-up factor at different
#     # height and a reference far field wind shear coefficient (alpha_far)"""
#
#     def __init__(self, alpha_far=.143):
#         self.alpha_far = alpha_far
#
#     def setup(self, ds):
#         ds['wind_shear'] = copy.deepcopy(ds['spd'])
#
#         heights = ds['wind_shear'].coords['z'].data
#
#         # if there is only one layer, assign default value
#         if len(heights) == 1:
#
#             ds['wind_shear'].data = (np.zeros_like(ds['wind_shear'].data) + self.alpha_far)
#
#             print('Note there is only one layer of wind resource data, ' +
#                   'wind shear are assumed as uniform, i.e., {0}'.format(self.alpha_far))
#         else:
#             ds['wind_shear'].data[:, :, 0, :] = (self.alpha_far +
#                                                  np.log(ds['spd'].data[:, :, 0, :] / ds['spd'].data[:, :, 1, :]) /
#                                                  np.log(heights[0] / heights[1]))
#
#             for h in range(1, len(heights)):
#                 ds['wind_shear'].data[:, :, h, :] = (
#                     self.alpha_far +
#                     np.log(ds['spd'].data[:, :, h, :] / ds['spd'].data[:, :, h - 1, :]) /
#                     np.log(heights[h] / heights[h - 1]))
#
#     def __call__(self, WS_ilk, WD_ilk, h_i):
#         ????? WS_ilk = (WS_ilk * (H_hub / self.height_ref) ** wind_shear_il[i_wt, l_wd])
