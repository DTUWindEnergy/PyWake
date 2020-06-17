import numpy as np
from numpy import newaxis as na
from abc import abstractmethod, ABC


class Shear(ABC):
    @abstractmethod
    def __call__(self, WS_ilk, WD_ilk, h_i):
        """Get wind speed at height

        Parameters
        ----------
        WS_ilk : array_like
            wind speed
        WD_ilk : array_like
            wind direction
        h_i : array_like
            height
        Returns
        -------
        WS_ilk : array_like
            Wind speed at height h_i
        """


class PowerShear():
    def __init__(self, h_ref, alpha, interp_method='nearest'):
        self.h_ref = h_ref
        from py_wake.site._site import get_sector_xr
        self.alpha = get_sector_xr(alpha, "Power shear coefficient")
        self.interp_method = interp_method

    def __call__(self, WS, WD, h):
        alpha = self.alpha.interp_all(WD, method=self.interp_method)
        if alpha.shape == ():
            alpha = alpha.data
        return (h / self.h_ref) ** alpha * WS


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
