from py_wake.site.wasp_grid_site import WaspGridSite, WaspGridSiteBase
from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path
import numpy as np
from py_wake.site.distance import TerrainFollowingDistance


def ParqueFicticioSite(distance=TerrainFollowingDistance(distance_resolution=2000)):
    site = WaspGridSiteBase.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=True, distance=distance)
    site.distance_type = 'terrain_following'
    site.initial_position = np.array([
        [263655.0, 6506601.0],
        [263891.1, 6506394.0],
        [264022.2, 6506124.0],
        [264058.9, 6505891.0],
        [264095.6, 6505585.0],
        [264022.2, 6505365.0],
        [264022.2, 6505145.0],
        [263936.5, 6504802.0],
    ])
    return site
