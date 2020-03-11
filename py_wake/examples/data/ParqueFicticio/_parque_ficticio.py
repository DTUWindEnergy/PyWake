from py_wake.site.wasp_grid_site import WaspGridSite, load_wasp_grd
from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path
import numpy as np
from py_wake.site.distance import TerrainFollowingDistance

"""
min x: 262878
min y: 6504214
max x: 265078
max y: 6507414.0
Resolution: 100
columns: 23
rows: 33

30 and 200 m

"""


class ParqueFicticioSite(WaspGridSite):
    def __init__(self, distance=TerrainFollowingDistance(distance_resolution=2000), mode='valid'):
        ds = load_wasp_grd(ParqueFicticio_path, speedup_using_pickle=True)
        WaspGridSite.__init__(self, ds, distance, mode)
        self.initial_position = np.array([
            [263655.0, 6506601.0],
            [263891.1, 6506394.0],
            [264022.2, 6506124.0],
            [264058.9, 6505891.0],
            [264095.6, 6505585.0],
            [264022.2, 6505365.0],
            [264022.2, 6505145.0],
            [263936.5, 6504802.0],
        ])
