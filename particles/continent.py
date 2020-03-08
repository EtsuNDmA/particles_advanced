import numpy as np
import pandas as pd
from shapely import ops
from shapely.geometry import Point, Polygon
from tqdm import tqdm


class Continent:
    def __init__(self, df, scale, buffer):
        polygons = []
        df = df[(df['time'] == 1) & (df['depth'] == 1)]
        for row in tqdm(df.iterrows(), total=df.shape[0]):
            if any([pd.isna(row[1]['uo']), pd.isna(row[1]['vo'])]):
                lon = row[1]['lon']
                lat = row[1]['lat']

                exterior = np.array([[lat - 1, lon - 1],
                                     [lat - 1, lon + 1],
                                     [lat + 1, lon + 1],
                                     [lat + 1, lon - 1]])
                exterior *= np.array(scale)
                poly = Polygon(exterior)
                polygons.append(poly)
        self.buffer = buffer
        self.continent = ops.unary_union(polygons)
        self.continent_buffer = self.continent.buffer(self.buffer)

    def buffer_contains(self, xy):
        """Point is near the coastline"""
        return self.continent_buffer.contains(Point(*xy))

    def contains(self, xy):
        return self.continent.contains(Point(*xy))
