'''
Utils for working with ``shapely.geometry`` and ``geopandas.GeoDataFrame`` like objects.
'''

import geopandas as gpd
import warnings

from pathlib import Path
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from typing import Union
from typing import List
from typing import Optional

def read_geometries(
        in_dataset: Union[Path, gpd.GeoDataFrame]
        ) -> gpd.GeoDataFrame:
    """
    Returns a geodataframe containing vector features

    :param in_dataset:
        path-like object or ``GeoDataFrame``
    """

    if isinstance(in_dataset, gpd.GeoDataFrame):
        return in_dataset
    elif isinstance(in_dataset, Path):
        try:
            return gpd.read_file(in_dataset)
        except Exception as e:
            raise Exception from e
    else:
        raise NotImplementedError(
            f'Could not read geometries of input type {type(in_dataset)}'
        )

def check_geometry_types(
        in_dataset: Union[Path, gpd.GeoDataFrame],
        allowed_geometry_types: List[str],
        remove_empty_geoms: Optional[bool] = True
    ) -> gpd.GeoDataFrame:
    """
    Checks if a ``GeoDataFrame`` contains allowed ``shapely.geometry``
    types, only. Raises an error if geometry types other than those allowed are
    found.

    :param allowed_geometry_types:
        list of allowed geometry types
    :param in_dataset:
        file with vector geometries (e.g., ESRI shapefile or GEOJSON) or geodataframe
        to check
    :param remove_empty_geoms:
        when True (default) removes features with empty (None-type) geometry from
        ``in_dataset`` before carrying out the type checking.
    :return:
        ``GeoDataFrame`` with checked (and optionally cleaned) vector features.
    """

    # read dataset
    gdf = read_geometries(in_dataset)

    # check for None geometries (might happen when buffering polygons)
    if remove_empty_geoms:
        num_none_type_geoms = gdf[gdf.geometry == None].shape[0]
        if num_none_type_geoms > 0:
            warnings.warn(
                f'Ignoring {num_none_type_geoms} records where ' \
                f'geometries are of type None'
            )
            gdf = gdf.drop(gdf[gdf.geometry == None].index)

    # check for allowed geometry types
    gdf_aoi_geoms_types = list(gdf.geom_type.unique())
    not_allowed_types = [x for x in gdf_aoi_geoms_types if x not in allowed_geometry_types]

    if len(not_allowed_types) > 0:
        raise ValueError(
            f'Encounter geometry types not allowed for reading band data: ({not_allowed_types})'
        )
    return gdf

def convert_3D_2D(geometry: gpd.GeoSeries) -> gpd.GeoSeries:
    '''
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons.
    Snippet taken from https://gist.github.com/rmania/8c88377a5c902dfbc134795a7af538d8
    (accessed latest Jan 18th 2021)

    :param geometry:
        ``GeoSeries`` from ``GeoDataFrame``
    :returns:
        updated ``GeoSeries`` without third dimension (z)
    '''

    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
        else:
            new_geo = geometry
            break
    return new_geo
