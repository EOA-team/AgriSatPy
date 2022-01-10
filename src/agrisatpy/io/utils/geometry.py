'''
Utils for working with ``shapely.geometry`` and ``geopandas.GeoDataFrame`` like objects.
'''

import geopandas as gpd
import warnings

from pathlib import Path
from shapely import geometry
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
