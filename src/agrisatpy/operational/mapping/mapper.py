'''
Generic mapping module
'''

import cv2
from datetime import date
from geopandas import GeoDataFrame
from pandas import DataFrame
from pathlib import Path
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from agrisatpy.core.sat_data_handler import SatDataHandler


class Feature(object):
    """
    Class representing a feature, e.g., an area of interest

    :attrib identifier:
        identifier of the feature
    :attrib geom:
        geometry of the feature
    :attrib epsg:
        epsg code of the feature's geometry
    """

    def __init__(
            self,
            identifier: Any,
            geom: Union[Point, Polygon, MultiPolygon],
            epsg: int
        ):

        object.__setattr__(self, 'identifier', identifier)
        object.__setattr__(self, 'geom', geom)
        object.__setattr__(self, 'epsg', epsg)

    def __setattr__(self, *args):
        raise TypeError('Feature attributes are immutable')

    def __delattr__(self, *args):
        raise TypeError('Feature attributes are immutable')

    def __repr__(self):
        return str(self.__dict__)

    def to_gdf(self) -> GeoDataFrame:
        """Returns the feature as GeoDataFrame"""
        return GeoDataFrame(
            index=[self.identifier],
            crs=f'epsg:{self.epsg}',
            geometry=[self.geom]
        )


class MapperConfigs(object):
    """
    Class defining configurations for the ``Mapper`` class

    :attrib band_names:
        names of raster bands to process from each dataset found during the
        mapping process
    :attrib resampling_method:
        resampling might become necessary when the spatial resolution
        changes. Nearest neighbor by default.
    :attrib spatial_resolution:
        if provided brings all raster bands into the same spatial resolution
    :attrib reducers:
        optional list of spatial reducers (e.g., 'mean') converting all
        raster observations from 2d arrays to scalars.
    """

    def __init__(
            self,
            band_names: Optional[List[str]] = None,
            resampling_method: Optional[int] = cv2.INTER_NEAREST_EXACT,
            spatial_resolution: Optional[Union[int, float]] = None,
            reducers: Optional[List[str]] = None
        ):

        object.__setattr__(self, 'band_names', band_names)
        object.__setattr__(self, 'resampling_method', resampling_method)
        object.__setattr__(self, 'spatial_resolution', spatial_resolution)
        object.__setattr__(self, 'reducers', reducers)

    def __setattr__(self, *args):
        raise TypeError('MapperConfigs attributes are immutable')

    def __delattr__(self, *args):
        raise TypeError('MapperConfigs attributes are immutable')

    def __repr__(self):
        return str(self.__dict__)


class Mapper(object):
    """
    Generic Mapping class to extract raster data for a selection of areas of interest
    (AOIs) and time period.

    :attrib date_start:
        start date of the time period to consider (inclusive)
    :attrib date_end:
        end date of the time period to consider (inclusive)
    :attrib aoi_features:
        ``GeoDataFrame`` or any vector file understood by ``fiona`` with
        geometries of type ``Point``, ``Polygon`` or ``MultiPolygon``
        defining the Areas Of Interest (AOIs) to extract (e.g., agricultural
        field parcels). Each feature will be returned separately
    :attrib unique_id_attribute:
        attribute in the `polygon_features`'s attribute table making each
        feature (AOI) uniquely identifiable. If None (default) the features
        are labelled by a unique-identifier created on the fly.
    :attrib mapping_configs:
        Mapping configurations specified by `~agrisatpy.operational.mapping.MapperConfigs`.
        Uses default configurations if not provided.
    :attrib observations:
        data structure for storing DB query results per AOI.
    """

    def __init__(
            self,
            date_start: date,
            date_end: date,
            aoi_features: Union[Path, GeoDataFrame],
            unique_id_attribute: Optional[str] = None,
            mapper_configs: MapperConfigs = MapperConfigs()
        ):

        object.__setattr__(self, 'date_start', date_start)
        object.__setattr__(self, 'date_end', date_end)
        object.__setattr__(self, 'aoi_features', aoi_features)
        object.__setattr__(self, 'unique_id_attribute', unique_id_attribute)
        object.__setattr__(self, 'mapper_configs', mapper_configs)

        observations: Dict[str, DataFrame] = None
        object.__setattr__(self, 'observations', observations)

        features: Dict[str, Feature] = None
        object.__setattr__(self, 'features', features)

    def __setattr__(self, *args):
        raise TypeError('Mapper attributes are immutable')

    def __delattr__(self, *args):
        raise TypeError('Mapper attributes are immutable')

    def get_aoi_scenes(
            self,
            aoi_identifier: Any
        ) -> DataFrame:
        """
        Returns a ``DataFrame`` with all scenes found for a
        area of interest.

        NOTE:
            The scene count is termed ``raw_scene_count``. This
            highlights that the final scene count might be
            different due to orbit and spatial design pattern.

        :param aoi_identifier:
            unique identifier of the aoi. Must be the same identifier
            used during the database query
        :return:
            ``DataFrame`` with all scenes found for a given
            set of search parameters
        """

        try:
            return self.observations[aoi_identifier].copy()
        except Exception as e:
            raise KeyError(
                f'{aoi_identifier} did not return any results: {e}'
            )
