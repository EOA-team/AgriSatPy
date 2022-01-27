'''
Generic mapping module
'''

from datetime import date
from geopandas import GeoDataFrame
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union
from agrisatpy.io.sat_data_handler import SatDataHandler


class Mapper(object):
    """
    Generic Mapping class to extract raster data for a selection of areas of interest
    (AOIs) and time period.

    :param date_start:
        start date of the time period to consider (inclusive)
    :param date_end:
        end date of the time period to consider (inclusive)
    :param polygon_features:
        ``GeoDataFrame`` or any vector file understood by ``fiona`` with
        geometries of type ``Polygon`` or ``MultiPolygon`` defining the Areas
        Of Interest (AOIs) to extract (e.g., agricultural field parcels).
        Each feature will be returned separately in a dict-like structure.
    :param unique_id_attribute:
        attribute in the `polygon_features`'s attribute table making each
        feature (AOI) uniquely identifiable. If None (default) the features
        are labelled by a unique-identifier created on the fly.
    """

    def __init__(
            self,
            date_start: date,
            date_end: date,
            polygon_features: Union[Path, GeoDataFrame],
            unique_id_attribute: Optional[str] = None,
        ):

        self.date_start = date_start
        self.date_end = date_end
        self.polygon_features = polygon_features
        self.unique_id_attribute = unique_id_attribute
        self.feature_stack = {}


    def get_observation(
            self,
            obs_date: date
        ) -> SatDataHandler:
        """
        Returns the ``SatDataHandler`` instance whose sensing date is closest
        to the selected date

        :param obs_date:
            date for which to select the closest observation
        """

        pass

    def reduce(
            self,
            reducer_list: List[str]
        ) -> GeoDataFrame:
        """
        Reduces the raster bands currently loaded for each feature and date
        available to a scalar using ``numpy``. Each ``numpy`` function that
        takes a 2d array as input and returns a single scalar as output can
        be used.

        To get, for instance, the mean and median pass either
        ``reducer_list=['mean', 'median']`` or
        ``reducer_list=['nanmean', 'nanmedian']``
        if the band data contains NaNs.

        :param reducer_list:
            list of ``numpy`` function names to evaluate. Must take a 2d array
             as input and return a scalar
        """

        pass


    def to_xarray(
            self
        ):
        pass
    

