'''
The Sat_Data_Creator class allows to create (satellite) image raster data
with geo-referenzation from numpy.arrays.
'''

import geopandas as gpd

from typing import List
from typing import Optional

from agrisatpy.io import SatDataHandler



class Sat_Data_Creator(SatDataHandler):
    """
    class for creating new SatDataHandler-like objects.
    """

    @classmethod
    def from_dataframe(
            cls,
            gdf: gpd.GeoDataFrame,
            attribute_selection: Optional[List[str]] = None
        ):
        """
        Creates a new handler instance from a ``GeoDataFrame``

        :param gdf:
            ``GeoDataFrame`` with Point records to convert into a
            new handler instance
        :param attribute_selection:
            attributes (columns) of the ``GeoDataFrame`` to convert
            into raster bands. Each attribute is converted into a single
            band.
        """
        pass
