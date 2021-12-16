'''
'''

import numpy as np
import rasterio as rio

from typing import Dict
from typing import Any


def get_raster_attributes(
        riods: rio.io.DatasetReader
    ) -> Dict[str,Any]:
    """
    extracts immutable raster attributes (not changed by reprojections,
    resampling) and returns them as a dict.

    Code taken from
    https://github.com/pydata/xarray/blob/960010b00119367ff6b82e548f2b54ca25c7a59c/xarray/backends/rasterio_.py#L359

    :param riods:
        opened dataset reader
    :return:
        dictionary with extracted raster attributes (attrs)
    """

    attrs = {}

    if hasattr(riods, "is_tiled"):
        # Is the TIF tiled? (bool)
        # We cast it to an int for netCDF compatibility
        attrs["is_tiled"] = np.uint8(riods.is_tiled)
    if hasattr(riods, "nodatavals"):
        # The nodata values for the raster bands
        attrs["nodatavals"] = tuple(
            np.nan if nodataval is None else nodataval for nodataval in riods.nodatavals
        )
    if hasattr(riods, "scales"):
        # The scale values for the raster bands
        attrs["scales"] = riods.scales
    if hasattr(riods, "offsets"):
        # The offset values for the raster bands
        attrs["offsets"] = riods.offsets
    if hasattr(riods, "descriptions") and any(riods.descriptions):
        # Descriptions for each dataset band
        attrs["descriptions"] = riods.descriptions
    if hasattr(riods, "units") and any(riods.units):
        # A list of units string for each dataset band
        attrs["units"] = riods.units

    return attrs
