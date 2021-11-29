'''
Created on Jul 9, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

from pydantic.main import BaseModel
from typing import List

# TODO integrate this module into the constants submodule in agrisatpy.utils
class Sentinel2(BaseModel):
    """
    base class defining Sentinel-2 product, archive and sensor details
    """

    PROCESSING_LEVELS: List[str] = ['L1C', 'L2A']

    SPATIAL_RESOLUTIONS: dict = {
        60.: ['B01', 'B09', 'B10'],
        10.: ['B02', 'B03', 'B04', 'B08'],
        20.: ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    }
    BAND_INDICES: dict = {
        'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4, 'B06': 5,
        'B07': 6, 'B08': 7, 'B8A': 8, 'B09': 9, 'B10': 10, 'B11': 11,
        'B12': 12
    }

    # define nodata values for ESA Sentinel-2 spectral bands (reflectance) and
    # scene classification layer (SCL)
    NODATA_REFLECTANCE: int = 64537
    NODATA_SCL: int = 254
    