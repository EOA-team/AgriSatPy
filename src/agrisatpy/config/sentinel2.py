'''
Created on Jul 9, 2021

@author: graflu
'''

from pydantic import BaseModel
from typing import List


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