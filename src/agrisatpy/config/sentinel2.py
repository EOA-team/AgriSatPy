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