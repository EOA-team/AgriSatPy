'''
Created on Aug 28, 2021

@author: Lukas Graf, D-USYS ETHZ
'''

from pathlib import Path
from enum import Enum
from typing import Optional
from datetime import date
import pandas as pd


class Platforms(Enum):
    Sentinel1: 'Sentinel-1'
    Sentinel2: 'Sentinel-2'
    Sentinel3: 'Sentinel-3'


def download_from_copernicus(footprint_geojson: Path,
                             date_start: date,
                             date_end: date,
                             platform: Platforms,
                             download_data: Optional[bool]=True,
                             **kwargs
                             ) -> pd.DataFrame:
    """
    This method can be used to download Sentinel (1,2,3) data
    from Copernicus Scientific Data Hub (DHUS) using the user
    provided credentials for calling the product search and
    download archive.

    NOTE: Since Copernicus changed its archive policy to a
    rolling-archive, older data might not be maintained in the
    SciHub archive (i.e., it is not "online") and has to be
    requested from the long-term archive (LTA). Alternatively,
    downloading from a national data access point or DIAS is
    recommended if the data requested is in the LTA.

    :param footprint_geojson:
        filepath to a geojson file containing the geometry for which
        to search and optionally download Sentinel data
    :param date_start:
        start date of the search period
    :param date_end:
        end date of the search period
    :param platform:
        name of the Sentinel platform (1,2,3) to download
    :param download_data:
        boolean flag indicating if found dataset shall be donwloaded
        or not. The default is True (i.e, download data).
    :param kwargs:
        additional keyword arguments to pass sentinelsat
    :return products:
        data frame with the found datasets (products).
    """
    # TODO
    pass

    
    