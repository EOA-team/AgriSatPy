"""
sample script showing how to query the Copernicus Sentinel archive
(Sentinel-1, 2, and 3) and optionally download data.

Note: This only works for data not transferred to the long term archive!
"""

from datetime import date
from pathlib import Path
from agrisatpy.downloader.sentinels import query_from_copernicus
from agrisatpy.downloader.sentinels import Platforms
from agrisatpy.downloader.sentinels import download_data


# user inputs
footprint_file = Path('/mnt/ides/Lukas/04_Work/ESCH_2021/AOI_Esch.shp')
date_start = date(2021,1,1)
date_end = date(2021,8,15)
platform = Platforms.Sentinel2

# filter by tile and processing level
options = {
    'tileid': '32TMT',
    'processinglevel': 'Level-1C'
}

# query products
products = query_from_copernicus(
        footprint_file,
        date_start,
        date_end,
        platform,
        **options
)

# download them
download_dir = Path('/tmp/')
download_data(products, download_dir)
