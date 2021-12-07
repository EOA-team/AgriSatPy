"""
This script shows how to download Sentinel-2 data
from Creodias for a given region of interest, date range,
cloud coverage and processing level.

It requires an account at Creodias (https://creodias.eu/) and
the account's username and password set in the environmental
variables as (example for Win)

SET CREODIAS_USER=<your-username>
SET CREODIAS_PASSWORD0<your-password>
"""


import geopandas as gpd
from datetime import date
from pathlib import Path
from agrisatpy.downloader.sentinel2.creodias import query_creodias
from agrisatpy.downloader.sentinel2.creodias import download_datasets
from agrisatpy.downloader.sentinel2.creodias import ProcessingLevels
from agrisatpy.downloader.sentinel2.utils import unzip_datasets


if __name__ == '__main__':
    
    # define inputs
    # processing level
    processing_level = ProcessingLevels.L1C
    # date range
    start_date = date(2016,4,20)
    end_date = date(2016,12,31)
    year = start_date.year

    # max_records defines the maximum number of datasets to download, increase if
    # necessary; however, CREODIAS might impose a limitation...
    max_records = 2000

    # filter by cloud cover (all scenes with a cloud cover lower than the threshold
    # will be downloaded)
    cloud_cover_threshold = 75

    # shapefile defining the bounds of your region of interest
    aoi_file = '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/Senseco/Intercomparison/Senseco-WG2-IntercomparisonStudyData_v1/Shapefiles/2015-2016/AUR/AUR.shp'
    bbox_data = gpd.read_file(aoi_file)

    # project to geographic coordinates (required for API query)
    bbox_data.to_crs(4326, inplace=True)
    # use the first feature (all others are ignored)
    bounding_box = bbox_data.geometry.iloc[0]

    # check for available datasets
    datasets = query_creodias(
        start_date=start_date,
        end_date=end_date,
        max_records=max_records,
        processing_level=processing_level,
        bounding_box=bounding_box,
        cloud_cover_threshold=cloud_cover_threshold
    )

    download_dir = r'/mnt/ides/Lukas/data/Senseco_Intercomparison/Sat_Data/FR/2016'
    download_datasets(datasets, download_dir)

    unzip_datasets(download_dir=Path(download_dir))

