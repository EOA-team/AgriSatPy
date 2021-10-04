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
from agrisatpy.downloader.sentinel2.creodias import query_creodias
from agrisatpy.downloader.sentinel2.creodias import download_datasets
from agrisatpy.downloader.sentinel2.creodias import ProcessingLevels


if __name__ == '__main__':
    
    # define inputs
    # processing level
    processing_level = ProcessingLevels.L1C
    # date range
    start_date = date(2019,1,1)
    end_date = date(2019,1,31)
    # max_records defines the maximum number of datasets to download, increase if
    # necessary; however, CREODIAS might impose a limitation...
    max_records = 200

    # shapefile defining the bounds of your region of interest
    aoi_file = '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Uncertainty/STUDY_AREA/AOI_Esch_EPSG32632.shp'
    bbox_data = gpd.read_file(aoi_file)

    # project to geographic coordinates (required for API query)
    bbox_data.to_crs(4326, inplace=True)
    # use the first feature (all others are ignored)
    bounding_box = bbox_data.geometry.iloc[0]

    # check for available datasets
    datasets = query_creodias(
        start_date,
        end_date,
        max_records,
        processing_level,
        bounding_box
    )

    download_dir = '/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Rawdata/L1C/CH/2019'
    download_datasets(datasets, download_dir)
