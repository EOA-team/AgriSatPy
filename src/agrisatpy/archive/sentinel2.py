'''
Created on Jul 9, 2021

@author:    Gregor Perich & Lukas Graf (D-USYS, ETHZ)

@purpose:   This module is an easy and safe way to setup a Sentinel-2 data archive
            for storing resampled, bandstacked Sentinel-2 geoTiff files.
            It also supports cheching if the local archive contains all the datasets from Creodias.
            In case, datasets are missing, they can be downloaded ("pulled") from Creodias.

'''

import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
from typing import List
from datetime import date
from sqlalchemy import create_engine
from pathlib import Path

from agrisatpy.config import Sentinel2
from agrisatpy.utils.decorators import check_processing_level
from agrisatpy.utils.exceptions import RegionNotFoundError
from agrisatpy.utils.exceptions import ArchiveCreationError
from agrisatpy.downloader.sentinel2.creodias import query_creodias
from agrisatpy.downloader.sentinel2.creodias import download_datasets
from agrisatpy.downloader.sentinel2.creodias import ProcessingLevels
from agrisatpy.utils.constants.sentinel2 import ProcessingLevelsDB
from agrisatpy.config import get_settings


Settings = get_settings()
logger = Settings.logger

# connection to local metadata base
DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)

# object with information about Sentinel-2
s2 = Sentinel2()


@check_processing_level
def add_tile2archive(
        archive_dir: Path,
        processing_level: str,
        region: str,
        year: int,
        tile: str
    ) -> Path:
    """
    adds a new tile to an existing Sentinel-2 archive structure for
    storing bandstacked tiff files. Returns the path of the created
    tile directory.

    :param archive_dir:
        directory where the Sentinel2 data is stored. Must be the
        'archive root', i.e., the top-level directory under which the
        years and tiles are stored.
    :param processing_level:
        processing level of the data. Must be one of 'L1C', 'L2A'
    :param region:
        string-identifier of geographic region (e.g., CH for Switzerland)
    :param year
        year for which to create a sub-directories.
    :param tile:
        selected tile for which to create a storage directory.
    :return tile_dir:
        path of the directory created for storing the tile data
    """
    # check if the processing level is valid
    if processing_level not in s2.PROCESSING_LEVELS:
        raise ValueError(
            f'{processing_level} is not allowed for Sentinel-2. '\
            f'Must be one of {s2.PROCESSING_LEVELS})')
    # create a subdirectory for the processing level if it does not exist
    proc_dir = archive_dir.joinpath(processing_level)
    if not proc_dir.exists():
        try:
            os.mkdir(proc_dir)
        except Exception as e:
            raise ArchiveCreationError(f'Could not create {proc_dir}: {e}')

    # each region is a sub-directory
    if region == '':
        raise ValueError('Region identifier cannot be empty!')
    region_dir = proc_dir.joinpath(region)

    # each year is a sub-directory underneath the archive directory
    if year < 0:
        raise ValueError(f'{year} is not a valid value for a year')
    year_dir = region_dir.joinpath(str(year))

    # try to create a directory for the specified year if it does not
    # exist
    if not year_dir.exists():
        try:
            os.mkdir(year_dir)
        except Exception as e:
            raise ArchiveCreationError(f'Could not create {year_dir}: {e}')

    # try to create a directory for the tile if it does not exist
    tile_dir = year_dir.joinpath(tile)
    if not tile_dir.exists():
        try:
            os.mkdir(tile_dir)
        except Exception as e:
            raise ArchiveCreationError(f'Could not create {tile_dir}: {e}')
    return tile_dir
  

def create_archive_struct(
        in_dir: Path,
        processing_levels: List[str],
        regions: List[str],
        tile_selection: List[str],
        year_selection: List[int]
    ) -> None:
    """
    creates an empty archive file system structure for storing Sentinel-2
    bandstacks per tile and year.

    :param in_dir:
        directory where the Sentinel2 data is stored. Must be the
        'archive root', i.e., the toplevel directory under which the
        years and tiles are stored.
    :param processing_level:
        list of processing levels of the data.
    :param regions:
        list of geographic regions for which to create sub-directories
    :param tile_selection:
        list of tiles for which to create sub-directories
    :param year_selection:
        list of year for which to create sub-directories.
    """
    if not in_dir.exists():
        try:
            os.makedirs(in_dir)
        except Exception as e:
                logger.error(f'Could not create {in_dir}: {e}')
                sys.exit()

    # loop over processing levels
    for processing_level in processing_levels:
        # regions
        for region in regions:
            # tiles
            for tile in tile_selection:
                # years
                for year in year_selection:
                    # try to create an according storage directory for the current
                    # selection
                    try:
                        tile_dir = add_tile2archive(archive_dir=in_dir,
                                                    processing_level=processing_level,
                                                    region=region,
                                                    year=year,
                                                    tile=tile)
                    except Exception as e:
                        logger.error(f'Could not create {tile_dir}: {e}')
                        sys.exit()
                    logger.info(f'Created {tile_dir}')


def pull_from_creodias(
        start_date: date,
        end_date: date,
        processing_level: ProcessingLevels,
        path_out: Path,
        region: str
    ) -> pd.DataFrame:
    '''
    Checks if CREODIAS has Sentinel-2 datasets not yet available locally
    and downloads these datasets.

    :param start_date:
        Start date of the database & creodias query
    :param end_date:
        End date of the database & creodias query
    :param processing_level:
        Select S2 processing level L1C or L2A
    :param path_out:
        Out directory where the additional data from creodias should be downloaded
    :param region:
        Region identifier of the Sentinel-2 archive. By region we mean a
        geographic extent (bounding box) in which the data is organized. The bounding
        box extent is taken from the metadata base based on the region identifier.
    :return:
        dataframe with downloaded datasets
    '''

    # select processing level for DB query
    processing_level_db = ProcessingLevelsDB[processing_level.name]

    # query database to get the bounding box of the selected region
    query = f"""
    SELECT
        region_geom AS geom
    WHERE
        region_identifier = {region};
    """
    region_gdf = gpd.read_postgis(query, engine)
    if region_gdf.empty:
        raise RegionNotFoundError(f'{region} is not defined in the metadata base!')

    bounding_box = region_gdf.geometry.iloc[0]
    bounding_box_wkt = bounding_box.to_wkt()

    # local database query
    query = f"""
        SELECT
            product_uri, cloudy_pixel_percentage
        FROM
            sentinel2_raw_metadata
        WHERE
            sensing_time between '{start_date}' and '{end_date}'
        AND
            processing_level = '{processing_level_db}'
        AND
            ST_Intersects(
                geom,
                ST_GeomFromText('{bounding_box_wkt}', 4326)
            );
    """
    meta_db_df = pd.read_sql(query, engine)

    # determine max_records and cloudy_pixel_perecentage thresholds from local DB
    max_records = 1.25 * meta_db_df.shape[0]
    # Creodias has a hard cap of 2001 max_records
    if max_records > 2000:
        max_records = 2000
        logger.info(f"No. of max. records greater than the allowed max of 2001. "
                    f"Query set to {max_records}")
    else:
        logger.info(f'Set number of maximum records for Creodias query to {max_records}')

    cloud_cover_threshold = int(meta_db_df['cloudy_pixel_percentage'].max())
    logger.info(f'Set cloudy pixel percentage for Creodias query to {cloud_cover_threshold}')

    # check for available datasets
    datasets = query_creodias(
        start_date=start_date,
        end_date=end_date,
        max_records=max_records,
        processing_level=processing_level,
        bounding_box=bounding_box,
        cloud_cover_threshold=cloud_cover_threshold
    )

    # get .SAFE from croedias datasets
    datasets['product_uri'] = datasets.properties.apply(lambda x: Path(x['productIdentifier']).name)

    # compare with records from local meta database and keep those records not available locally
    missing_datasets = np.setdiff1d(datasets['product_uri'].values,
                                    meta_db_df['product_uri'].values)
    datasets_filtered = datasets[datasets.product_uri.isin(missing_datasets)]

    # download those scenes not available in the local database from Creodias
    download_datasets(datasets_filtered, path_out)

    return datasets_filtered
