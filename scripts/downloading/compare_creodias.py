# -*- coding: utf-8 -*-
"""
Created on 16.11.2021 11:36

@author:    Lukas Graf & Gregor Perich (D-USYS, ETHZ)

@purpose:   This script shows how to check if data in the local
            satellite data archive contains all scenes available at
            CREODIAS. If missing scenes (i.e., scenes available at
            CREODIAS but not in the local archive) are detected, they
            are automatically downloaded.

            This script could be called by a cronjob e.g., once a week
            to keep the Sentinel-2 data archive up-to-date
"""

import shutil
from pathlib import Path
from datetime import date

from agrisatpy.config import get_settings
from agrisatpy.archive.sentinel2 import pull_from_creodias
from agrisatpy.downloader.sentinel2.creodias import ProcessingLevels
from agrisatpy.downloader.sentinel2.utils import unzip_datasets
from agrisatpy.metadata.sentinel2.parsing import parse_s2_scene_metadata
from agrisatpy.metadata.sentinel2.database.ingestion import metadata_dict_to_database

logger = get_settings().logger


def compare_creodias(
        s2_raw_data_archive: Path,
        region: str,
        processing_level: ProcessingLevels
    ) -> None:
    """
    Loops over an existing Sentinel-2 rawdata (i.e., *.SAFE datasets) archive
    and checks the datasets available locally with those datasets available at
    CREODIAS for a defined Area of Interest (AOI) and time period (we store the
    data year by year, therefore, we always check an entire year).

    The missing data (if any) is downloaded into a temporary download directory
    `temp_dl` and unzip. The data is then copied into the actual Sentinel-2
    archive and the metadata is extracted and ingested into the metadata base.

    IMPORTANT: Requires a CREODIAS user account (user name and password).

    :param s2_raw_data_archive:
        Sentinel-2 raw data archive (containing *.SAFE datasets) to monitor.
        Existing datasets must have been already ingested into the metadata
        base!
    :param region:
        AgriSatPy's archive philosophy organizes datasets by geographic regions.
        Each region is identified by a unique region identifier (e.g., we use
        `CH` for Switzerland) and has a geographic extent described by a polygon
        geometry (bounding box) in geographic coordinates (WGS84). The geometry
        also defines the geographic dimension of the CREODIAS query. It is
        stored as a entry in the metadata base.
    :param processing_level:
        Sentinel-2 processing level (L1C or L2A) to check.
    """

    # construct S2 archive path
    in_dir = s2_raw_data_archive.joinpath(processing_level.name).joinpath(region)

    # since the data is stored by year (each year is a single sub-directory) we
    # can simple loop over the sub-directories and do the check
    for path in Path(in_dir).iterdir():

        if path.is_dir():
    
            # get year automatically
            year = int(path.name)
    
            # create temporary download directory
            path_out = path.joinpath('temp_dl')
    
            if not path_out.exists():
                path_out.mkdir()

            # download data from CREODIAS
            downloaded_ds = pull_from_creodias(
                start_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
                processing_level=processing_level,
                path_out=path_out,
                region=region
            )

            # unzip datasets and remove the zips afterwards
            unzip_datasets(download_dir=path_out)

            # move the datasets into the actual SAT archive (on level up)
            error_happened = False
            errored_datasets = []
            error_msgs = []
            for _, record in downloaded_ds.iterrows():
                try:
                    shutil.move(record.dataset_name, '..')
                    # once the dataset is moved successfully parse its metadata and
                    # ingest it into the database
                    in_dir = path.joinpath(record.dataset_name)
                    scene_metadata, _ = parse_s2_scene_metadata(in_dir)
                    metadata_dict_to_database(scene_metadata)
                except Exception as e:
                    error_happened = True
                    errored_datasets.append(record.dataset_name)
                    error_msgs.append(e)

            # if everything worked without errors delete the temp_dl directory
            if not error_happened:
                shutil.rmtree(path_out)
            # else log the errored datasets
            else:
                with open(path_out.joinpath('datasets.errored'), 'w') as src:
                    src.write('dataset_name, error_message\n')
                    for error in list(zip(errored_datasets, error_msgs)):
                        src.write(error[0] + ',' + error[1] + '\n')


if __name__ == '__main__':

    # debug
    from agrisatpy.metadata.sentinel2.database import add_region

    aoi_file = Path("/home/graflu/public/Evaluation/Sentinel-2/Documentation/CH_Polygon/CH_bounding_box_wgs84.shp")
    region = 'CH'
    add_region(region_identifier=region, region_file=aoi_file)

    # define inputs
    processing_level = ProcessingLevels.L2A
    region = 'CH'
    s2_raw_data_archive = Path('/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Rawdata')

    compare_creodias(
        s2_raw_data_archive=s2_raw_data_archive,
        region=region,
        processing_level=processing_level
    )
