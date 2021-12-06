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
from typing import Dict
from typing import Optional
from datetime import datetime

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
        processing_level: ProcessingLevels,
        file_system_options: Optional[Dict[str, str]] = {}
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

    IMPORTANT: For database ingestion it is important to map the paths correctly
    and store them in the database in a way that allows accessing the data from
    all your system components (``file_system_options``).
    
    * In the easiest case, this the absolute path to the datasets (or URI)
    * If your data is stored on a NAS, you might specify the address of the NAS
      in the variable `storage_device_ip` and provide a `mount_point`, i.e.,
      the path (or drive on Windows) where the NAS is mounted into your local
      file system. Also aliasing of the is supported (`storage_device_ip_alias`),
      however, if possible try to avoid it.
    

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
    :param file_system_options:
        optional dictionary specifying storage_device_ip, storage_device_ip_alias
        (if applicable) and mount point in case the data is stored on a NAS
        and should be accessible from different operating systems or file systems
        with different mount points of the NAS share. If not provided, the absolute
        path of the dataset is used in the database.
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

            if downloaded_ds.empty:
                logger.info(f'No new datasets found for year {year} on CREODIAS')
                continue

            # unzip datasets and remove the zips afterwards
            unzip_datasets(download_dir=path_out)

            # move the datasets into the actual SAT archive (on level up)
            error_happened = False
            errored_datasets = []
            error_msgs = []
            parent_dir = path_out.parent
            for _, record in downloaded_ds.iterrows():
                try:
                    shutil.move(record.dataset_name, '..')
                    logger.info(f'Moved {record.dataset_name} to {parent_dir}')
                    # once the dataset is moved successfully parse its metadata and
                    # ingest it into the database
                    in_dir = path.joinpath(record.dataset_name)
                    scene_metadata, _ = parse_s2_scene_metadata(in_dir)

                    # some path handling if required
                    if file_system_options != {}:
                        scene_metadata['storage_device_ip'] = file_system_options.get('storage_device_ip','')
                        scene_metadata['storage_device_ip_alias'] = file_system_options.get('storage_device_ip_alias','')
                        mount_point = file_system_options.get('mount_point', None)
                        if mount_point is not None:
                            scene_metadata['storage_share'] = scene_metadata['storage_share'].replace(mount_point,'')

                    # database insert
                    metadata_dict_to_database(scene_metadata)
                    logger.info(f'Ingested scene metadata for {record.dataset_name} into DB')

                except Exception as e:
                    error_happened = True
                    logger.error(f'{record.dataset_name} produced an error: {e}')
                    errored_datasets.append(record.dataset_name)
                    error_msgs.append(e)

            # delete the temp_dl directory
            shutil.rmtree(path_out)

            # log the errored datasets if any error happened
            if error_happened:

                # get the timestamp (date) of the current run
                processing_date = datetime.now().date().strftime('%Y-%m-%d')

                # write to log directory
                log_dir = path_out.parent.joinpath('logs')
                if not log_dir.exists():
                    log_dir.mkdir()
                errored_logfile = log_dir.joinpath('datasets_errored.csv')
                if not errored_logfile.exists():
                    with open(errored_logfile, 'w') as src:
                        line = 'download_date,dataset_name,error_message\n'
                        src.writelines(line)
                with open(errored_logfile, 'a') as src:
                    for error in list(zip(errored_datasets, error_msgs)):
                        line = processing_date + ',' + error[0] + ',' + str(error[1])
                        src.writelines(line + '\n')


if __name__ == '__main__':

    # TODO -> move to archive creation
    # from agrisatpy.metadata.sentinel2.database import add_region
    #
    # aoi_file = Path("/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Documentation/CH_Polygon/CH_bounding_box_wgs84.shp")
    # region = 'CH'
    # add_region(region_identifier=region, region_file=aoi_file)

    # define inputs
    processing_level = ProcessingLevels.L2A
    region = 'CH'
    s2_raw_data_archive = Path('/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Rawdata')

    # file-system specific handling
    file_system_options = {
        'storage_device_ip': '//hest.nas.ethz.ch/green_groups_kp_public',
        'storage_device_ip_alias': '//nas12.ethz.ch/green_groups_kp_public',
        'mount_point': '/home/graflu/public/'
    }

    compare_creodias(
        s2_raw_data_archive=s2_raw_data_archive,
        region=region,
        processing_level=processing_level,
        file_system_options=file_system_options
    )
