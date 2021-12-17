'''
Scriptable high-level function interfaces to Sentinel-2 processing pipeline
'''

import shutil

from pathlib import Path
from datetime import date
from datetime import datetime
from typing import Optional
from typing import Dict

from agrisatpy.operational.resampling.sentinel2 import exec_pipeline
from agrisatpy.operational.archive.sentinel2 import pull_from_creodias
from agrisatpy.metadata.sentinel2.database import meta_df_to_database
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels
from agrisatpy.downloader.sentinel2.utils import unzip_datasets
from agrisatpy.metadata.sentinel2.parsing import parse_s2_scene_metadata
from agrisatpy.metadata.sentinel2.database.ingestion import metadata_dict_to_database
from agrisatpy.config import get_settings

logger = get_settings().logger


def cli_s2_pipeline_fun(
        processed_data_archive: Path,
        date_start: date,
        date_end: date,
        tile: str,
        processing_level: ProcessingLevels,
        n_threads: int,
        resampling_options: dict,
        path_options: Optional[dict] = None
    ) -> None:
    """
    Function calling Sentinel-2 processing pipeline (resampling and merging
    withhin single tiles

    TODO: add doc string
    """
    # start the processing
    metadata, failed_datasets = exec_pipeline(
        processed_data_archive=processed_data_archive,
        date_start=date_start,
        date_end=date_end,
        tile=tile,
        processing_level=processing_level,
        n_threads=n_threads,
        **resampling_options
    )
    
    # set storage paths
    metadata['storage_share'] = str(processed_data_archive)

    if path_options is not None:
        metadata['storage_device_ip'] = path_options.get('storage_device_ip', '')
        metadata['storage_device_ip_alias'] = path_options.get('storage_device_ip_alias', '')
        metadata['path_type'] = 'posix'
        mount_point = path_options.get('mount_point', '')
        mount_point_replacement = path_options.get('mount_point_replacement', '')
        metadata['storage_share'] = metadata["storage_share"].apply(
            lambda x: str(Path(x).as_posix()).replace(mount_point, mount_point_replacement)
        )

    # write to database (set raw_metadata option to False)
    meta_df_to_database(
        meta_df=metadata,
        raw_metadata=False
    )

    # write failed datasets to disk
    failed_datasets.to_csv(processed_data_archive.joinpath('failed_datasets.csv'))


def cli_s2_creodias_update(
        s2_raw_data_archive: Path,
        region: str,
        processing_level: ProcessingLevels,
        path_options: Optional[Dict[str, str]] = None
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
    :param path_options:
        optional dictionary specifying storage_device_ip, storage_device_ip_alias
        (if applicable) and mount point in case the data is stored on a NAS
        and should be accessible from different operating systems or file systems
        with different mount points of the NAS share. If not provided, the absolute
        path of the dataset is used in the database.
    """

    # since the data is stored by year (each year is a single sub-directory) we
    # can simple loop over the sub-directories and do the check
    for path in Path(s2_raw_data_archive).iterdir():

        if path.is_dir():
    
            # get year automatically
            year = int(path.name)
    
            # create temporary download directory
            path_out = path.joinpath(f'temp_dl_{year}')
    
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
                    if path_options != {}:
                        scene_metadata['storage_device_ip'] = path_options.get('storage_device_ip','')
                        scene_metadata['storage_device_ip_alias'] = path_options.get('storage_device_ip_alias','')
                        mount_point = path_options.get('mount_point', '')
                        mount_point_replacement = path_options.get('mount_point_replacement', '')
                        scene_metadata['storage_share'] = scene_metadata['storage_share'].replace(
                            mount_point, mount_point_replacement)

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

