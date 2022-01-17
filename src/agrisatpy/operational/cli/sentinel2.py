'''
Scriptable high-level function interfaces to the Sentinel-2 processing pipeline and
related operational functionalities.

NOTE:
    All of the functions make use of AgriSatPy's metadata DB. Only those Sentinel-2
    scenes ingested into the database can be accessed!
'''

import shutil
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from datetime import date
from datetime import datetime
from typing import Optional
from typing import Dict
from typing import Union

from agrisatpy.operational.resampling.sentinel2 import exec_pipeline
from agrisatpy.operational.archive.sentinel2 import pull_from_creodias
from agrisatpy.metadata.sentinel2.database import meta_df_to_database
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels
from agrisatpy.downloader.sentinel2.utils import unzip_datasets
from agrisatpy.metadata.sentinel2.parsing import parse_s2_scene_metadata
from agrisatpy.metadata.sentinel2.database.ingestion import metadata_dict_to_database
from agrisatpy.config import get_settings
from agrisatpy.metadata.sentinel2.database.querying import find_raw_data_by_tile

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
    Function calling Sentinel-2 pre-processing pipeline for user-defined
    time period and Sentinel-2 tile. The pre-processing pipeline brings
    those Sentinel-2 bands with a spatial resolution of 20m into a spatial
    resolution of 10m and generates scene quicklooks. The function accepts
    Sentinel-2 input data in .SAFE format in L1C and L2A processing level.

    :param processed_data_archive:
        directory where to store the processed data
    :param date_start:
        date defining the begin of the time period to process
    :param date_end:
        date defining the end of the time period to process
    :param tile:
        Sentinel-2 tile to process (e.g., 'T32TLT')
    :param processing_level:
        Processing level of the Sentinel-2 data (accepts L1C and L2A)
    :param n_threads:
        number of threads to use for execution of the pipeline
    :param resampling_options:
        key-word arguments to pass to `~exex_pipeline`
    :param path_options:
        optional key-word arguments for handling filepaths of the resulting
        datasets

    Example
    ------
    # TODO!
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
        cloud_cover_threshold: Optional[int] = 100,
        path_options: Optional[Dict[str, str]] = None,
        overwrite_existing_zips: Optional[bool] = False
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
    :param cloud_cover_threshold:
        optional cloud cover threshold to filter out to cloudy scenes as integer
        between 0 and 100%.
    :param path_options:
        optional dictionary specifying storage_device_ip, storage_device_ip_alias
        (if applicable) and mount point in case the data is stored on a NAS
        and should be accessible from different operating systems or file systems
        with different mount points of the NAS share. If not provided, the absolute
        path of the dataset is used in the database.
    :param overwrite_existing_zips:
        if False (default) overwrites eventually existing zip files. If the download
        process was interrupted (e.g., due to a connection timeout) setting the flag
        to True can save time because datasets already downloaded are ignored. NOTE:
        The function does **not** check if a dataset was downloaded completely!

    Example
    -------

    .. code-block:: python

    from pathlib import Path
    from agrisatpy.operational.cli.sentinel2 import cli_s2_creodias_update
    from agrisatpy.utils.constants import ProcessingLevels

    # define processing level (usually L2A but also L1C works)
    processing_level = ProcessingLevels.L2A

    # specifiy region, we use Switzerland (the extent of Switzerland is defined in the database)
    region = 'CH'

    # the archive should be always mounted in the same way for each user
    user_name = '<your_username>'
    s2_raw_data_archive = Path(f'/home/{user_name}/public/Evaluation/Satellite_data/Sentinel-2/Rawdata')

    # file-system specific handling: this allows to store the paths in the database
    # in such way that the dataset paths can be found from Linux and Windows machines
    file_system_options = {
        'storage_device_ip': '<your_nas_ip>',
        'storage_device_ip_alias': '<alternative_nas_ip_if_any',
        'mount_point': f'/home/{user_name}/public/'
    }

    cli_s2_creodias_update(
        s2_raw_data_archive=s2_raw_data_archive,
        region=region,
        processing_level=processing_level,
        path_options=file_system_options
    )

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
                region=region,
                cloud_cover_threshold=cloud_cover_threshold
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


def cli_s2_scene_selection(
        tile: str,
        date_start: date,
        date_end: date,
        processing_level: ProcessingLevels,
        out_dir: Path,
        cloud_cover_threshold: Optional[Union[int, float]] = 100
    ) -> None:
    """
    Function to query the Sentinel-2 metadata using a set of search criteria, including
    filtering by date range, cloud cover and Sentinel-2 tile.

    As a result, a CSV file with those metadata entries fulfilling the criteria is
    returned plus a plot showing the cloud cover (extracted from the scene metadata)
    per image acquisition date.

    :param tile:
        Sentinel-2 tile for which the scene selection should be performed. For instance,
        'T32TMT'
    :param date_start:
        start date for filtering the data
    :param end_date:
        end data for filtering the data
    :param processing_level:
        Sentinel-2 processing level (L1C or L2A).
    :param out_dir:
        directory where to store the subset metadata CSV file and the cloud cover plot.
    :param cloud_cover_threshold:
        optional cloud cover threshold to filter out to cloudy scenes as integer
        between 0 and 100%.
    """

    # query metadata from database
    try:
        metadata = find_raw_data_by_tile(
            date_start=date_start,
            date_end=date_end,
            processing_level=processing_level,
            tile=tile,
            cloud_cover_threshold=cloud_cover_threshold
        )
    except Exception as e:
        logger.error(f'Metadata query for Sentinel-2 data failed: {e}')
        return

    # drop xml columns
    metadata.drop('mtd_tl_xml', axis=1, inplace=True)
    metadata.drop('mtd_msi_xml', axis=1, inplace=True)

    # calculate average cloud cover for the selected scenes
    cc_avg = metadata.cloudy_pixel_percentage.mean()

    # get timestamp of query execution
    query_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # write out metadata of the query as CSV
    metadata.to_csv(out_dir.joinpath(f'{query_time}_query.csv'), index=False)

    # Plot available scenes for query
    fig = plt.figure(
        figsize=(8, 6),
        dpi=300
    )
    ax = fig.add_subplot(111)
    ax.plot(metadata['sensing_date'], metadata['cloudy_pixel_percentage'], 
            marker = 'o', markersize = 10)
    ax.set_xlabel('Sensing Date')
    ax.set_ylabel('Cloud cover [%]')
    ax.set_ylim(0., 100.)
    ax.set_title(f'Tile {tile} - No. of scenes: {metadata.shape[0]}'
                 + '\n' + f'Average cloud cover: {np.round(cc_avg, 2)}%')
    plt.savefig(
        out_dir.joinpath(f'{query_time}_query_CCplot.png'), 
        bbox_inches="tight"
    )
    plt.close()
