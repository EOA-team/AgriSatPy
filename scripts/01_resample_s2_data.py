"""
sample script showing how to start a resampling job for Sentinel-2 data
(begin of the resampling and extraction pipeline)

Requirements:
    - having downloaded data from ESA/Copernicus and stored them locally (.SAFE)
    - having created a metadata file per year in the archive with the .SAFE datasets
    - having created a local target archive for storing the resampled, stacked data

"""

import os
from pathlib import Path
from datetime import date
from agrisatpy.processing.resampling import exec_parallel
from agrisatpy.metadata.sentinel2.database import meta_df_to_database


if __name__ == '__main__':

    # define tile, region, processing level and date range
    tile = 'T32TLT'
    region = 'CH'
    processing_level = 'L2A'

    date_start = date(2017,11,1)
    date_end = date(2017,11,1)
    
    # specify the number of threads
    n_threads = 1

    # database usage?
    use_database = True
    
    # set output path according to AgriSatPy conventions
    year = date_start.year
    target_s2_archive = Path(
        f'/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Processed/{processing_level}/{region}/{year}/{tile}'
    )
    # target_s2_archive = Path('/mnt/ides/Lukas/03_Debug/Sentinel2/pipeline')
    
    # further options as key-value pairs.
    # pixel_division is a special approach that multiplies pixel values instead of doing an interpolation
    # use the is_L2A keyword to specify the processing level of the data
    # when setting is_mundi to False we assume that all ESA datasets are named .SAFE (Mundi breaks with this
    # convention)
    options = {'pixel_division': False,
               'is_L2A': True
               }

    # start the processing
    metadata, failed_datasets = exec_parallel(
        target_s2_archive,
        date_start,
        date_end,
        n_threads,
        tile,
        use_database,
        **options
    )
    
    # set storage paths
    metadata['storage_device_ip'] = '//hest.nas.ethz.ch/green_groups_kp_public'
    metadata['storage_device_ip_alias'] = '//nas12.ethz.ch/green_groups_kp_public'
    metadata['storage_share'] = str(target_s2_archive)
    metadata['path_type'] = 'posix'
    # in case of win10, remove the datafolder O:/ 
    if os.name == "nt":
        metadata["storage_share"] = metadata["storage_share"].apply(lambda x: str(Path(x).as_posix()).replace("O:", "Evaluation"))
    # for unix, get user and remove datafolder
    else:
        username = os.environ.get('USER')
        metadata['storage_share'] = metadata['storage_share'].apply(lambda x: x.replace(f'/home/{username}/public/',''))
    
    # write to database (set raw_metadata option to False)
    meta_df_to_database(
        meta_df=metadata,
        raw_metadata=False
    )

    # save to CSV in addition
    metadata.to_csv(target_s2_archive.joinpath('metadata.csv'))

    failed_datasets.to_csv(target_s2_archive.joinpath('failed_datasets.csv'))
