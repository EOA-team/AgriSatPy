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

from agrisatpy.operational.cli import cli_s2_pipeline_fun
from agrisatpy.utils.constants import ProcessingLevels


if __name__ == '__main__':

    # define tile, region, processing level and date range
    tile = 'T32TLT'
    processing_level = ProcessingLevels.L2A

    date_start = date(2017,11,1)
    date_end = date(2017,11,1)
    
    # specify the number of threads
    n_threads = 1

    # set output path
    processed_data_archive = Path('/mnt/ides/Lukas/debug/Processed')

    # some path handling stuff
    if os.name == 'nt':
        mount_point = 'O:'
        mount_point_replacement = 'Evaluation'
    else:
        username = os.environ.get('USER')
        mount_point = f'/home/{username}/public/'
        mount_point_replacement = ''
        
    path_options = {
        'storage_device_ip': '//hest.nas.ethz.ch/green_groups_kp_public',
        'storage_device_ip_alias': '//nas12.ethz.ch/green_groups_kp_public',
        'mount_point': mount_point,
        'mount_point_replacement': mount_point_replacement
    }

    resampling_options = {'pixel_division': False}

    cli_s2_pipeline_fun(
        processed_data_archive,
        date_start,
        date_end,
        tile,
        processing_level,
        n_threads,
        resampling_options,
        path_options
    )

