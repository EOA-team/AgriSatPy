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

from agrisatpy.downloader.sentinel2.creodias import ProcessingLevels
from agrisatpy.operational.cli import cli_s2_creodias_update


if __name__ == '__main__':

    # TODO: -> move to archive creation
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
    path_options = {
        'storage_device_ip': '//hest.nas.ethz.ch/green_groups_kp_public',
        'storage_device_ip_alias': '//nas12.ethz.ch/green_groups_kp_public',
        'mount_point': '/home/graflu/public/'
    }

    cli_s2_creodias_update(
        s2_raw_data_archive=s2_raw_data_archive,
        region=region,
        processing_level=processing_level,
        path_options=path_options
    )
