# -*- coding: utf-8 -*-
"""
Created on 16.11.2021 11:36

@author: graflu & gperich


"""

import os
from pathlib import Path
from agrisatpy.archive.sentinel2 import pull_from_creodias
from agrisatpy.downloader.sentinel2.creodias import ProcessingLevels
from datetime import datetime

# ================= Loop over existing folders ===============
processing_level = ProcessingLevels.L2A
region = "CH"
in_dir = Path(f'O:/Satellite_data/Sentinel-2/Rawdata/{processing_level.name}/{region}')
aoi_file = Path("O:/Satellite_data/Sentinel-2/Documentation/CH_Polygon/CH_bounding_box_wgs84.shp")


# loop through all subdirectories (e.g. all years of locally stored data)
for path in Path(in_dir).iterdir():
    if path.is_dir():

        # get year automatically
        year = path.name

        # create temp download dir
        path_out = path.joinpath("tempDL")

        if not path_out.exists():
            path_out.mkdir()

        pull_from_creodias(
            start_date=datetime.date(year, 1, 1),
            end_date=datetime.date(year, 12, 31),
            processing_level=processing_level,
            path_out=path_out,
            aoi_file=aoi_file
        )


"""

# put this part into tutorial (eventually..., soon (tm))

# ================= Manual use =====================
start_date = datetime.strptime("2017-01-01", "%Y-%m-%d").date()
end_date = datetime.strptime("2017-12-31", "%Y-%m-%d").date()
processing_level = ProcessingLevels.L2A
path_out = Path("O:/Satellite_data/Sentinel-2/Rawdata")
aoi_file = Path("O:/Satellite_data/Sentinel-2/Documentation/CH_Polygon/CH_bounding_box_wgs84.shp")

pull_from_creodias(
    start_date = start_date,
    end_date = end_date,
    processing_level=processing_level,
    path_out=path_out,
    aoi_file = aoi_file
)
"""