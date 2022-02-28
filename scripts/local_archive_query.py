"""
sample script showing how to perform a simple metadata query to identify the
number of available scenes for a Sentinel-2 tile below a user-defined cloud
cover threshold on data already downloaded and ingested into the metadata base

For calling the Copernicus archive (and optionally downloading data) refer
to the script './copernicus_archive_query.py'

The called function also plots the cloud cover over time.
"""

import os
from datetime import datetime
from pathlib import Path
from agrisatpy.operational.cli import cli_s2_scene_selection
from agrisatpy.utils.constants import ProcessingLevels

# user inputs
tile = 'T32TMT'
processing_level = ProcessingLevels.L2A
out_dir = Path('/mnt/ides/Lukas/03_Debug')
date_start = '2020-01-01'
date_end = '2020-08-31'
cc_threshold = 80.

# date range
date_start = datetime.strptime(date_start, '%Y-%m-%d')
date_end = datetime.strptime(date_end, '%Y-%m-%d')

# execute scene selection
cli_s2_scene_selection(
    tile=tile,
    processing_level=processing_level,
    cloud_cover_threshold=cc_threshold,
    date_start=date_start,
    date_end=date_end,
    out_dir=Path(out_dir)
)
