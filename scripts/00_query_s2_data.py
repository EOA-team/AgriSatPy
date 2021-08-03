"""
sample script showing how to perform a simple metadata query to identify the
number of available scenes for a Sentinel-2 tile below a user-defined cloud
cover threshold. The called function also plots the cloud cover over time.
"""

import os
from datetime import datetime

from agrisatpy.metadata import scene_selection


# read in metadata df
if os.name == 'posix':
    posix_user = os.environ.get('LOGNAME')
    project_drive = f'/home/{posix_user}/public/Evaluation/Projects'
else:
    project_drive = "O:/Projects"

year = 2019
s2_archive = os.path.join(project_drive, f'KP0022_DeepField/Sentinel-2/S2_L2A_data/CH/{year}')
metadata_file = os.path.join(s2_archive, f'metadata_CH_{year}.csv')

tile = 'T31TGM'
out_dir = f'/run/media/graflu/ETH-KP-SSD6/SAT/L2A/{year}/{tile}'

# date range
date_start = datetime.strptime(f'{year}-01-01', '%Y-%m-%d')
date_end = datetime.strptime(f'{year}-12-31', '%Y-%m-%d')

# cloud cover threshold
cc_threshold = 100.

scene_selection(metadata_file=metadata_file,
                tile=tile,
                cloudcover_threshold=cc_threshold,
                date_start=date_start,
                date_end=date_end,
                out_dir=out_dir
)