"""
sample script showing how to perform a simple metadata query to identify the
number of available scenes for a Sentinel-2 tile below a user-defined cloud
cover threshold on data already downloaded.

For calling the Copernicus archive (and optionally downloading data) refer
to the script './copernicus_archive_query.py'

The called function also plots the cloud cover over time.
"""

import os
from datetime import datetime
from agrisatpy.metadata import scene_selection

# user inputs
s2_archive = input('Enter path to Sentinel-2 directory: ')
metadata_file = input('Enter filepath to metadata file: ')
tile = input('Select a tile to query (e.g., "T32TMT"): ')
out_dir = input('Enter path where outputs shall be stored: ')
date_start = input('Enter start date (format: %Y-%m-%d): ')
date_end = input('Enter end date (format: %Y-%m-%d): ')
cc_treshold = input('Define cloud cover threshold (0-100 %): ')

# date range
date_start = datetime.strptime(date_start, '%Y-%m-%d')
date_end = datetime.strptime(date_end, '%Y-%m-%d')

# execute scene selection
scene_selection(metadata_file=metadata_file,
                tile=tile,
                cloudcover_threshold=cc_threshold,
                date_start=date_start,
                date_end=date_end,
                out_dir=out_dir
)