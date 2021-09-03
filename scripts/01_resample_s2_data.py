"""
sample script showing how to start a resampling job for Sentinel-2 data
(begin of the resampling and extraction pipeline)

Requirements:
    - having downloaded data from ESA/Copernicus and stored them locally (.SAFE)
    - having created a metadata file per year in the archive with the .SAFE datasets
    - having created a local target archive for storing the resampled, stacked data
    
"""

from pathlib import Path
from datetime import date
from agrisatpy.processing.resampling import exec_parallel


# tile = input('Select a tile to query (e.g., "T32TMT"): ')
# out_dir = input('Enter toplevel directory where outputs shall be stored (e.g., ./SAT/L2A): ')
# year = input('Specify year to process (e.g., 2019): ')
# date_start = input('Enter start date (format: %Y-%m-%d): ')
# date_end = input('Enter end date (format: %Y-%m-%d): ')
# n_threads = input('Enter numer of threads for parallel execution: ')

tile = 'T32TMT'
# out_dir = Path('/mnt/ides/Lukas/03_Debug/test_archive')
date_start = date(2020,1,1)
date_end = date(2020,1,31)
n_threads = 1

year = date_start.year
target_s2_archive = f'/run/media/graflu/ETH-KP-SSD6/SAT/L1C/{year}/{tile}'

# specify the number of threads

# further options as key-value pairs.
# pixel_division is a special approach that multiplies pixel values instead of doing an interpolation
# use the is_L2A keyword to specify the processing level of the data
# when setting is_mundi to False we assume that all ESA datasets are named .SAFE (Mundi breaks with this
# convention)
options = {'pixel_division': True,
           'is_L2A': False,
           'is_mundi': False
           }

exec_parallel(out_dir,
              date_start,
              date_end,
              n_threads,
              tile,
              **options
)
