"""
sample script showing how to start a resampling job for Sentinel-2 data
"""

from agrisatpy.processing.resampling import exec_parallel


# define a date range
year = 2020
date_start = f'{year}-08-11'
date_end = f'{year}-08-18'

# define a Sentinel2 tile
tile = 'T32TMT'

# define in and output directory
# input: directory where the *.SAFE files are located
raw_data_archive = f'/home/graflu/public/Evaluation/Projects/KP0022_DeepField/Sentinel-2/S2_L2A_data/CH/{year}'

# output: directory where to store the resampled, band stacked Sentinel2 scenes
# assuming the default AgriSatPy directory structure
target_s2_archive = f'/run/media/graflu/ETH-KP-SSD6/SAT/L2A/{year}/{tile}'

# specify the number of threads
n_threads = 4

# further options as key-value pairs.
# pixel_division is a special approach that multiplies pixel values instead of doing an
# interpolation
# use the is_L2A keyword to specify the processing level of the data
options = {'pixel_division': True,
           'is_L2A': True,
           'is_mundi': False
           }

exec_parallel(raw_data_archive,
              target_s2_archive,
              date_start,
              date_end,
              n_threads,
              tile,
              **options
)
