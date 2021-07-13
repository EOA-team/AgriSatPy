"""
sample script showing how to start a resampling job for Sentinel-2 data
"""

from agrisatpy.processing.resampling import exec_parallel


# define in and output directory
# input: directory where the *.SAFE files are located
raw_data_archive = '/home/graflu/public/Evaluation/Projects/KP0022_DeepField/Sentinel-2/S2_L1C_data/ESCH/ESCH_2018/PRODUCT'
# output: directory where to store the resampled, band stacked Sentinel2 scenes
target_s2_archive = '/mnt/ides/Lukas/03_Debug/test_archive/L1C/2018/T32TMT'

# define a date range
date_start = '2018-05-01'
date_end = '2018-06-01'

# define a Sentinel2 tile
tile = 'T32TMT'

# specify the number of threads
n_threads = 4

# further options as key-value pairs.
# pixel_division is a special approach that multiplies pixel values instead of doing an
# interpolation
# use the is_L2A keyword to specify the processing level of the data
options = {'pixel_division': True,
           'is_L2A': False}

exec_parallel(raw_data_archive,
              target_s2_archive,
              date_start,
              date_end,
              n_threads,
              tile,
              **options
)
