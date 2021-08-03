"""
sample scripts extracting pixel values (i.e., spectra) from Sentinel-2 files.

Pixel values are extracted tile- and year-wise.

Requirements:
    - Having created a SAT-data archive following the AgriSatPy conventions
    - Having run 01_resample_s2_data.py for at least one tile and sensing date
"""

import os
import pandas as pd
from datetime import datetime
from agrisatpy.processing.extraction import S2bandstack2table
from agrisatpy.config import get_settings

Settings = get_settings()


# define inputs
tiles = ['T31TGM', 'T32TMT']
shp_dir = '/mnt/ides/Lukas/02_Research/PhenomEn/01_Data/01_ReferenceData/'
files_shp = {
    'T31TGM': os.path.join(shp_dir, 'lac_lemon_agroscope_epsg32631.shp'),
    'T32TMT': os.path.join(shp_dir, 'zurich_agroscope_epsg32632.shp')
    }
year = 2019
s2_archive = f'/run/media/graflu/ETH-KP-SSD6/SAT/{year}'

id_column = 'FID'
buffer = -20

# define date range for which values should be extracted
date_start = f'{year}-10-01'
date_end = f'{year}-12-31'

# loop over tiles
for tile in tiles:

    s2_tile_dir = os.path.join(s2_archive, tile)
    os.chdir(s2_tile_dir)

    # read the shapefile containing the field geometries
    in_file_polys = files_shp[tile]

    # make an ouptut directory for storing the CSV files if necessary
    out_dir = os.path.join(s2_tile_dir, Settings.SUBDIR_PIXEL_CSVS)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # read metadata file and filter by date
    metadata = pd.read_csv(Settings.RESAMPLED_METADATA_FILE)
    date_start_dt = datetime.strptime(date_start, Settings.DATE_FMT_INPUT)
    date_end_dt = datetime.strptime(date_end, Settings.DATE_FMT_INPUT)
    metadata = metadata[pd.to_datetime(metadata.SENSING_DATE).between(date_start_dt,
                                                              date_end_dt,
                                                              inclusive=True)]
    metadata = metadata.sort_values(by='SENSING_DATE')

    # loop over the sensing dates available
    for idx in range(metadata.shape[0]):
        
        date = metadata.SENSING_DATE.iloc[idx]
        in_file_scl = metadata.FPATH_SCL.iloc[idx]
        bandstack = metadata.FPATH_BANDSTACK.iloc[idx]

        # extract the pixel values
        refl, scl_stats = S2bandstack2table(in_file=bandstack,
                                            in_file_scl=in_file_scl,
                                            in_file_polys=in_file_polys,
                                            buffer=buffer,
                                            id_column=id_column,
                                            product_date=date)

        # save data-frames to CSV files (if they contain data)
        out_file = os.path.join(out_dir, f'{os.path.splitext(bandstack)[0]}.csv')
        out_file_scl = os.path.join(out_dir, f'{os.path.splitext(bandstack)[0]}_SCL.csv')
        if not refl.empty:
            refl.to_csv(out_file, index=False)
            scl_stats.to_csv(out_file_scl, index=False)
        