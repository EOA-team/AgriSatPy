"""
sample scripts extracting pixel values (i.e., spectra) from Sentinel-2 files.

Pixel values are extracted tile- and year-wise.

Requirements:
    - Having created a SAT-data archive following the AgriSatPy conventions
    - Having run 01_resample_s2_data.py for at least one tile and sensing date
"""

import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from datetime import datetime
from agrisatpy.processing.extraction import S2bandstack2table
from agrisatpy.config import get_settings
from agrisatpy.utils import reconstruct_path

Settings = get_settings()
DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)

# connect to database for executing the metadata query


# filter DB by time, tile, processing_level


# define inputs
tiles = ['T32TLT']
path_shp_file = "O:/Projects/KP0022_DeepField/Kapitel_02/01_Data/Erntedaten_shp_buffer20/20170705_P21_Wintergerste_utm32_buffer20.shp"
id_column = 'FID'
buffer = 0
# define storage location where to store the extracted pixel values as CSV file(s)
out_dir = 'O:/Projects/KP0022_DeepField/Kapitel_02/01_Data/Debug_Pipeline'

# define date range for which values should be extracted
date_start = '2018-01-01'
date_end = '2018-12-31'

processing_level = 'Level-2A'
drop_multipolygon = True

# loop over tiles
for tile in tiles:

    # query the database
    query = f"""
    select
        proc.scene_id,
        proc.storage_device_ip, 
        proc.storage_device_ip_alias, 
        proc.storage_share, 
        proc.product_uri, 
        proc.bandstack, 
        proc.scl,
        raw.sensing_date
    from sentinel2_processed_metadata as proc
    left join sentinel2_raw_metadata as raw
    on proc.product_uri=raw.product_uri
    where
        raw.sensing_date between '{date_start}' and '{date_end}'
    and
        raw.tile_id = '{tile}'
    and
        raw.processing_level = '{processing_level}'
    order by sensing_date;
    """
    metadata = pd.read_sql(query, engine)

    # read the shapefile containing the field geometries
    in_file_polys = path_shp_file

    # loop over the sensing dates available
    for idx in range(metadata.shape[0]):
        
        date = metadata.sensing_date.iloc[idx]

        in_dir = reconstruct_path(
            record=metadata.iloc[idx],
            is_raw_data=False
        )

        in_file_scl = Path(in_dir).joinpath(metadata.scl.iloc[idx])
        bandstack = Path(in_dir).joinpath(metadata.bandstack.iloc[idx])

        # extract the pixel values
        try:
            refl, scl_stats = S2bandstack2table(in_file=bandstack,
                                                in_file_scl=in_file_scl,
                                                in_file_polys=in_file_polys,
                                                buffer=buffer,
                                                id_column=id_column,
                                                product_date=date,
                                                drop_multipolygon=drop_multipolygon
            )
            refl['scene_id'] = metadata.scene_id.iloc[idx]
            refl['product_uri'] = metadata.product_uri.iloc[idx]
    
            # save data-frames to CSV files (if they contain data)
            out_file = os.path.join(out_dir, f'{os.path.splitext(bandstack.name)[0]}.csv')
            out_file_scl = os.path.join(out_dir, f'{os.path.splitext(bandstack.name)[0]}_SCL.csv')

            if not refl.empty:
                refl.to_csv(out_file, index=False)
                scl_stats.to_csv(out_file_scl, index=False)

        except Exception as e:
            print(f'Could not extract pixel values for {in_file_polys}')
            continue
