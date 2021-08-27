'''
Created on Aug 3, 2021

@author: graflu
'''

import os
import glob
from typing import List
from datetime import date
from datetime import datetime
import pandas as pd

from agrisatpy.config import get_settings

class ArchiveNotFoundError(Exception):
    pass

class MetadataNotFoundError(Exception):
    pass

Settings = get_settings()


def create_processed_metadata(raw_data_archive: str,
                              target_s2_archive: str,
                              out_file: str,
                              date_start: date,
                              date_end: date,
                              tile: str,
                              additional_columns: List[str]=['SCENE_ID','SENSING_DATE','PRODUCT_URI', 'TILE'],
                              ):
    """
    Helper function to create metadata files for the spatially resampled, band-stacked
    Sentinel-2 files, only. This function can be used to recreate metadata of the processed
    data (actually, the processing pipeline will create the metadata file anyways)
    """
    # check the metadata in the raw data archive
    if not os.path.isdir(raw_data_archive):
        raise ArchiveNotFoundError(f'No such file or directory: {raw_data_archive}')

    # cd into raw_data_archive
    os.chdir(raw_data_archive)
    try:
        metadata_file = glob.glob(os.path.join(raw_data_archive, 'metadata*.csv'))[0]
    except Exception as e:
        raise MetadataNotFoundError(
            f'Could not find metadata*.csv file in {raw_data_archive}: {e}'
    )

    metadata_full = pd.read_csv(metadata_file)

    # filter by S2 tile
    metadata = metadata_full[metadata_full.TILE == tile].copy()

    # filter metadata by provided date range and tile
    metadata.SENSING_DATE = pd.to_datetime(metadata.SENSING_DATE)
    date_start = datetime.strptime(date_start, '%Y-%m-%d')
    date_end = datetime.strptime(date_end, '%Y-%m-%d')
    metadata = metadata[pd.to_datetime(metadata.SENSING_DATE).between(date_start,
                                                                      date_end,
                                                                      inclusive=True)]
    metadata = metadata.sort_values(by='SENSING_DATE')

    # drop duplicates based on the dates -> these are the so-called 'split-scenes'
    # because of the datastrip issue
    metadata = metadata.drop_duplicates(subset='SENSING_DATE', keep="last")

    # keep only those columns specified
    metadata = metadata[additional_columns]

    # search for band stacks in the target archive
    bandstacks = glob.glob(os.path.join(target_s2_archive, '*.tiff'))
    bandstacks = [os.path.basename(x) for x in bandstacks]
    bandstack_dates = [datetime.strptime(x.split('_')[0],'%Y%m%d') for x in bandstacks]
    bandstack_df = pd.DataFrame({'FPATH_BANDSTACK': bandstacks, 'DATE_BANDSTACK': bandstack_dates})

    # reconstruct the file names of the processed data
    metadata['FPATH_BANDSTACK'] = ''
    metadata['FPATH_RGB_PREVIEW'] = ''
    metadata['FPATH_SCL'] = ''
    for idx, record in metadata.iterrows():

        # find the resampled band stack
        try:
            fname_bandstack = bandstack_df[bandstack_df.DATE_BANDSTACK == record.SENSING_DATE] \
                ['FPATH_BANDSTACK'].values[0]
        except Exception as e:
            print(e)
            continue
        splitted = fname_bandstack.split('_')
        fname_scl = f'{splitted[0]}_{splitted[1]}_{splitted[3]}_SCL_{splitted[-1]}'
        
        metadata.loc[idx,'FPATH_BANDSTACK'] = fname_bandstack
        
        metadata.loc[idx,'FPATH_SCL'] = os.path.join(
                 Settings.SUBDIR_SCL_FILES,
                 fname_scl
        )
        metadata.loc[idx,'FPATH_RGB_PREVIEW'] = os.path.join(
            Settings.SUBDIR_RGB_PREVIEWS,
            f'{os.path.splitext(fname_bandstack)[0]}.png'
        )
    
    metadata.to_csv(os.path.join(target_s2_archive, out_file), index=False)


if __name__ == '__main__':
    
    # define a date range
    year = 2020
    date_start = f'{year}-01-01'
    date_end = f'{year}-12-31'
    
    # define a Sentinel2 tile
    tile = 'T32TMT'
    
    # define in and output directory
    # input: directory where the *.SAFE files are located
    raw_data_archive = f'/home/graflu/public/Evaluation/Projects/KP0022_DeepField/Sentinel-2/S2_L2A_data/CH/{year}'
    
    # output: directory where to store the resampled, band stacked Sentinel2 scenes
    # assuming the default AgriSatPy directory structure
    target_s2_archive = f'/run/media/graflu/ETH-KP-SSD6/SAT/L2A/{year}/{tile}'
    out_file = 'processed_metadata.csv'

    additional_columns = ['SCENE_ID','SENSING_DATE','PRODUCT_URI', 'TILE', 'SUN_ZENITH_ANGLE',
                          'SUN_AZIMUTH_ANGLE', 'SENSOR_ZENITH_ANGLE', 'SENSOR_AZIMUTH_ANGLE',
                          'SPACECRAFT_NAME']

    create_processed_metadata(raw_data_archive,
                              target_s2_archive,
                              out_file,
                              date_start,
                              date_end,
                              tile,
                              additional_columns
    )


