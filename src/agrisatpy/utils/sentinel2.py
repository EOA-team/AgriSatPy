'''
Created on Sep 2, 2021

@author:      Lukas Graf (D-USYS, ETHZ)

@purpose:     Sentinel-2 specific helper functions.
'''

import os
import glob
import pandas as pd

from datetime import date
from datetime import datetime
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from agrisatpy.config import get_settings
from agrisatpy.config import Sentinel2
from agrisatpy.utils.exceptions import ArchiveNotFoundError, BandNotFoundError
from agrisatpy.utils.exceptions import MetadataNotFoundError
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels

# global definition of spectral bands and their spatial resolution
s2 = Sentinel2()
Settings = get_settings()


def get_S2_processing_level(
        dot_safe_name: str
    ) -> ProcessingLevels:
    """
    Determines the processing level of a dataset in .SAFE format
    based on the file naming

    :param dot_safe_name:
        name of the .SAFE dataset
    :return:
        processing level of the dataset
    """
    if dot_safe_name.find('MSIL1C'):
        return ProcessingLevels.L1C
    elif dot_safe_name.find('MSIL2A'):
        return ProcessingLevels.L2A
    else:
        raise ValueError(
            f'Could not determine processing level for {dot_safe_name}'
        )


def get_S2_bandfiles(
        in_dir: Path,
        resolution: Optional[int]=None,
        is_L2A: Optional[bool]=True
    ) -> List[Path]:
    '''
    returns all JPEG-2000 files (*.jp2) found in a dataset directory
    (.SAFE).

    :param search_dir:
        directory containing the JPEG2000 band files
    :param resolution:
        select only spectral bands with a certain spatial resolution.
        Works currently on Sentinel-2 Level-2A data, only.
    :return files:
        list of Sentinel-2 single band files
    '''
    if resolution is None:
        search_pattern = 'GRANULE/*/IM*/*/*B*.jp2'
    else:
        if is_L2A:
            search_pattern = f'GRANULE/*/IM*/R{int(resolution)}m/*B*.jp2'
        else:
            search_pattern = f'GRANULE/*/IM*/*B*.jp2'
    files = glob.glob(str(in_dir.joinpath(search_pattern)))
    return [Path(x) for x in files]


def get_S2_sclfile(
        in_dir: Path,
        from_bandstack: Optional[bool]=False,
        in_file_bandstack: Optional[Path]=None
    ) -> Path:
    '''
    return the path to the S2 SCL (scene classification file). The method
    either searches for the SCL file in .SAFE structure (default, returning
    SCL file in 20m spatial resolution) or the resampled SCL file in case
    a ``agrisatpy.processing.resampling`` derived band-stack was passed. 

    :param in_dir:
        either .SAFE directory (default use case) or the the directory
        containing the band-stacked geoTiff files (must have a sub-directory
        where the SCL files are stored)
    :param from_bandstack:
        if False (Default) assumes the data to be in .SAFE format. If True
        the data must be band-stacked resampled geoTiffs derived from
        AgriSatPy's processing pipeline
    :param in_file_bandstack:
        file name of the bandstack for which to search the SCL file. Must be
        passed if ``from_bandstack=True``
    :return scl_file:
        SCL file-path
    '''

    if not from_bandstack:
        # take SCL file in 20m spatial resolution
        search_pattern = str(in_dir.joinpath('GRANULE/*/IM*/*/*_SCL_20m.jp2'))

    else:
        # check if bandstack file was passed correctly
        if in_file_bandstack is None:
            raise ValueError(
                'If from_banstack then `in_file_bandstack` must not be None'
            )

        fname_splitted = in_file_bandstack.name.split('_')
        file_pattern_date = fname_splitted[0]
        file_pattern_tile = fname_splitted[1]
        sensor = fname_splitted[3]
        file_pattern = f'{file_pattern_date}_{file_pattern_tile}_{sensor}_SCL_*.tiff'
        search_pattern = str(in_dir.joinpath(Settings.SUBDIR_SCL_FILES).joinpath(file_pattern))

    try:
        scl_file = glob.glob(search_pattern)[0]
    except Exception as e:
        raise BandNotFoundError(
            f'Could not find SCL file based on "{search_pattern}": {e}'
        )
        
    return Path(scl_file)


def get_S2_bandfiles_with_res(
        in_dir: Path,
        resolution_selection: Optional[List[Union[int,float]]]=[10.0, 20.0],
        search_str: Optional[str]='*B*.jp2',
        is_L2A: Optional[bool]=True
    ) -> pd.DataFrame:
    '''
    Returns a selection of native resolution Sentinel-2 bands (Def.: 10, 20 m).
    Works on MSIL2A data (sen2core derived) but also allows to work on Sentinel2
    L1C data. In the latter case, the spatial resolution of the single bands is
    hard-coded since the L1C data structure does not allow to extract the spatial
    resolution from the file or directory name.

    :param in_dir:
        Directory where the search_string is applied. Here - for Sentinel-2 - the
        it is the .SAFE directory of S2 rawdata files.
    :param resolution_selection:
        list of Sentinel-2 spatial resolutions to process (Def: [10, 20] m)
    :param search_str:
        search pattern for Sentinel-2 band files
    :param is_L2A:
        if False, assumes that the data is in Sentinel-2 L1C processing level. The
        spatial resolution is then hard-coded for each spectral band. Default is True.
    :returns:
        pandas dataframe of jp2 files
    '''
    # search for files in subdirectories in case of L2A data
    if is_L2A:
        band_list = [
            glob.glob(
                str(in_dir.joinpath(f'GRANULE/*/IM*/*{int(x)}*/{search_str}')))
            for x in resolution_selection
        ]
        # convert list of list to dictionary using resolutions as keys
        band_dict = dict.fromkeys(resolution_selection)
        for idx, key in enumerate(band_dict.keys()):
            band_dict[key] = band_list[idx]
    else:
        band_dict = {}
        for spatial_resolution in s2.SPATIAL_RESOLUTIONS.keys():
            if spatial_resolution not in resolution_selection: continue
            tmp_list = []
            for band_name in s2.SPATIAL_RESOLUTIONS[spatial_resolution]:
                tmp_list.extend(glob.glob(
                    str(in_dir.joinpath(f'GRANULE/*/IMG_DATA/*_{band_name}.jp2'))))
            band_dict[spatial_resolution] = tmp_list

    # find the highest resolution
    highest_resolution = min(resolution_selection)
    # save the band numbers of those bands with the highest resolution
    if is_L2A:
        hires_bands = [x.split('_')[-2] for x in band_dict[highest_resolution]]
    else:
        hires_bands = [x.split('_')[-1].replace('jp2', '') \
                        for x in band_dict[highest_resolution]]
    
    # loop over the other resolutions and drop the downsampled high resolution
    # bands and keep only the native bands per resolution
    native_bands = band_dict[highest_resolution]
    resolution_per_band = [highest_resolution for x in hires_bands]
    for key in band_dict.keys():

        if key == highest_resolution:
            continue

        if is_L2A:
            lowres_bands = [(x.split('_')[-2], x) for x in band_dict[key]]
        else:
            lowres_bands = [(x.split('_')[-1].replace('.jp2', ''), x) \
                            for x in band_dict[key]]
        lowres_bands2keep = [x[1] for x in lowres_bands if x[0] not in hires_bands]
        native_bands.extend(lowres_bands2keep)
        
        lowres_resolution = [key for x in lowres_bands2keep]
        resolution_per_band.extend(lowres_resolution)

    if is_L2A:
        native_band_names = [x.split('_')[-2] for x in native_bands]
    else:
        native_band_names = [x.split('_')[-1].replace('.jp2','') \
                             for x in native_bands]
    native_band_df = pd.DataFrame(native_band_names, columns=['band_name'])
    native_band_df["band_path"] = native_bands
    native_band_df["band_resolution"] = resolution_per_band
    
    native_band_df = native_band_df.sort_values(by='band_name')

    # Sentinel-2 Band 8A needs "special" treatment in terms of band ordering
    if 'B8A' in native_band_df['band_name'].values:

        tmp_bandnums = [int(x.replace('B','')) for x in list(native_band_df['band_name'].values[0:-1])]
        indices2shift = [i for i,v in enumerate(tmp_bandnums) if v > 8]
        index_b8a = indices2shift[0]
        final_band_df = native_band_df.iloc[0:index_b8a]
        final_band_df = final_band_df.append(native_band_df[native_band_df['band_name']=='B8A'])
        final_band_df = final_band_df.append(native_band_df.iloc[indices2shift])
        return final_band_df
    else:
        return native_band_df


def get_S2_tci(
        in_dir: Path,
        is_L2A: Optional[bool]=True,
    ) -> Path:
    '''
    Returns path to S2 TCI (quicklook) img (10m resolution). Works for both
    Sentinel-2 processing levels ('L2A' and 'L1C').

    :param in_dir:
        .SAFE folder which contains Sentinel-2 data
    :param is_L2A:
        if False, it is assumed that the data is organized in L1C .SAFE folder
        structure. The default is True.
    :return file_tci:
        file path to the quicklook image
    '''
    file_tci = ''
    if is_L2A:
        file_tci = glob.glob(str(in_dir.joinpath('GRANULE/*/IM*/*10*/*TCI*')))[0]
    else:
        file_tci = glob.glob(str(in_dir.joinpath('GRANULE/*/IM*/*TCI*')))[0]
    return Path(file_tci)


def create_processed_metadata(
        
        raw_data_archive: str,
        target_s2_archive: str,
        out_file: str,
        date_start: date,
        date_end: date,
        tile: str,
        additional_columns: List[str]=['SCENE_ID','SENSING_DATE','PRODUCT_URI', 'TILE'],
    ) -> None:
    """
    Helper function to create metadata files for the spatially resampled, band-stacked
    Sentinel-2 files, only. This function can be used to recreate metadata of the processed
    data (actually, the processing pipeline will create the metadata file anyways)

    IMPORTANT: This function is deprecated is maintained for legacy reasons, only
    """

    raise DeprecationWarning('This function is deprecated!')
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

