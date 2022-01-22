'''
Sentinel-2 specific helper functions.
'''

import os
import glob
import pandas as pd

from datetime import date
from datetime import datetime
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
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
        dot_safe_name: Union[str,Path]
    ) -> ProcessingLevels:
    """
    Determines the processing level of a dataset in .SAFE format
    based on the file naming

    :param dot_safe_name:
        name of the .SAFE dataset
    :return:
        processing level of the dataset
    """

    if isinstance(dot_safe_name,Path):
        dot_safe_name = dot_safe_name.name

    if dot_safe_name.find('MSIL1C') >= 0:
        return ProcessingLevels.L1C
    elif dot_safe_name.find('MSIL2A') >= 0:
        return ProcessingLevels.L2A
    else:
        raise ValueError(
            f'Could not determine processing level for {dot_safe_name}'
        )


def get_S2_acquistion_time_from_safe(
        dot_safe_name: str
    ) -> date:
    """
    Determines the image acquisition time of a dataset in .SAFE format
    based on the file naming

    :param dot_safe_name:
        name of the .SAFE dataset
    :return:
        image acquistion time (full timestamp)
    """

    return datetime.strptime(dot_safe_name.name.split('_')[2], '%Y%m%dT%H%M%S')


def get_S2_acquistion_date_from_safe(
        dot_safe_name: str
    ) -> date:
    """
    Determines the image acquisition date of a dataset in .SAFE format
    based on the file naming

    :param dot_safe_name:
        name of the .SAFE dataset
    :return:
        image acquistion date
    """

    return get_S2_acquistion_time_from_safe(dot_safe_name).date()


def get_S2_platform_from_safe(
        dot_safe_name: str
    ) -> str:
    """
    Determines the platform (e.g., S2A) from the dataset in .SAFE format
    based on the file naming

    :param dot_safe_name:
        name of the .SAFE dataset
    :return:
        platform name
    """

    return dot_safe_name.name.split('_')[0]


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
    a ``agrisatpy.operational.resampling`` derived band-stack was passed. 

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
        # platform_level = fname_splitted[2]
        sensor = fname_splitted[3]
        file_pattern = f'{file_pattern_date}_{file_pattern_tile}*{sensor}*SCL*tiff'
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
        band_selection: List[Tuple[str, Union[int,float]]],
        is_l2a: Optional[bool]=True
    ) -> pd.DataFrame:
    '''
    Returns the file-paths to the jp2 files with the spectral band data
    from a .SAFE Sentinel-2 dataset. Supports .SAFE datasets in L1C and
    L2A processing level.

    :param in_dir:
        Directory where the search_string is applied. Here - for Sentinel-2 - the
        it is the .SAFE directory of S2 rawdata files.
    :param band_selection:
        list of tuples. Each tuple contains the name of Sentinel-2 band and
        its spatial resolution in meters.
    :param is_l2a:
        if True (default), assumes that the data is in Sentinel-2 L2A processing
        level. If False, assumes L1C processing level.
    :returns:
        pandas dataframe of jp2 files
    '''

    # L1C and L2A data are organized in a slightly different manner
    # by looping over the list of tuples provided the file-paths can be extracted
    band_list = []
    for item in band_selection:

        # unpack tuple
        band_name, band_res = item
        # save returned values to dict
        band_props = {}

        # search expression for the file depends on the processing level
        if is_l2a:
            search_expr = in_dir.joinpath(
                f'GRANULE/*/IMG_DATA/R{int(band_res)}m/T*_{band_name.upper()}_{int(band_res)}m.jp2'
            )
        else:
            search_expr = in_dir.joinpath(
                f'GRANULE/*/IMG_DATA/T*_{band_name.upper()}.jp2'
            )
        try:
            band_fpath = Path(glob.glob(str(search_expr))[0])
        except Exception as e:
            raise BandNotFoundError(
                f'Could not determine file-path of {band_name} from {in_dir.name}: {e}'
            )

        band_props['band_name'] = band_name
        band_props['band_path'] = band_fpath
        band_props['band_resolution'] = band_res

        band_list.append(band_props)

    # construct dataframe with all band entries and return
    band_df = pd.DataFrame(band_list)

    return band_df


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

