'''
Created on Sep 2, 2021

@author: graflu
'''

import glob
import pandas as pd
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from agrisatpy.config import Sentinel2

# global definition of spectral bands and their spatial resolution
s2 = Sentinel2()


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
        in_dir: Path
    ) -> Path:
    '''
    return the path to the S2 SCL (scene classification file) 20m resolution!

    :param search_dir 
        directory containing the SCL band files (jp2000 file).
    :return scl_file:
        SCL file-path
    '''
    search_pattern = "GRANULE/*/IM*/*/*_SCL_20m.jp2"
    scl_file = glob.glob(str(in_dir.joinpath(search_pattern)))[0]
    glob.glob(str(in_dir.joinpath(search_pattern)))[0]
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
    else:
        band_list = []
        for spatial_resolution in s2.SPATIAL_RESOLUTIONS.keys():
            if spatial_resolution not in resolution_selection: continue
            tmp_list = []
            for band_name in s2.SPATIAL_RESOLUTIONS[spatial_resolution]:
                tmp_list.extend(glob.glob(
                    str(in_dir.joinpath(f'GRANULE/*/IMG_DATA/*_{band_name}.jp2'))))
            band_list.append(tmp_list)
                
    # convert list of list to dictionary using resolutions as keys
    band_dict = dict.fromkeys(resolution_selection)
    for idx, key in enumerate(band_dict.keys()):
        band_dict[key] = band_list[idx]

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
