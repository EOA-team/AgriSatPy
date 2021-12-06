'''
Created on Jul 9, 2021

@author:    Lukas Graf (D-USYS, ETHZ)

@puprose:   Deals with split S2 tiles due to data take end/beginning causing
            scenes to be split into two datasets containing a significant amount
            of blackfill
'''

import os
import shutil
from typing import Tuple
import rasterio as rio
import numpy as np
import pandas as pd
from pathlib import Path

from .sentinel2 import resample_and_stack_S2
from .sentinel2 import scl_10m_resampling
from agrisatpy.config import get_settings

Settings = get_settings()
logger = Settings.logger


def identify_split_scenes(
        metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Sentinel-2 records data in so-called datatakes. When a datatake is over and a new
    begins the acquired image data is written to different files (based on the datatake
    time). Sometimes, this cause scenes of a single acquisition date to be split into
    two datasets which differ in their datatake. Thus, both datasets have a reasonable
    amount of blackfill in those areas not covered by the datatake they belong to. For
    users of satellite data, however, it is much more convenient to have those split
    scenes merged into one since the division into two scenes by the datatake has
    technical reasons only.

    With this function it is possible to identify those split scenes based on their
    sensing date. If two scenes from the same Sentinel-2 tile have the same sensing date
    then they are split by the datatake.

    :param metadata_df:
        dataframe containing extracted Sentinel-2 metadata (L1C or L2A level)
    """
    return metadata_df[metadata_df.sensing_date.duplicated(keep=False)]


def find_rgb_preview(
        scene_out: Path
    ) -> Path:
    """
    returns the file path to the RGB preview created by raster_resampling for
    a given stacked, resampled Sentinel-2 scene
    
    :param scene_out:
        file path to the band-stacked, resampled Sentinel-2 file. It is assumed
        that the RGB preview is located in a subdirectory underneath the band-
        stack ('rgb_previews') and named as the bandstack but ends with *.png
    :return:
        file-path to the RGB preview
    """
    return Path(os.path.join(
        os.path.abspath(os.path.join(str(scene_out), os.pardir)),
        os.path.join(Settings.SUBDIR_RGB_PREVIEWS,
                     f'{os.path.splitext(os.path.basename(scene_out))[0]}.png')))


def merge_split_files(
        in_file_1: Path,
        in_file_2: Path,
        is_blackfill: np.array
    ) -> Tuple[dict, np.array]:
    """
    takes two image files and fills the blackfilled values from the first
    file with values from the second file.
    Returns a tuple with the meta-dict and an array with the combined pixel
    values

    :param in_file_1:
        first band-stacked geoTiff with blackfill
    :param in_file_2:
        second band-stacked geoTiff with blackfill
    :param is_blackfill:
        bool array indicating the pixels in the first file that contain
        black fill
    """
    with rio.open(in_file_1, 'r') as src:
 
        # read all bands into memory
        meta = src.meta
        img_data_1 = src.read()
     
        # open the second file and read the data into memory
        with rio.open(in_file_2, 'r') as src2:
            img_data_2 = src2.read()

            for row in range(is_blackfill.shape[0]):
                if not any(is_blackfill[row,:]):
                    continue
                for col in range(is_blackfill.shape[1]):
                    if is_blackfill[row, col]:
                        img_data_1[:,row, col] = img_data_2[:,row, col]

    return (meta, img_data_1)


def get_blackfill(
        in_file: Path
    ) -> np.array:
    """
    returns a bool array where each True elements indicates that
    pixel is blackfill. Using this information it is possible to
    fill black-filled values in one image with real values from the
    second image

    :param in_file:
        file path to the band-stack geoTiff file that containes
        blackfill (all reflectance values are zero)
    :return is_blackfill:
        logical array indicating pixels that are black-filled
    """
    with rio.open(in_file, 'r') as src:
        is_blackfill = src.read(1) == 0
    return is_blackfill


def merge_split_scenes(
        scene_1: Path,
        scene_2: Path,
        out_dir: Path,
        is_L2A: bool,
        **kwargs
    ) -> dict:
    """
    merges two datasets of the same sensing date and tile split by the datatake beginning/
    end.
    First, both scenes are resampled to 10m and stacked in temporary working directory;
    second, they are merged together so that the blackfill of the first scenes is replaced
    by the values of the second one.
    Returns a dictionary of the merged files including the actual reflectance values,
    the preview RGB png-file and the scene classification layer (SCL)

    :param scene_1:
        .SAFE directory containing the first of two scenes split by the datatake beginning/
        end of Sentinel-2
    :param scene_2:
        .SAFE directory containing the first of two scenes split by the datatake beginning/
        end of Sentinel-2
    :param out_dir:
        directory where to create a "working_dir" sub-directory to save the 
        intermediate products to.
    :param is_L2A:
        boolean flag indicating if the data is L2A processing level (Default) or L1C
        (thus, no SCL file available)
    :param kwargs:
        key word arguments to pass to resample_and_stack_S2 and scl_10m_resampling.
        The out_dir option, however, is ignored
    """
    # check if working directory exists
    working_dir = out_dir.joinpath("temp_blackfill")
    if not working_dir.exists():
        os.mkdir(working_dir)

    # save the outputs of the two scenes to different subdirectories within the working
    # directory to avoid to override the output
    out_dirs = [working_dir.joinpath('1'), working_dir.joinpath('2')]
    for out_dir in out_dirs:
        if out_dir.exists():
            # for clean "start" of next loop iteration
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    
    # kwargs for AOI and masking (if applicable)
    masking = kwargs.get('masking', False)

    # do the spatial resampling for the two scenes
    # first scene
    scene_out_1 = resample_and_stack_S2(
        in_dir=scene_1,
        out_dir=out_dirs[0],
        is_L2A=is_L2A,
        **kwargs
    )

    scl_out_1 = ''
    if is_L2A:
        scl_out_1 = scl_10m_resampling(
            in_dir=scene_1,
            out_dir=out_dirs[0],
            **kwargs
    )

    # second scene
    scene_out_2 = resample_and_stack_S2(
        in_dir=scene_2,
        out_dir=out_dirs[1],
        is_L2A=is_L2A,
        **kwargs
    )

    scl_out_2 = ''
    if is_L2A:
        scl_out_2 = scl_10m_resampling(
            in_dir=scene_2,
            out_dir=out_dirs[1],
            **kwargs
        )

    # logic for masked scenes (they are already masked after "resample_and_stack_S2")
    if masking:
        file1_yes = scene_out_1.exists()
        file2_yes = scene_out_2.exists()
        
        # if 1 scene only has blackfill (e.g. does not exist) keep only this one
        if (file1_yes and not file2_yes) or (not file1_yes and file2_yes):
            if file1_yes:
                out_file = os.path.basename(str(scene_out_1))
                out_file = os.path.join(str(out_dirs[0]), out_file)
                out_file_scl = ''
                if is_L2A:
                    out_file_scl = os.path.join(
                        str(out_dirs[0]),
                        Settings.SUBDIR_SCL_FILES, 
                        os.path.basename(scl_out_1)
                    )
                quicklook = find_rgb_preview(scene_out_1)
                out_file_rgb = os.path.join(
                    str(out_dirs[0]),
                    Settings.SUBDIR_RGB_PREVIEWS, 
                    os.path.basename(quicklook)
                )
                return {'bandstack': out_file,
                        'scl': out_file_scl,
                        'preview': out_file_rgb}
            
            if file2_yes:
                out_file = os.path.basename(str(scene_out_2))
                out_file = os.path.join(str(out_dirs[1]), out_file)
                out_file_scl = ''
                if is_L2A:
                    out_file_scl = os.path.join(
                        str(out_dirs[1]),
                        Settings.SUBDIR_SCL_FILES, 
                        os.path.basename(scl_out_2)
                    )
                quicklook = find_rgb_preview(scene_out_2)
                out_file_rgb = os.path.join(
                    str(out_dirs[1]),
                    Settings.SUBDIR_RGB_PREVIEWS, 
                    os.path.basename(quicklook)
                )
                return {'bandstack': out_file,
                        'scl': out_file_scl,
                        'preview': out_file_rgb}
        
        # if no scenes exist, just return an empty dict
        elif (not file1_yes and not file2_yes):
            return {}

    # find blackfilled pixels in the first image file
    is_blackfill = get_blackfill(in_file=scene_out_1)
    
    # get band descriptions (band names)
    with rio.open(scene_out_1, 'r') as src:
        band_names = list(src.descriptions)

    # merge band stacks
    logger.info(f'Starting merging bandstack files {scene_out_1} and {scene_out_2}')
    meta, img_data = merge_split_files(
        in_file_1=scene_out_1,
        in_file_2=scene_out_2,
        is_blackfill=is_blackfill
    )
    # save output
    out_file = os.path.basename(scene_out_1)
    out_file = os.path.join(working_dir, out_file)
    with rio.open(out_file, 'w', **meta) as dst:
        for idx in range(meta['count']):
            dst.write(img_data[idx,:,:], idx+1)
            # set band names
            dst.set_band_description(idx+1, band_names[idx])
    logger.info(f'Finished merging bandstack files {scene_out_1} and {scene_out_2}')

    # generate quicklook
    quicklook_1 = find_rgb_preview(scene_out_1)
    quicklook_2 = find_rgb_preview(scene_out_2)

    logger.info(f'Starting merging quicklook files {quicklook_1} and {quicklook_2}')
    meta, rgb = merge_split_files(
        in_file_1=quicklook_1,
        in_file_2=quicklook_2,
        is_blackfill=is_blackfill
    )

    # save new overview to file
    out_file_rgb = os.path.join(working_dir, os.path.basename(quicklook_1))
    with rio.open(out_file_rgb, 'w', **meta) as dst:
        dst.write(rgb)              
    logger.info(f'Finished merging quicklook files {quicklook_1} and {quicklook_2}')

    # merge SCL scenes (L2A processing level ,only)
    out_file_scl = ''
    if is_L2A:
        logger.info(f'Starting merging SCL files {scl_out_1} and {scl_out_2}')
        meta, scl_data = merge_split_files(
            in_file_1=scl_out_1,
            in_file_2=scl_out_2,
            is_blackfill=is_blackfill
        )
        out_file_scl = os.path.join(working_dir, os.path.basename(scl_out_1))
        with rio.open(out_file_scl, 'w', **meta) as dst:
            dst.write(scl_data)
        logger.info(f'Finished merging SCL files {scl_out_1} and {scl_out_2}')

    # remove intermediate files (out_dirs) -> go into the working directory
    # and delete the subfolders there
    cwd = os.getcwd()
    os.chdir(cwd)
    os.chdir(working_dir)
    shutil.rmtree('1')
    shutil.rmtree('2')
    os.chdir(cwd)

    # save filepaths to dict and return
    return {'bandstack': Path(out_file).name,
            'scl': Settings.SUBDIR_SCL_FILES + "/" + Path(out_file_scl).name, 
            'preview': Settings.SUBDIR_RGB_PREVIEWS  + "/" + Path(out_file_rgb).name}
