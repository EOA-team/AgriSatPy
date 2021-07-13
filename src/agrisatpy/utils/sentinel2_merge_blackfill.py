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

from agrisatpy.spatial_resampling.sentinel2 import resample_and_stack_S2, scl_10m_resampling
from agrisatpy.config import get_settings

Settings = get_settings()


def identify_split_scenes(metadata_df: pd.DataFrame,
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
    return metadata_df[metadata_df["SENSING_DATE"].duplicated(keep=False)]


def find_rgb_preview(scene_out: str
                    ) -> str:
    """
    returns the file path to the RGB preview created by raster_resampling for
    a given stacked, resampled Sentinel-2 scene
    
    :param scene_out:
        file path to the band-stacked, resampled Sentinel-2 file. It is assumed
        that the RGB preview is located in a subdirectory underneath the band-
        stack ('rgb_previews') and named as the bandstack but ends with *.png
    """
    return os.path.join(
        os.path.abspath(os.path.join(scene_out, os.pardir)),
        os.path.join(Settings.SUBDIR_RGB_PREVIEWS,
                     f'{os.path.splitext(os.path.basename(scene_out))[0]}.png'))


def merge_split_files(in_file_1: str,
                      in_file_2: str,
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


def get_blackfill(in_file: str):
    """
    returns a bool array where each True elements indicates that
    pixel is blackfill. Using this information it is possible to
    fill black-filled values in one image with real values from the
    second image

    :param in_file:
        file path to the band-stack geoTiff file that containes
        blackfill (all reflectance values are zero)
    """
    with rio.open(in_file, 'r') as src:
        is_blackfill = src.read(1) == 0
    return is_blackfill


#TODO: include clipping by AOI logic
def merge_split_scenes(scene_1: str,
                       scene_2: str,
                       out_dir: str,
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
        directory where to create a "working_dir" subfolder to save the 
        intermediate products to.
    :param kwargs:
        key word arguments to pass to resample_and_stack_S2 and scl_10m_resampling.
        The out_dir option, however, is ignored
    """
    # check if working directory exists
    working_dir = os.path.join(out_dir, "temp_blackfill")
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    # save the outputs of the two scenes to different subdirectories within the working
    # directory to avoid to override the output
    out_dirs = [os.path.join(working_dir, '1'), os.path.join(working_dir, '2')]
    for out_dir in out_dirs:
        os.mkdir(out_dir)

    # do the resampling for the two scenes
    # first scene
    scene_out_1 = resample_and_stack_S2(in_dir=scene_1,
                                        out_dir=out_dirs[0],
                                        **kwargs)
    scl_out_1 = scl_10m_resampling(in_dir=scene_1,
                                   out_dir=out_dirs[0])

    # second scene
    scene_out_2 = resample_and_stack_S2(in_dir=scene_2,
                                        out_dir=out_dirs[1],
                                        **kwargs)
    scl_out_2 = scl_10m_resampling(in_dir=scene_2,
                                   out_dir=out_dirs[1])

    # find blackfilled pixels in the first image file
    is_blackfill = get_blackfill(in_file=scene_out_1)
    
    # get band descriptions (band names)
    with rio.open(scene_out_1, 'r') as src:
        band_names = list(src.descriptions)

    # merge band stacks
    meta, img_data = merge_split_files(in_file_1=scene_out_1,
                                       in_file_2=scene_out_2,
                                       is_blackfill=is_blackfill)
    # save output
    out_file = os.path.basename(scene_out_1)
    out_file = os.path.join(working_dir, out_file)
    with rio.open(out_file, 'w', **meta) as dst:
        for idx in range(meta['count']):
            dst.write(img_data[idx,:,:], idx+1)
            # set band names
            dst.set_band_description(idx+1, band_names[idx])

    # generate quicklook
    quicklook_1 = find_rgb_preview(scene_out_1)
    quicklook_2 = find_rgb_preview(scene_out_2)

    meta, rgb = merge_split_files(in_file_1=quicklook_1,
                                  in_file_2=quicklook_2,
                                  is_blackfill=is_blackfill)

    # save new overview to file
    out_file_rgb = os.path.join(working_dir, os.path.basename(quicklook_1))
    with rio.open(out_file_rgb, 'w', **meta) as dst:
        dst.write(rgb)              

    # merge SCL scenes
    meta, scl_data = merge_split_files(in_file_1=scl_out_1,
                                       in_file_2=scl_out_2,
                                       is_blackfill=is_blackfill)
    out_file_scl = os.path.join(working_dir, os.path.basename(scl_out_1))
    with rio.open(out_file_scl, 'w', **meta) as dst:
        dst.write(scl_data)

    # remove intermediate files (out_dirs)
    for out_dir in out_dirs:
        shutil.rmtree(out_dir)

    # save filepaths to dict and return
    return {'bandstack': out_file,
            'scl': out_file_scl,
            'preview': out_file_rgb}
