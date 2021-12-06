'''
Created on Jul 9, 2021

@author:    Gregor Perich and Lukas Graf (D-USYS, ETHZ)

@purpose:   Spatial resampling of raster images to 10m resolution
            developed for Sentinel2.
            It is also possible to resample from a higher to
            a lower spatial resolution.

            The module can handle Sentinel-2 data in L1C and L2A
            processing level.

@history:   Rewritten in v1.2 using object-oriented IO methods by
            Lukas Graf
'''

import cv2
import glob
import rasterio as rio
import rasterio.mask
import numpy as np

from pathlib import Path
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio import Affine
from typing import Optional
from typing import Union
from typing import List
from typing import Dict

from agrisatpy.utils.io.sentinel2 import S2_Band_Reader
from agrisatpy.utils.sentinel2 import get_S2_processing_level
from agrisatpy.config import get_settings
from agrisatpy.processing import resampling

Settings = get_settings()
logger = Settings.logger


def _get_output_file_names(
        in_dir: Path,
        resampling_method: str,
        target_resolution: Union[int,float]
    ) -> Dict[str]:
    """
    auxiliary method to get the output file names
    for the band-stack, the quicklooks and (if applicable) the
    SCL.

    The file-naming convention for the band-stack and previews is
    ``
    <date>_<tile>_<processing_level>_<sensor>_<resampling_method>_<spatial_resolution>
    ``

    :param in_dir:
        path of the .SAFE directory where the S2 data resides.
    :param resampling_method:
        name of the resampling method used
    :param target_resolution:
        spatial resolution of the output
    :return:
        dict with output file names
    """

    # get S2 UID
    s2_uid = in_dir.name

    splitted = s2_uid.split('_')
    date = splitted[2].split("T")[0]
    tile = splitted[-2]
    level = splitted[1]
    sensor = splitted[0]
    resolution = f'{int(target_resolution)}m'

    # define filenames
    basename = date + '_' + tile + '_' + level + '_' + \
        sensor + '_' + resampling_method + '_' + resolution

    return {
        'bandstack': f'{basename}.tiff',
        'rgb_preview': f'{basename}.png',
        'scl_preview': f'{basename}_SCL.png',
        'scl': f'{basename}_SCL.tiff'
    }


def _get_resampling_name(
        resampling_method: int
    ) -> str:
    """
    auxiliary method to map opencv's integer codes to meaningful resampling names
    (unknown if the method is not known)

    :param resampling_method:
        integer code from opencv2 for one of its image resizing methods
    :return:
        resampling method name or 'unknown' if the integer code cannot be
        translated
    """

    translator = {
        0: 'nearest',
        1: 'linear',
        2: 'cubic',
        3: 'area',
        4: 'lanczos',
        5: 'linear-exact',
        6: 'nearest-exact',
    }

    return translator.get(resampling_method, 'unknown')
    

def resample_and_stack_s2(
        in_dir: Path,
        out_dir: Path,
        target_resolution: Optional[Union[int, float]] = 10,
        resampling_method: Optional[int]=cv2.INTER_CUBIC,
        pixel_division: Optional[bool]=False,
        in_file_aoi: Optional[Path] = None
    ) -> Path:
    """
    Function to spatially resample a S2 scene in *.SAFE format and write it to a
    single stacked geoTiff. Creates also a RGB preview png-file of the scene and
    stores the scene classification layer that comes with L2A products in 10m spatial
    resolution.

    The function checks the processing level of the data (L1C or L2A) based on the
    name of the .SAFE dataset.

    Depending on the processing level the output will look a bit differently:
    
    * in L1C level (top-of-atmosphere) the band-stack of the spectral bands and
      the RGB quicklook is produced
    * in L2A level (bottom-of-atmosphere) the same inputs as in the L1C case
      are generated PLUS the scene classification layer (SCL) resampled to 10m
      spatial resolution

    The function takes the 10 and 20m Sentinel-2 bands, i.e., all bands but those
    with 60m spatial resolution.

    :param in_dir:
        path of the .SAFE directory where the S2 data resides.
    :param out_dir:
        path where to save the resampled & stacked geoTiff files to.
    :param target_resolution:
        target spatial resolution you want to resample to. The default is 10 (meters).
    :param resampling_method:
        The interpolation algorithm you want to use for spatial resampling. 
        The default is opencv's ``cv2.INTER_CUBIC``. See the opencv documentation for
        other options such ``cv2.INTER_NEAREST`` for nearest neighbor, etc.
    :param pixel_division:
        if set to True then pixel values will be divided into n*n subpixels 
        (only even numbers) depending on the target resolution. Takes the 
        current band resolution (for example 20m) and checks against the desired
        target_resolution and applies a scaling_factor. 
        This works, however, only if the spatial resolution is increased, e.g.
        from 20 to 10m. The ``resampling_method`` argument is ignored then.
        Default value is False.
    :param in_file_aoi:
        optional vector geometry file defining an Area of Interest (AOI). If provided
        the band-stack and the other products are only generated for the extent of
        that specific AOI.
    :return:
        dictionary with filepaths to bandstack, rgb_quicklook and (L2A, only) SCL
    """

    # read the data from .SAFE
    s2_stack = S2_Band_Reader()

    # determine processing level
    try:
        processing_level = get_S2_processing_level(dot_safe_name=in_dir.name)
    except Exception as e:
        logger.error(e)
        return {}

    # read data from .SAFE dataset
    try:
        s2_stack.read_from_safe(
            in_dir=in_dir,
            processing_level=processing_level,
            in_file_aoi=in_file_aoi,
            int16_to_float=False # keep original datatypes
        )
    except Exception as e:
        logger.error(e)
        return {}

    # resample the 20m bands to 10m using etiher
    # opencv2
    if not pixel_division:

        # if L2A then do not resample the SCL layer using the custom
        # interpolation method
        bands_to_exclude = []
        if processing_level.name == 'L2A':
            bands_to_exclude = ['scl']
        try:
            s2_stack.resample(
                target_resolution=target_resolution,
                resampling_method=resampling_method,
                bands_to_exclude=bands_to_exclude
            )
        except Exception as e:
            logger.error(e)
            return {}
        
        # resample SCL band using pixel division for whole scenes
        if in_file_aoi is None:
            try:
                s2_stack.resample(
                    target_resolution=target_resolution,
                    resampling_method=resampling_method,
                    pixel_division=True
                )
            except Exception as e:
                logger.error(e)
                return {}
        # but for AOIs it's better to use NEAREST_NEIGHBOR because this
        # method will snap the layer to the correct extent
        else:
            try:
                s2_stack.resample(
                    target_resolution=target_resolution,
                    resampling_method=cv2.INTER_NEAREST_EXACT,
                )
            except Exception as e:
                logger.error(e)
                return {}
    # or using pixel_divsion (all bands including scl layer if available)
    else:
        try:
            s2_stack.resample(
                target_resolution=target_resolution,
                pixel_division=True
            )
        except Exception as e:
            logger.error(e)
            return {}

    logger.info(f'Completed spatial resampling of {in_dir}')

    # check if image contains any valid pixel, if not return
    band_list = s2_stack.get_bandnames()
    test_band = s2_stack.get_band(band_name=band_list[0])
    if test_band.sum() == 0:
        logger.info('Image data seems to contain No-Data only')
        return {}

    # determine file name for output
    if not pixel_division:
        resampling_method = _get_resampling_name(resampling_method=resampling_method)
    else:
        resampling_method = 'pixel-division'

    out_file_names = _get_output_file_names(
        in_dir=in_dir,
        resampling_method=resampling_method,
        target_resolution=target_resolution
    )

    # create RGB and (if available) SCL quicklooks
    rgb_subdir = out_dir.joinpath(Settings.SUBDIR_RGB_PREVIEWS)
    if not rgb_subdir.exists():
        rgb_subdir.mkdir()

    fig_rgb = s2_stack.plot_rgb()
    fig_scl = s2_stack.plot_scl()
    fig_rgb.savefig(
        fname=rgb_subdir.joinpath(out_file_names['rgb_preview']),
        bbox_inches='tight'
    )
    fig_scl.savefig(
        fname=rgb_subdir.joinpath(out_file_names['scl_preview']),
        bbox_inches='tight'
    )

    # write bandstack preserving the band names
    # TODO ...
    



# if __name__ == '__main__':
#
#     in_dir = '/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Rawdata/L2A/CH/2018/S2A_MSIL2A_20180816T104021_N0208_R008_T32TLT_20180816T190612'
#     out_dir = '/mnt/ides/Lukas/03_Debug/Sentinel2/L1C/'
#     is_L2A = True
#
#     out_file = resample_and_stack_S2(
#         in_dir=Path(in_dir),
#         out_dir=Path(out_dir),
#         is_L2A=is_L2A
#     )
#
#     if is_L2A:
#         out_file_scl = scl_10m_resampling(
#             in_dir=Path(in_dir),
#             out_dir=Path(out_dir)
#         )
#
#     print(out_file)
