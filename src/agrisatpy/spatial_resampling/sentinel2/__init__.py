'''
This module allows for spatial resampling of Sentinel-2 images to bring
all or a subset of Sentinel-2 bands into a common spatial resolution.
In the default use case, the function allows to resample the 20m Sentinel-2 bands
into 10m- Other scenarios (e.g., bringing all bands to 20m resolution or resampling
the 60m bands) are possible.
'''

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional
from typing import Union
from typing import List
from typing import Dict

from agrisatpy.io.sentinel2 import S2_Band_Reader
from agrisatpy.utils.sentinel2 import get_S2_processing_level
from agrisatpy.config import get_settings
from agrisatpy.processing import resampling
from agrisatpy.config.sentinel2 import Sentinel2
from agrisatpy.utils.constants.sentinel2 import s2_band_mapping


Settings = get_settings()
logger = Settings.logger

S2 = Sentinel2()

def _get_output_file_names(
        in_dir: Path,
        resampling_method: str,
        target_resolution: Union[int,float]
    ) -> Dict[str, str]:
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


def create_rgb(
        out_dir: Path,
        s2_stack: S2_Band_Reader,
        out_filename: str
    ) -> None:
    """
    Creates the RGB quicklook image (stored in a sub-directory).

    :param out_dir:
        directory where the band-stacked geoTiff are written to
    :param s2_stack:
        opened S2_Band_Reader with 'blue', 'green' and 'red' band
    :param out_filename:
        file name of the resulting RGB quicklook image (*.png)
    """

    # RGB previews are stored in their own sub-directory
    rgb_subdir = out_dir.joinpath(Settings.SUBDIR_RGB_PREVIEWS)
    if not rgb_subdir.exists():
        rgb_subdir.mkdir()

    fig_rgb = s2_stack.plot_rgb()
    fig_rgb.savefig(
        fname=rgb_subdir.joinpath(out_filename),
        bbox_inches='tight'
    )
    plt.close(fig_rgb)


def create_scl_preview(
        out_dir: Path,
        s2_stack: S2_Band_Reader,
        out_filename: str
    ) -> None:
    """
    Creates the SCL quicklook image (stored in a sub-directory).

    :param out_dir:
        directory where the band-stacked geoTiff are written to
    :param s2_stack:
        opened S2_Band_Reader with 'scl' band
    :param out_filename:
        file name of the resulting SCL quicklook image (*.png)
    """
    # SCL previews are stored in their own sub-directory alongside with the RGBs
    rgb_subdir = out_dir.joinpath(Settings.SUBDIR_RGB_PREVIEWS)
    if not rgb_subdir.exists():
        rgb_subdir.mkdir()

    fig_scl = s2_stack.plot_scl()
    fig_scl.savefig(
        fname=rgb_subdir.joinpath(out_filename),
        bbox_inches='tight'
    )
    plt.close(fig_scl)


def create_scl(
        out_dir: Path,
        s2_stack: S2_Band_Reader,
        out_filename: str
    ) -> None:
    """
    Creates the SCL raster datasets (stored in a sub-directory).

    :param out_dir:
        directory where the band-stacked geoTiff are written to
    :param s2_stack:
        opened S2_Band_Reader with 'scl' band
    :param out_filename:
        file name of the resulting SCL raster image (*.tiff)
    """
    
    scl_subdir = out_dir.joinpath(Settings.SUBDIR_SCL_FILES)
    if not scl_subdir.exists():
        scl_subdir.mkdir()

    s2_stack.write_bands(
        out_file=scl_subdir.joinpath(out_filename),
         band_selection=['scl']
    )


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
    single, stacked geoTiff. Creates also a RGB preview png-file of the scene and
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

    # read the data from .SAFE, keep the original datatype (uint16)
    s2_stack = S2_Band_Reader()
    try:
        s2_stack.read_from_safe(
            in_dir=in_dir,
            in_file_aoi=in_file_aoi,
            int16_to_float=False
        )
    except Exception as e:
        logger.error(e)
        return {}

    # check processing level
    if 'scl' in s2_stack.get_bandnames():
        is_l2a = True
    
    # resample the 20m bands to 10m using opencv2 or pixel_division
    if not pixel_division:

        # if L2A then do not resample the SCL layer using the custom
        # interpolation method
        bands_to_exclude = []
        if is_l2a:
            bands_to_exclude.append('scl')

        try:
            s2_stack.resample(
                target_resolution=target_resolution,
                resampling_method=resampling_method,
                bands_to_exclude=bands_to_exclude
            )
        except Exception as e:
            logger.error(e)
            return {}
        
        # handling of the SCL (Level-2A, only)
        if is_l2a:
            # resample SCL (if available) using pixel division for whole scenes
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
            # but for AOIs it's better to use NEAREST_NEIGHBOR_EXACT because this
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

    # check if image contains black-fill, only. In this case return
    if s2_stack.is_blackfilled():
        logger.info(f'{in_dir} contains blackfill, only')
        return

    # determine file names for output
    if not pixel_division:
        resampling_method = _get_resampling_name(resampling_method=resampling_method)
    else:
        resampling_method = 'pixel-division'

    out_file_names = _get_output_file_names(
        in_dir=in_dir,
        resampling_method=resampling_method,
        target_resolution=target_resolution
    )

    # plot RGB quicklook, errors here are considered non-critical
    try:
        create_rgb(
            out_dir=out_dir,
            out_filename=out_file_names['rgb_preview'],
            s2_stack=s2_stack
        )
    except Exception as e:
        logger.error(f'Could not generate RGB preview: {e}')

    # plot SCL if available and save it to geoTiff
    if is_l2a:

        # plot quicklook
        try:
            create_scl_preview(
                out_dir=out_dir,
                s2_stack=s2_stack,
                out_filename=out_file_names['scl_preview']
            )
        except Exception as e:
            logger.error(f'Could not generate SCL preview: {e}')
    
        # write scl dataset as geotiff
        try:
            create_scl(
                out_dir=out_dir,
                s2_stack=s2_stack,
                out_filename=out_file_names['scl']
            )
        except Exception as e:
            logger.error(e)
            return {}
    else:
        # remove unused entries for L1C
        out_file_names.pop('scl')
        out_file_names.pop('scl_preview')

    # write bandstack preserving the band names
    band_aliases = S2.BAND_INDICES
    # drop bands in 60m spatial resolution
    drop_bands = S2.SPATIAL_RESOLUTIONS.get(60.)
    band_aliases = [k for k in band_aliases.keys() if k not in drop_bands]
    band_selection = list(s2_band_mapping.values())
    # SCL file is written to its own file
    band_selection.remove('scl')

    try:
        out_file = out_dir.joinpath(out_file_names['bandstack'])
        s2_stack.write_bands(
            out_file=out_file,
            band_selection=band_selection,
            band_aliases=band_aliases
        )
    except Exception as e:
        logger.error(e)
        return {}

    return out_file_names


if __name__ == '__main__':

    in_dir = Path('/mnt/ides/Lukas/04_Work/S2A_MSIL2A_20190524T101031_N0212_R022_T32UPU_20190524T130304.SAFE')
    out_dir = Path('/mnt/ides/Lukas/03_Debug/Sentinel2/Resampling')

    pixel_division = True

    resample_and_stack_s2(
        in_dir=in_dir,
        out_dir=out_dir,
        pixel_division=pixel_division
    )
