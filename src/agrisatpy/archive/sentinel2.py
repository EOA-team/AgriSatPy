'''
Created on Jul 9, 2021

@author:    Lukas Graf (D-USYS, ETHZ)
@purpose:   This module is an easy and safe way to setup a Sentinel-2 data archive
            for storing resampled, bandstacked Sentinel-2 geoTiff files
'''

import os
import sys
from typing import List
from pathlib import Path

from agrisatpy.config import Sentinel2
from agrisatpy.config import get_settings
from agrisatpy.utils.decorators import check_processing_level

Settings = get_settings()
logger = Settings.logger

s2 = Sentinel2()


class ArchiveCreationError(Exception):
    pass


@check_processing_level
def add_tile2archive(archive_dir: Path,
                     processing_level: str,
                     year: int,
                     tile: str
                     ) -> Path:
    """
    adds a new tile to an existing Sentinel-2 archive structure for
    storing bandstacked tiff files. Returns the path of the created
    tile directory.

    :param archive_dir:
        directory where the Sentinel2 data is stored. Must be the
        'archive root', i.e., the top-level directory under which the
        years and tiles are stored.
    :param processing_level:
        processing level of the data. Must be one of 'L1C', 'L2A'
    :param year
        year for which to create a sub-directories.
    :param tile:
        selected tile for which to create a storage directory.
    :return tile_dir:
        path of the directory created for storing the tile data
    """
    # check if the processing level is valid
    if processing_level not in s2.PROCESSING_LEVELS:
        raise ValueError(
            f'{processing_level} is not allowed for Sentinel-2. '\
            f'Must be one of {s2.PROCESSING_LEVELS})')
    # create a subdirectory for the processing level if it does not exist
    proc_dir = os.path.join(archive_dir, processing_level)
    if not os.path.isdir(proc_dir):
        try:
            os.mkdir(proc_dir)
        except Exception as e:
            raise ArchiveCreationError(f'Could not create {proc_dir}: {e}')
    # each year is a sub-directory underneath the archive directory
    
    if year < 0:
        raise ValueError(f'{year} is not a valid value for a year')
    year_dir = os.path.join(proc_dir, str(year))

    # try to create a directory for the specified year if it does not
    # exist
    if not os.path.isdir(year_dir):
        try:
            os.mkdir(year_dir)
        except Exception as e:
            raise ArchiveCreationError(f'Could not create {year_dir}: {e}')

    # try to create a directory for the tile if it does not exist
    tile_dir = os.path.join(year_dir, tile)
    if not os.path.isdir(tile_dir):
        try:
            os.mkdir(tile_dir)
        except Exception as e:
            raise ArchiveCreationError(f'Could not create {tile_dir}: {e}')
    return tile_dir
  

def create_archive_struct(in_dir: str,
                          processing_levels: List[str],
                          tile_selection: List[str],
                          year_selection: List[int]
                          ) -> None:
    """
    creates an empty archive file system structure for storing Sentinel-2
    bandstacks per tile and year.

    :param in_dir:
        directory where the Sentinel2 data is stored. Must be the
        'archive root', i.e., the toplevel directory under which the
        years and tiles are stored.
    :param processing_level:
        list of processing levels of the data.
    :param tile_selection:
        list of tiles for which to create sub-directories
    :param year_selection:
        list of year for which to create sub-directories.
    """
    if not os.path.isdir(in_dir):
        try:
            os.makedirs(in_dir)
        except Exception as e:
                logger.error(f'Could not create {in_dir}: {e}')
                sys.exit()

    # loop over processing levels
    for processing_level in processing_levels:
        # tiles
        for tile in tile_selection:
            # years
            for year in year_selection:
                # try to create an according storage directory for the current
                # selection
                try:
                    tile_dir = add_tile2archive(archive_dir=in_dir,
                                                processing_level=processing_level,
                                                year=year,
                                                tile=tile)
                except Exception as e:
                    logger.error(f'Could not create {tile_dir}: {e}')
                    sys.exit()
                logger.info(f'Created {tile_dir}')
