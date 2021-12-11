'''
Created on Dec 11, 2021

@author: graflu
'''

from pathlib import Path
from typing import Optional

from agrisatpy.io import Sat_Data_Reader
from agrisatpy.io.sentinel2 import S2_Band_Reader
from agrisatpy.utils.sentinel2 import get_S2_tci
from agrisatpy.utils.sentinel2 import get_S2_processing_level
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels


def read_s2_sclfile(
        in_dir: Path,
        in_file_aoi: Optional[Path] = None
    ) -> Sat_Data_Reader:
    """
    Reads the Sentinel-2 scene classification layer (SCL) file from
    a dataset in .SAFE format.

    IMPORTANT: The SCL file is available in Level-2 processing level,
    only

    :param in_dir:
        .SAFE Sentinel-2 archive in Level-2A
    :param in_file_aoi:
        optional vector geometry file defining an area of interest (AOI).
        If not provided, the entire spatial extent of the scene is read
    :return:
        ``Sat_Data_Reader`` with SCL band data
    """

    # read SCL file and return
    reader = S2_Band_Reader()
    reader.read_from_safe(
        in_dir=in_dir,
        in_file_aoi=in_file_aoi,
        band_selection=['B05']
    )
    reader.drop_band(band_name='red_edge_1')

    return reader


def read_s2_tcifile(
        in_dir: Path,
        in_file_aoi: Optional[Path] = None
    ) -> Sat_Data_Reader:
    """
    Reads the Sentinel-2 RGB quicklook file from a dataset in
    .SAFE format (processing levels L1C and L2A)

    :param in_dir:
        .SAFE Sentinel-2 archive
    :param in_file_aoi:
        optional vector geometry file defining an area of interest (AOI).
        If not provided, the entire spatial extent of the scene is read
    :return:
        ``Sat_Data_Reader`` with quicklook band data
    """

    # determine processing level first
    processing_level = get_S2_processing_level(dot_safe_name=in_dir)

    is_l2a = False
    if processing_level == ProcessingLevels.L2A:
        is_l2a = True

    try:
        tci_file = get_S2_tci(
            in_dir=in_dir,
            is_L2A=is_l2a
        )
    except Exception as e:
        raise Exception from e

    reader = Sat_Data_Reader()
    reader.read_from_bandstack(
        fname_bandstack=tci_file,
        in_file_aoi=in_file_aoi
    )
    # replace band names with red, green, blue
    try:
        reader.reset_bandnames(
            new_bandnames=['red', 'green', 'blue']
        )
    except Exception as e:
        raise Exception from e

    return reader

        