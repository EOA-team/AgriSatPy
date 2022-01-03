'''
Helper functions to read Sentinel-2 TCI (RGB quicklook) and Scene Classification Layer
(SCL) file from a .SAFE dataset.
'''

from pathlib import Path
from typing import Optional

from agrisatpy.io import SatDataHandler
from agrisatpy.io.sentinel2 import Sentinel2Handler
from agrisatpy.utils.sentinel2 import get_S2_tci, get_S2_sclfile
from agrisatpy.utils.sentinel2 import get_S2_processing_level
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels


def read_s2_sclfile(
        in_dir: Path,
        in_file_aoi: Optional[Path] = None
    ) -> Sentinel2Handler:
    """
    Reads the Sentinel-2 scene classification layer (SCL) file from
    a dataset in .SAFE format.

    ATTENTION:
        The SCL file is available in Level-2 processing level, only

    :param in_dir:
        .SAFE Sentinel-2 archive in Level-2A
    :param in_file_aoi:
        optional vector geometry file defining an area of interest (AOI).
        If not provided, the entire spatial extent of the scene is read
    :return:
        ``SatDataHandler`` with SCL band data
    """

    # read SCL file and return
    reader = Sentinel2Handler()
    reader.read_from_safe(
        in_dir=in_dir,
        in_file_aoi=in_file_aoi,
        band_selection=['B05']
    )
    reader.drop_band(band_name='B05')

    return reader


def read_s2_tcifile(
        in_dir: Path,
        in_file_aoi: Optional[Path] = None
    ) -> Sentinel2Handler:
    """
    Reads the Sentinel-2 RGB quicklook file from a dataset in
    .SAFE format (processing levels L1C and L2A)

    :param in_dir:
        .SAFE Sentinel-2 archive
    :param in_file_aoi:
        optional vector geometry file defining an area of interest (AOI).
        If not provided, the entire spatial extent of the scene is read
    :return:
        ``SatDataHandler`` with quicklook band data
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

    reader = SatDataHandler()
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

        