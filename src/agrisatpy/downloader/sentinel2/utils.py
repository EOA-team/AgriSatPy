'''
Created on Oct 29, 2021

@author: graflu
'''

import os
import glob
import subprocess
from pathlib import Path
from typing import Optional

from agrisatpy.config import get_settings

Settings = get_settings()
logger = Settings.logger


def unzip_datasets(
        download_dir: Path,
        remove_zips: Optional[bool]=True
    ) -> None:
    """
    Helper function to unzip downloaded Sentinel-2 L1C scenes
    once they are downloaded from CREODIAS. Works currently on
    *nix system only and requires `unzip` to be installed on the
    system.

    :param download_dir:
        directory where the zipped scenes in .SAFE format are located
    :param remove_zips:
        If set to False the zipped .SAFE scenes will be kept, otherwise
        (Default) they will be removed
    """

    # find zipped .SAFE archives
    dot_safe_zips = glob.glob(download_dir.joinpath('S2*.zip').as_posix())
    n_zips = len(dot_safe_zips)

    # change into the donwload directory
    current_dir = os.getcwd()

    # use unzip in subprocess call to unpack the zop files
    for idx, dot_safe_zip in enumerate(dot_safe_zips):

        os.chdir(download_dir)
        arg_list = ['unzip', '-n', dot_safe_zip]
        process = subprocess.Popen(
            arg_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, _ = process.communicate()

        logger.info(f'Unzipped {dot_safe_zip} ({idx+1}/{n_zips})')

        os.chdir(current_dir)
        if remove_zips:
            os.remove(dot_safe_zip)
