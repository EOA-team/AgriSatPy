'''
Created on Sep 2, 2021

@author: Lukas Graf (D-USYS; ETHZ)
'''

import os
import subprocess
import pandas as pd
from pathlib import Path
from typing import Optional


def reconstruct_path(
        record: pd.Series,
        is_raw_data: Optional[bool]=True
    ) -> Path:
    """
    auxiliary function to reconstruct the actual dataset location
    based on the entries in the metatdata base. Raises an error
    if the dataset was not found.

    :param record:
        single record from the metadata base denoting a single dataset
    :return in_dir:
        filepath to the directory for the local machine
    """
    
    ip = Path(record.storage_device_ip)

    # check if ip points towards a network drive under Linux
    if os.name == 'posix':
        ip = record.storage_device_ip_alias
        exe = 'cat /proc/mounts | grep cifs'
        response = subprocess.getoutput(exe)
        lines = response.split('\n')
        for line in lines:
            if line.find(ip.strip()) >= 0:
                # data is on mounted share -> get local file system mapping
                local_path = response[response.find(ip.strip()):].split()[1]
                del ip
                ip = Path(local_path)

    share = ip.joinpath(record.storage_share)

    if is_raw_data:
        in_dir = share.joinpath(record.product_uri)
    else:
        in_dir = share

    # the path should work 'as it is' on Windows machines by replacing the
    # slashes
    if os.name == 'nt':
        # TODO test on Windows
        tmp = str(in_dir)
        tmp = tmp.replace(r'//', r'\\').replace(r'/', os.sep)
        in_dir = Path(tmp)

    # handle products not ending with '.SAFE' (e.g., when data comes from Mundi)
    if not in_dir.exists():
        in_dir = Path(str(in_dir).replace('.SAFE',''))

        if not in_dir.exists():
            raise NotADirectoryError(f'Could not find {str(in_dir)}')
    
    return in_dir
