'''

'''


import os
import subprocess
import pandas as pd
from pathlib import Path
from typing import Optional
from typing import Union
from agrisatpy.utils.exceptions import DataNotFoundError


def _check_linux_cifs(ip: Union[str,Path]):
    """
    Searches for mount point of an external file system on a Linux
    operating system
    """
    
    res = Path('')

    exe = 'cat /proc/mounts | grep cifs'
    response = subprocess.getoutput(exe)
    lines = response.split('\n')
    for line in lines:
        if line.find(str(ip).strip()) >= 0:
            # data is on mounted share -> get local file system mapping
            local_path = response[response.find(str(ip).strip()):].split()[1]
            return Path(local_path)

    return res


def reconstruct_path(
        record: pd.Series,
        is_raw_data: Optional[bool] = True,
        path_to_nas: Optional[bool] = True,
    ) -> Path:
    """
    auxiliary function to reconstruct the actual dataset location
    based on the entries in the metatdata base. Raises an error
    if the dataset was not found.

    :param record:
        single record from the metadata base denoting a single dataset
    :param is_raw_data:
        if True (default) assumes the queried data is Sentinel-2 ESA
        derived "raw" data and not already processed by AgriSatPy to band
        stacks etc.
    :param path_to_nas:
        if True (default) tries to find the mount of the NAS file system
        on the local machine's file system.
    :return in_dir:
        filepath to the directory for the local machine
    """
    
    ip = Path(record.storage_device_ip)

    # check if ip points towards a network drive under Linux
    if path_to_nas:
        if os.name == 'posix':
    
            # search for mount point of NAS file system
            mount_point = _check_linux_cifs(ip=ip)
    
            # if this attempt failed, we can check the alias if available
            if str(mount_point) == '' and record.storage_device_ip_alias != '':
                mount_point = _check_linux_cifs(ip=record.storage_device_ip_alias)
    
            # if no mount point is found raise an error
            if str(mount_point) == '':
                raise DataNotFoundError(
                    'Couldnot find mount point for external file system'
                )
                
            share = mount_point.joinpath(record.storage_share)

        # Windows does not know about mount points, it should be able to work with network paths
        elif os.name == 'nt':
            
            share = Path(record.storage_device_ip).joinpath()
            # TODO: test alias if available
    

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
