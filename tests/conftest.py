'''
Global pytest fixtures
'''

import os
import pytest
import requests

from distutils import dir_util
from pathlib import Path
from agrisatpy.downloader.sentinel2.utils import unzip_datasets


@pytest.fixture
def tmppath(tmpdir):
    '''
    Fixture to make sure that test function receive proper
    Posix or Windows path instead of 'localpath'
    '''
    return Path(tmpdir)


@pytest.fixture
def datadir(tmppath, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    Taken from stackoverflow
    https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    (May 6th 2021)
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmppath))

    return tmppath


@pytest.fixture()
def get_s2_safe_l2a():
    """
    Get Sentinel-2 testing data in L2A processing level. If not available yet
    download the data from the Menedely dataset link provided
    """

    def _get_s2_safe_l2a():

        testdata_dir = Path('../../data')
        testdata_fname = testdata_dir.joinpath('S2A_MSIL2A_20190524T101031_N0212_R022_T32UPU_20190524T130304.SAFE')
    
        # download URL
        url = 'https://data.mendeley.com/public-files/datasets/ckcxh6jskz/files/e97b9543-b8d8-436e-b967-7e64fe7be62c/file_downloaded'
    
        if not testdata_fname.exists():
        
            # download dataset
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(testdata_fname, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=5096):
                    fd.write(chunk)
        
            # unzip dataset
            unzip_datasets(download_dir=testdata_dir)
            
        return testdata_fname

    return _get_s2_safe_l2a
