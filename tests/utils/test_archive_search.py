'''
Created on May 6, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

import os
import pytest
import pandas as pd
from typing import Tuple

from agrisatpy.utils import search_data


@pytest.fixture()
def exec_archive_search():
    """
    fixture returning function prototype to execute
    the archive search (on resampled, band-stacked SAT data)
    """

    def _exec_archive_search(start_date: str,
                             end_date: str,
                             tile: str,
                             processing_level: str,
                             search_pattern: str,
                             search_pattern_scl: str,
                             in_dir: str
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        executes the actual archive search based on user-defined date
        range, tile and search patterns
        """
        df, df_scl = search_data(in_dir=in_dir,
                                 start_date=start_date,
                                 end_date=end_date,
                                 processing_level=processing_level,
                                 search_pattern=search_pattern,
                                 search_pattern_scl=search_pattern_scl,
                                 tile=tile)
        return df, df_scl

    return _exec_archive_search


@pytest.mark.parametrize('start_date, end_date, processing_level, tile',
                          [('2019-10-01', '2020-07-31', 'L2A', 'T31TGM'),
                           ('2019-10-01', '2020-07-31', 'L2A', 'T32TMT'),
                           ('2019-12-01', '2020-03-31', 'L2A', 'T31TGM'),
                           ('2019-12-01', '2020-03-31', 'L2A', 'T32TMT'),
                           ('2017-12-01', '2020-03-31', 'L2A', 'T32TMT')])
def test_archive_search_valid(datadir,
                              exec_archive_search,
                              start_date,
                              end_date,
                              processing_level,
                              tile
                              ):
    """
    tests archive search with a set of valid user inputs
    """

    search_pattern = '*10m.csv'
    search_pattern_scl = '*10m_SCL.csv'

    # conduct archive search
    df, df_scl = exec_archive_search(
        start_date=start_date,
        end_date=end_date,
        tile=tile,
        processing_level=processing_level,
        search_pattern=search_pattern,
        search_pattern_scl=search_pattern_scl,
        in_dir=datadir
    )

    # test assertions
    assert not df.empty, 'valid archive search returned no results'
    assert not df_scl.empty, 'valid archive search returned no results'


def test_non_existing_tile(datadir,
                           exec_archive_search,
                           ):
    """
    test archive search for non-existing tile and non-existing processing
    level
    """
    # non-existing tile
    tile = 'T40TGM'
    search_pattern = '*10m.csv'
    search_pattern_scl = '*10m_SCL.csv'
    start_date = '2019-01-01'
    end_date = '2019-01-31'
    processing_level = 'L2A'

    # conduct archive search
    df, df_scl = exec_archive_search(
        start_date=start_date,
        end_date=end_date,
        tile=tile,
        processing_level=processing_level,
        search_pattern=search_pattern,
        search_pattern_scl=search_pattern_scl,
        in_dir=datadir
    )
    
    assert df is None, 'non-existing tile returned non-empty dataframe'
    assert df_scl is None, 'non-existing tile returned non-empty dataframe'

    # non-existing processing level
    tile = 'T31TGM'
    search_pattern = '*10m.csv'
    search_pattern_scl = '*10m_SCL.csv'
    start_date = '2019-01-01'
    end_date = '2019-01-31'
    processing_level = 'L1C'

    # conduct archive search
    df, df_scl = exec_archive_search(
        start_date=start_date,
        end_date=end_date,
        tile=tile,
        processing_level=processing_level,
        search_pattern=search_pattern,
        search_pattern_scl=search_pattern_scl,
        in_dir=datadir
    )

    assert df is None, 'non-existing processing level returned non-empty dataframe'
    assert df_scl is None, 'non-existing processing level returned non-empty dataframe'

    # not-implemented processing level
    tile = 'T31TGM'
    search_pattern = '*10m.csv'
    search_pattern_scl = '*10m_SCL.csv'
    start_date = '2019-01-01'
    end_date = '2019-01-31'
    processing_level = 'L2C'

    # conduct archive search -> should raise an exception from the decorator
    with pytest.raises(Exception) as excinfo:   
        df, df_scl = exec_archive_search(
            start_date=start_date,
            end_date=end_date,
            tile=tile,
            processing_level=processing_level,
            search_pattern=search_pattern,
            search_pattern_scl=search_pattern_scl,
            in_dir=datadir
        ) 
    assert str(excinfo.value) == f"{processing_level} is not part of ['L1C', 'L2A']"

    assert df is None, 'not-defined processing level returned non-empty dataframe'
    assert df_scl is None, 'not-defined processing level returned non-empty dataframe'


def test_wrong_search_pattern(datadir,
                              exec_archive_search,
                              ):
    """
    test archive search with incorrect search pattern for files
    """
    tile = 'T32TMT'
    search_pattern = '*20m.csv'
    search_pattern_scl = '*10m_SCL.csv'
    start_date = '2019-01-01'
    end_date = '2019-01-31'
    processing_level = 'L2A'

    # conduct archive search
    df, df_scl = exec_archive_search(
        start_date=start_date,
        end_date=end_date,
        tile=tile,
        processing_level=processing_level,
        search_pattern=search_pattern,
        search_pattern_scl=search_pattern_scl,
        in_dir=datadir
    )
    
    assert df is None, 'incorrect file pattern returned non-empty dataframe'
    assert df_scl is None, 'incorrect file pattern returned non-empty dataframe'
