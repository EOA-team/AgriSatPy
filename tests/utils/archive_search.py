'''
Created on May 6, 2021

@author: graflu
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
    the archive search
    """

    def _exec_archive_search(start_date: str,
                             end_date: str,
                             tile: str,
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
                             search_pattern=search_pattern,
                             search_pattern_scl=search_pattern_scl,
                             tile=tile)
        return df, df_scl

    return _exec_archive_search


@pytest.mark.parametrize('start_date, end_date, tile',
                          [('2019-10-01', '2020-07-31', 'T31TGM'),
                           ('2019-10-01', '2020-07-31', 'T32TMT'),
                           ('2019-12-01', '2020-03-31', 'T31TGM'),
                           ('2019-12-01', '2020-03-31', 'T32TMT'),
                           ('2017-12-01', '2020-03-31', 'T32TMT')])
def test_archive_search_valid(datadir,
                              exec_archive_search,
                              start_date,
                              end_date,
                              tile
                              ):
    """
    tests archive search with a set of valid user inputs
    """

    search_pattern = '*10m.csv'
    search_pattern_scl = '*10m_SCL.csv'

    df, df_scl = exec_archive_search(
        start_date,
        end_date,
        tile,
        search_pattern,
        search_pattern_scl,
        datadir
    )

    # test assertions
    assert not df.empty, 'valid archive search returned no results'
    assert not df_scl.empty, 'valid archive search returned no results'


def test_non_existing_tile(datadir,
                           exec_archive_search,
                           ):
    """
    test archive search for non-existing tile
    """
    tile = 'T40TGM'
    search_pattern = '*10m.csv'
    search_pattern_scl = '*10m_SCL.csv'
    start_date = '2019-01-01'
    end_date = '2019-01-31'

    df, df_scl = exec_archive_search(
        start_date,
        end_date,
        tile,
        search_pattern,
        search_pattern_scl,
        datadir
    )
    
    assert df is None, 'non-existing tile returned non-empty dataframe'
    assert df_scl is None, 'non-existing tile returned non-empty dataframe'


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

    df, df_scl = exec_archive_search(
        start_date,
        end_date,
        tile,
        search_pattern,
        search_pattern_scl,
        datadir
    )
    
    assert df is None, 'incorrect file pattern returned non-empty dataframe'
    assert df_scl is None, 'incorrect file pattern returned non-empty dataframe'
