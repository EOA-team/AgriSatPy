'''
Created on Jul 8, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

import os
import pandas as pd
from typing import Tuple
from datetime import datetime
from pathlib import Path

from agrisatpy.utils import stack_dataframes
from agrisatpy.config import get_settings
from agrisatpy.utils.decorators import check_processing_level


Settings = get_settings()
logger = Settings.logger

@check_processing_level
def search_data(in_dir: Path,
                processing_level: str,
                start_date: str,
                end_date: str,
                search_pattern: str,
                search_pattern_scl: str,
                tile: str
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    given a location where processed Sentinel-2 bandstacks are stored, a date range
    (between YYYY-MM-DD and YYYY-MM-DD) and a search pattern to identify CSV files
    with extracted reflectance values, this function reads data fulfilling the criteria
    into a pandas DF - also over multiple years - per Sentinel-2 tile.
    In addition, returns the SCL statistic per (field) polygon and selected date.

    If no data was found, returns a tuple consisting of two None-Type objects.

    To make it work, the directory must be structured as follows (<> indicates user-inputs,
    while SUBDIR_PIXEL_CSVS is a global configuration of the package):
    
    /<in_dir>
        /<processing_level>
            /<year>
                /<tile>
                    /SUBDIR_PIXEL_CSVS

    For Sentinel-2 (processing level L2A) this would like:

    /in_dir
        /L2A
            /2020
                /T31TGM
                    /tables_w_pixelvalues
                /T32TMT
                    /tables_w_pixelvalues
                ...
        /L2A
            /2019
                /T31TGM
                    /tables_w_pixelvalues
                /T32TMT
                    /tables_w_pixelvalues
                ...

    :param in_dir:
        directory where extracted reflectance values are stored in the same structure
        as specified above
    :param processing_level:
        processing level of the satellite data. Must be one of 'L1C', 'L2A'
    :param start_date:
        start of the date range for plotting data
    :param end_date:
        end of the date range for plotting data
    :param search_pattern:
        search pattern identifying CSV files with extracted reflectance values
        (e.g., '*_10m.csv). Note that the use of a wildcard character is necessary
        to make the file search work.
    :param search_pattern_scl:
         search pattern identifying CSV files with extracted SCL statistics per
        (field) polygon and sensing date. Note that the use of a wildcard character is necessary
        to make the file search work.
    :param tile:
        Sentinel-2 tile to select (e.g., 'T32TMT')
    """
    # how many and which years need to be processed?
    date_fmt = Settings.DATE_FMT_INPUT
    start_date = datetime.strptime(start_date, date_fmt)
    end_date = datetime.strptime(end_date, date_fmt)
    start_year = start_date.year
    end_year = end_date.year
    years = [x for x in range(start_year, end_year+1)]

    # convert start_date and end_date to string expression to match the file naming
    # convention of the extracted pixel data
    date_fmt_fnames = Settings.DATE_FMT_FILES
    start_date_fnames = int(start_date.strftime(date_fmt_fnames))
    end_date_fnames = int(end_date.strftime(date_fmt_fnames))

    # loop over years and extract the data
    refls = []
    scls = []
    for year in years:

        logger.info(f'Searching for data in {year}')

        expr = processing_level + os.sep + str(year) + os.sep + tile
        in_dir_year = Path(in_dir.joinpath(
                        os.path.join(
                            expr,
                            Settings.SUBDIR_PIXEL_CSVS
                        )
        ))

        if not os.path.isdir(in_dir_year):
            logger.warning(f'No such directory: {in_dir_year}')
            continue

        # extract reflectance and SCL data for the year and tile
        # check specified date range
        try:
            df_refl = stack_dataframes(in_dir=in_dir_year,
                                       search_pattern=search_pattern,
                                       start_date=start_date_fnames,
                                       end_date=end_date_fnames)
        except Exception as e:
            logger.error(f'Couldnot stack dataframes: {e}')
            return (None, None)

        try:
            df_scl = stack_dataframes(in_dir=in_dir_year,
                                      search_pattern=search_pattern_scl,
                                      start_date=start_date_fnames,
                                      end_date=end_date_fnames)
        except Exception as e:
            logger.error(f'Couldnot stack dataframes: {e}')
            return (None, None)

        # append dataframes to list in case multiple years are used
        refls.append(df_refl)
        scls.append(df_scl)

    # check if any data was found. Return always None whenever one of the
    # two file lists contains zero elements
    if len(refls) == 0 or len(scls) == 0:
        logger.info('No date found')
        return (None, None)
    
    refl = pd.concat(refls)
    scl = pd.concat(scls)
    return (refl, scl)
