'''
Created on Jul 8, 2021

@author: graflu
'''

import os
import glob
import pandas as pd


def stack_dataframes(in_dir: str,
                     search_pattern: str,
                     start_date: int=None,
                     end_date: int=None,
                     **kwargs
                     ) -> pd.DataFrame:
    """
    stacks a list of pandas dataframes into a single big one
    to allow for calculating multitemporal statistics and more
    convenient handling of pixels and field polygons

    :param in_dir:
        directory in which to search for CSV files to be read into memory
    :param search_pattern:
        wild-card expression for searching for CSV files with pixel reflectance
        values (e.g., '*10m.csv')
    :param start_date:
        start date in the format YYYYMMDD to use for filtering CSV files. If None
        (Default), all files are stacked
    :param end_date:
        end date in the format YYYYMMDD to use for filtering CSV files. If None
        (Default), all files are stacked
    :param **kwargs:
        keyword arguments to pass to pandas.read_csv()
    """
    # get a list of all CSV files matching the search pattern
    csv_files = glob.glob(os.path.join(in_dir, search_pattern))

    # loop over files and read them into dataframes
    all_df = []
    for csv_file in csv_files:
        if start_date is not None and end_date is not None:
            date_file = int(os.path.basename(csv_file)[0:8])
            if date_file < start_date or date_file > end_date:
                continue
        tmp_df = pd.read_csv(csv_file, **kwargs)
        all_df.append(tmp_df)

    # concat the obtained list of dataframes into a single one and return
    return pd.concat(all_df)
