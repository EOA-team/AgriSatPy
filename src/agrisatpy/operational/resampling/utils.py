'''
Some auxiliary functions for spatial resampling of raster data
'''

import pandas as pd


def identify_split_scenes(
        metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Returns entries in a pandas ``DataFrame`` retrieved from a query in AgriSatPy's
    metadata base that have the same sensing date. This could indicate, e.g.,
    that scenes have been split because of data take changes which sometimes cause
    Sentinel-2 scenes to be split into two separate .SAFE archives, each of them
    with a large amount of blackfill.

    :param metadata_df:
        dataframe from metadata base query in which to search for scenes with
        the same sensing_date
    :return:
        scenes with the same sensing date (might also be empty)
    """

    return metadata_df[metadata_df.sensing_date.duplicated(keep=False)]
