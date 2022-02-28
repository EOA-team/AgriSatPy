'''
Metadata filtering utilities for Sentinel-2 data
'''

import pandas as pd
from typing import Optional


def identify_updated_scenes(
        metadata_df: pd.DataFrame,
        return_highest_baseline: Optional[bool] = True
    ) -> pd.DataFrame:
    """
    Returns those S2 entries in a pandas ``DataFrame`` retrieved from a query in
    AgriSatPy's metadata base that originate from the same orbit and data take
    but were processed by different PDGS Processing Baseline number (the 'Nxxxx'
    in the ``product_uri`` entry in the scene metadata or .SAFE name).

    :param metadata_df:
        dataframe from metadata base query in which to search for scenes with
        the same sensing date and data take but different baseline versions
    :param return_highest_baseline:
        if True (default) return those scenes with the highest baseline. Otherwise
        return the baseline most products belong to
    :return:
        ``DataFrame`` with those S2 scenes belonging to either the highest
        PDGS baseline or the most common baseline version
    """

    # get a copy of the input to work with
    metadata = metadata_df.copy()

    # check product uri and extract the processing baseline
    metadata['baseline'] = metadata.product_uri.apply(
        lambda x: int(x.split('_')[3][1:4])
    )

    # get either the highest baseline version or the baseline most datasets
    # belong to depending on the user input
    if return_highest_baseline:
        baseline_sel = metadata.baseline.unique().max()
    else:
        baseline_sel = metadata.baseline.mode()

    # return only those data-set belonging to the selected baseline version
    return metadata[metadata.baseline == baseline_sel]
