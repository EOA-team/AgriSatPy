'''
Created on Aug 3, 2021

@author: Gregor Perich & Lukas Graf (D-USYS, ETHZ)
'''

import os
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scene_selection(metadata_file: str,
                    tile: str,
                    cloudcover_threshold: float,
                    date_start: date,
                    date_end: date,
                    out_dir: str
                    ) -> None:
    """
    Function to query the metadata CSV file (generated using AgriSatpy) and extract
    a subset out of it fulfilling a range of criteria. It allows to filter the metadata
    by date range, cloud cover and Sentinel-2 tile.

    As a result, a new CSV file with those metadata entries fulfilling the criteria is
    returned plus a plot showing the cloud cover (extracted from the metadata xml file)
    per image acquisition date.

    :param metadata_file:
        CSV file containing the full (i.e., extracted from the xml files) metadata of
        all scenes in a specific sat directory containing .SAFE files
    :param tile:
        Sentinel-2 tile for which the scene selection should be performed. For instance,
        'T32TMT'
    :param cloudcover_threshold:
        cloud cover threshold in percent (0-100) to filter scenes. All scenes with a
        cloud cover lower than the threshold will be returned
    :param date_start:
        start date for filtering the data
    :param end_date:
        end data for filtering the data
    :param out_dir:
        directory where to store the subset metadata CSV file and the cloud cover plot.
    """
    # read metadata file
    try:
        metadata_full = pd.read_csv(metadata_file)
    except Exception as e:
        raise Exception(f'Could not read metadata file: {e}')

    # filter metadata by tile
    metadata = metadata_full[metadata_full.TILE == tile].copy()

    # filter and sort by date
    metadata.SENSING_DATE = pd.to_datetime(metadata.SENSING_DATE)
    metadata = metadata[pd.to_datetime(metadata.SENSING_DATE).between(
        date_start, date_end, inclusive=True)]
    metadata = metadata.sort_values(by='SENSING_DATE')

    # filter by cloud cover
    # '''
    # cloud cover as reported in the S2 L2A metadata consists of the sum of: 
    #     HIGH_PROBA_CLOUDS_PERCENTAGE + MEDIUM_PROBA_CLOUDS_PERCENTAGE +
    #     THIN_CIRRUS_PERCENTAGE (SCL classes 3, 8, 9)
    # '''
    metadata = metadata[metadata.CLOUDY_PIXEL_PERCENTAGE <= cloudcover_threshold]

    # calculate average cloud cover for the selected scenes
    cc_avg = metadata.CLOUDY_PIXEL_PERCENTAGE.mean()

    # get timestamp of query execution
    query_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # write out metadata of the query as CSV
    metadata.to_csv(os.path.join(out_dir, f'{query_time}_query.csv'), index=False)

    # Plot available scenes for query
    fig = plt.figure(figsize = (8, 6), dpi = 300)
    ax = fig.add_subplot(111)
    ax.plot(metadata['SENSING_DATE'], metadata['CLOUDY_PIXEL_PERCENTAGE'], 
            marker = 'o', markersize = 10)
    ax.set_xlabel("Sensing Date")
    ax.set_ylabel("Cloud cover [%]")
    ax.set_ylim(0., 100.)
    ax.set_title(f'Tile {tile} - No. of scenes: {metadata.shape[0]}'
                 + '\n' + f'Average cloud cover: {np.round(cc_avg, 2)}%')
    plt.savefig(
        os.path.join(
            out_dir,
            f'{query_time}_query_CCplot.png'
        ), 
        bbox_inches="tight"
    )
    plt.close()
