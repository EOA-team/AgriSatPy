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
from pathlib import Path
from sqlalchemy import desc
from sqlalchemy import and_
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agrisatpy.config import get_settings
from agrisatpy.metadata.sentinel2.database import S2_Raw_Metadata

Settings = get_settings()
engine = create_engine(Settings.DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()


def scene_selection(tile: str,
                    cloudcover_threshold: float,
                    date_start: date,
                    date_end: date,
                    out_dir: Path
                    ) -> None:
    """
    Function to query the metadata CSV file (generated using AgriSatpy) and extract
    a subset out of it fulfilling a range of criteria. It allows to filter the metadata
    by date range, cloud cover and Sentinel-2 tile.

    As a result, a new CSV file with those metadata entries fulfilling the criteria is
    returned plus a plot showing the cloud cover (extracted from the metadata xml file)
    per image acquisition date.

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

    # query metadata from database
    metadata = pd.read_sql(session.query(S2_Raw_Metadata).filter(
        and_(
            S2_Raw_Metadata.tile_id == tile,
            S2_Raw_Metadata.cloudy_pixel_percentage < cloudcover_threshold
        )).filter(
            and_(
                S2_Raw_Metadata.sensing_date <= date_end,
                S2_Raw_Metadata.sensing_date >= date_start
            )
        ).order_by(
            S2_Raw_Metadata.sensing_date.desc()
        ).statement,
        session.bind
    )

    # drop xml columns
    metadata.drop('mtd_tl_xml', axis=1, inplace=True)
    metadata.drop('mtd_msi_xml', axis=1, inplace=True)

    # calculate average cloud cover for the selected scenes
    cc_avg = metadata.cloudy_pixel_percentage.mean()

    # get timestamp of query execution
    query_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # write out metadata of the query as CSV
    metadata.to_csv(out_dir.joinpath(f'{query_time}_query.csv'), index=False)

    # Plot available scenes for query
    fig = plt.figure(figsize = (8, 6), dpi = 300)
    ax = fig.add_subplot(111)
    ax.plot(metadata['sensing_date'], metadata['cloudy_pixel_percentage'], 
            marker = 'o', markersize = 10)
    ax.set_xlabel('Sensing Date')
    ax.set_ylabel('Cloud cover [%]')
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
