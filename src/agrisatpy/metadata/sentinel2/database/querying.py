'''
Functions to query Sentinel-2 specific metadata from the metadata DB.

Query criteria include

- the processing level (either L1C or L2A for ESA derived Sentinel-2 data)
- the acquisition period (between a start and an end dat)
- the tile (e.g., "T32TLT") or a bounding box (provided as extended well-known-text)
- the scene-wide cloud coverage (derived from the scene metadata); this is optional.
'''

import pandas as pd

from datetime import date
from sqlalchemy import create_engine
from sqlalchemy import and_
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker
from typing import Optional
from typing import Union

from agrisatpy.utils.constants import ProcessingLevels
from agrisatpy.utils.constants.sentinel2 import ProcessingLevelsDB
from agrisatpy.metadata.database import S2_Raw_Metadata
from agrisatpy.config import get_settings
from agrisatpy.utils.exceptions import DataNotFoundError

Settings = get_settings()
logger = Settings.logger

DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()


def find_raw_data_by_bbox(
        date_start: date,
        date_end: date,
        processing_level: ProcessingLevels,
        bounding_box: str,
        cloud_cover_threshold: Optional[Union[int,float]] = 100
    ) -> pd.DataFrame:
    """
    Queries the metadata DB by Sentinel-2 bounding box, time period and processing
    level (and cloud cover)

    :param date_start:
        start date of the time period
    :param date_end:
        end date of the time period
    :param processing_level:
        Sentinel-2 processing level
    :param bounding_box_wkt:
        bounding box as extented well-known text in geographic coordinates
    :param cloud_cover_threshold:
        optional cloud cover threshold to filter datasets by scene cloud coverage.
        Must be provided as number between 0 and 100%.
    :return:
        dataframe with references to found Sentinel-2 scenes
    """

    # translate processing level
    processing_level_db = ProcessingLevelsDB[processing_level.value]

    # TODO: test
    query_statement = session.query(
        S2_Raw_Metadata.product_uri,
        S2_Raw_Metadata.scene_id,
        S2_Raw_Metadata.storage_share,
        S2_Raw_Metadata.storage_device_ip_alias,
        S2_Raw_Metadata.storage_device_ip,
        S2_Raw_Metadata.sensing_date,
        S2_Raw_Metadata.cloudy_pixel_percentage
    ).filter(
        S2_Raw_Metadata.geom == bounding_box
    ).filter(
        and_(
            S2_Raw_Metadata.sensing_date <= date_end,
            S2_Raw_Metadata.sensing_date >= date_start
        )
    ).filter(
        S2_Raw_Metadata.processing_level == processing_level_db
    ).filter(
        S2_Raw_Metadata.cloudy_pixel_percentage <= cloud_cover_threshold
    ).order_by(
        S2_Raw_Metadata.sensing_date.desc()
    ).statement

    try:
        return pd.read_sql(query_statement, session.bind)
    except Exception as e:
        raise DataNotFoundError(
            f'Could not find Sentinel-2 data by bounding box: {e}'
        )


def find_raw_data_by_tile(
        date_start: date,
        date_end: date,
        processing_level: ProcessingLevels,
        tile: str,
        cloud_cover_threshold: Optional[Union[int,float]] = 100
    ) -> pd.DataFrame:
    """
    Queries the metadata DB by Sentinel-2 tile, time period and processing
    level.

    :param date_start:
        start date of the time period
    :param date_end:
        end date of the time period
    :param processing_level:
        Sentinel-2 processing level
    :param tile:
        Sentinel-2 tile
    :param cloud_cover_threshold:
        optional cloud cover threshold to filter datasets by scene cloud coverage.
        Must be provided as number between 0 and 100%.
    :return:
        dataframe with references to found Sentinel-2 scenes
    """

    # translate processing level
    processing_level_db = ProcessingLevelsDB[processing_level.name]
    
    query_statement = session.query(
        S2_Raw_Metadata.product_uri,
        S2_Raw_Metadata.scene_id,
        S2_Raw_Metadata.storage_share,
        S2_Raw_Metadata.storage_device_ip_alias,
        S2_Raw_Metadata.storage_device_ip,
        S2_Raw_Metadata.sensing_date,
        S2_Raw_Metadata.cloudy_pixel_percentage
    ).filter(
        S2_Raw_Metadata.tile_id == tile
    ).filter(
        and_(
            S2_Raw_Metadata.sensing_date <= date_end,
            S2_Raw_Metadata.sensing_date >= date_start
        )
    ).filter(
        S2_Raw_Metadata.processing_level == processing_level_db
    ).filter(
        S2_Raw_Metadata.cloudy_pixel_percentage <= cloud_cover_threshold
    ).order_by(
        S2_Raw_Metadata.sensing_date.desc()
    ).statement

    try:
        return pd.read_sql(query_statement, session.bind)
    except Exception as e:
        raise DataNotFoundError(
            f'Could not find Sentinel-2 data by tile: {e}'
        )
