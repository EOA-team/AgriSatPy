'''
Functions to query Sentinel-2 specific metadata from the metadata DB.
'''

import pandas as pd

from datetime import date
from sqlalchemy import create_engine
from sqlalchemy import and_
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker

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
        bounding_box: str
    ) -> pd.DataFrame:
    """
    Queries the metadata DB by Sentinel-2 bounding box, time period and processing
    level.

    :param date_start:
        start date of the time period
    :param date_end:
        end date of the time period
    :param processing_level:
        Sentinel-2 processing level
    :param bounding_box_wkt:
        bounding box as extented well-known text in geographic coordinates
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
        S2_Raw_Metadata.sensing_date
    ).filter(
        S2_Raw_Metadata.geom == bounding_box
    ).filter(
        and_(
            S2_Raw_Metadata.sensing_date <= date_end,
            S2_Raw_Metadata.sensing_date >= date_start
        )
    ).filter(
        S2_Raw_Metadata.processing_level == processing_level_db
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
        tile: str
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
    :return:
        dataframe with references to found Sentinel-2 scenes
    """

    # translate processing level
    processing_level_db = ProcessingLevelsDB[processing_level.value]
    
    query_statement = session.query(
        S2_Raw_Metadata.product_uri,
        S2_Raw_Metadata.scene_id,
        S2_Raw_Metadata.storage_share,
        S2_Raw_Metadata.storage_device_ip_alias,
        S2_Raw_Metadata.storage_device_ip,
        S2_Raw_Metadata.sensing_date
    ).filter(
        S2_Raw_Metadata.tile_id == tile
    ).filter(
        and_(
            S2_Raw_Metadata.sensing_date <= date_end,
            S2_Raw_Metadata.sensing_date >= date_start
        )
    ).filter(
        S2_Raw_Metadata.processing_level == processing_level_db
    ).order_by(
        S2_Raw_Metadata.sensing_date.desc()
    ).statement

    try:
        return pd.read_sql(query_statement, session.bind)
    except Exception as e:
        raise DataNotFoundError(
            f'Could not find Sentinel-2 data by tile: {e}'
        )
