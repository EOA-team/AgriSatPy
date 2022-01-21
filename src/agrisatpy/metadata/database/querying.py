'''
Functions to query remote sensing plaform independent metadata from the metadata DB.
'''

import geopandas as gpd

from sqlalchemy import create_engine

from agrisatpy.metadata.database import Regions
from agrisatpy.utils.exceptions import RegionNotFoundError
from agrisatpy.config import get_settings
from sqlalchemy.orm.session import sessionmaker


Settings = get_settings()
logger = Settings.logger

DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()


def get_region(
        region: str
    ) -> gpd.GeoDataFrame:
    """
    Queries the metadata DB for a specific region and its geographic
    extent.

    :param region:
        unique region identifier

    :return:
        geodataframe with the geometry of the queried region
    """

    query_statement = session.query(
        Regions.geom,
        Regions.region_uid
    ).filter(
        Regions.region_uid == region
    ).statement

    try:
        return gpd.read_postgis(query_statement, session.bind)

    except Exception as e:
        raise RegionNotFoundError(f'{region} not found: {e}')
