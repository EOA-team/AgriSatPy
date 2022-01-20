'''
Functions to ingest new, remote sensing platform-independent
data into the metadata DB.
'''

import geopandas as gpd

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agrisatpy.config import get_settings
from agrisatpy.metadata.database.db_model import Regions


Settings = get_settings()
logger = Settings.logger

DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()


def add_region(
        region_identifier: str,
        region_file: Path
    ) -> None:
    """
    Adds a new region to the database. Regions are geograhic extents
    (aka bounding boxes) used to organize archive queries (e.g., from
    CREODIAS) as these queries need some kind of geographic extent to
    be efficient.

    :param region_identifier:
        unique region identifier (e.g., 'CH' for Switzerland)
    :param region_file:
        shapefile or similar vector format defining extent of the region.
        Will be reprojected to geographic coordinates (WGS84) if it has
        a different projection.
    """

    # read shapefile defining the bounds of your region of interest
    region_data = gpd.read_file(region_file)
    # project to geographic coordinates (required by data model)
    region_data.to_crs(4326, inplace=True)
    # use the first feature (all others are ignored)
    bounding_box = region_data.geometry.iloc[0]
    # the insert requires extended Well Know Text (eWKT)
    bounding_box_ewkt = f'SRID=4326;{bounding_box.wkt}'

    insert_dict = {
        'region_uid': region_identifier,
        'geom': bounding_box_ewkt
    }

    try:
        session.add(Regions(**insert_dict))
        session.commit()
    except Exception as e:
        logger.error(f'Insert of region "{region_identifier} failed: {e}')
        session.rollback()
