'''
Implementation of a data model for AgriSatPy's metadata DB including
platform specific tables for storing e.g., Sentinel-2 metadata
'''


from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Float, Date, Integer, Text, Boolean, TIMESTAMP
from geoalchemy2 import Geometry

from agrisatpy.config import get_settings


Settings = get_settings()
logger = Settings.logger

metadata = MetaData(schema=Settings.DEFAULT_SCHEMA)
Base = declarative_base(metadata=metadata)

DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)


class Regions(Base):

    __tablename__ = 'sentinel2_regions'

    region_uid = Column(String, nullable=False, primary_key=True)
    geom = Column(Geometry(geometry_type='POLYGON',srid=4326), nullable=False)


class S2_Raw_Metadata(Base):

    __tablename__ = 'sentinel2_raw_metadata'

    # scene and tile id
    scene_id = Column(String, nullable=False, primary_key=True)
    product_uri = Column(String, nullable=False, primary_key=True)
    tile_id = Column(String, nullable=False)
    l1c_tile_id = Column(String)

    # processing baseline
    pdgs_baseline = Column(String, nullable=False)

    # Datatake Information
    datatakeidentifier = Column(String, nullable=False)

    # processing level, orbit and spacecraft
    processing_level = Column(String, nullable=False)
    sensing_orbit_number = Column(Integer, nullable=False)
    spacecraft_name = Column(String, nullable=False)
    sensing_orbit_direction = Column(String, nullable=False)

    # temporal information
    sensing_time = Column(TIMESTAMP, nullable=False)
    sensing_date = Column(Date, nullable=False)

    # geometry and geolocalisation
    nrows_10m = Column(Integer, nullable=False)
    ncols_10m = Column(Integer, nullable=False)
    nrows_20m = Column(Integer, nullable=False)
    ncols_20m = Column(Integer, nullable=False)
    nrows_60m = Column(Integer, nullable=False)
    ncols_60m = Column(Integer, nullable=False)

    epsg = Column(Integer, nullable=False)
    
    ulx = Column(Float, nullable=False, comment='Upper left corner x coordinate')
    uly = Column(Float, nullable=False, comment='Upper left corner y coordinate')

    geom = Column(Geometry(geometry_type='POLYGON',srid=4326), nullable=False)

    # viewing and illumination angles
    sun_zenith_angle = Column(Float, nullable=False)
    sun_azimuth_angle = Column(Float, nullable=False)
    sensor_zenith_angle = Column(Float, nullable=False)
    sensor_azimuth_angle = Column(Float, nullable=False)

    # solar irradiance per spectral band (W/m2sr)
    solar_irradiance_b01 = Column(Float, nullable=False)
    solar_irradiance_b02 = Column(Float, nullable=False)
    solar_irradiance_b03 = Column(Float, nullable=False)
    solar_irradiance_b04 = Column(Float, nullable=False)
    solar_irradiance_b05 = Column(Float, nullable=False)
    solar_irradiance_b06 = Column(Float, nullable=False)
    solar_irradiance_b07 = Column(Float, nullable=False)
    solar_irradiance_b08 = Column(Float, nullable=False)
    solar_irradiance_b8a = Column(Float, nullable=False)
    solar_irradiance_b09 = Column(Float, nullable=False)
    solar_irradiance_b10 = Column(Float, nullable=False)
    solar_irradiance_b11 = Column(Float, nullable=False)
    solar_irradiance_b12 = Column(Float, nullable=False)

    # reflectance conversion factor to account for variations in the sun-earth distance
    reflectance_conversion = Column(Float)

    # cloudy pixel percentage
    cloudy_pixel_percentage = Column(Float, nullable=False)
    degraded_msi_data_percentage = Column(Float, nullable=False)

    # L2A specific data; therefore nullable
    nodata_pixel_percentage = Column(Float)
    dark_features_percentage = Column(Float)
    cloud_shadow_percentage = Column(Float)
    vegetation_percentage = Column(Float)
    not_vegetated_percentage = Column(Float)
    water_percentage = Column(Float)
    unclassified_percentage = Column(Float)
    medium_proba_clouds_percentage = Column(Float)
    high_proba_clouds_percentage = Column(Float)
    thin_cirrus_percentage = Column(Float)
    snow_ice_percentage = Column(Float)

    # raw representation of metadata XML since not all data was parsed
    mtd_tl_xml = Column(Text, nullable=False)
    mtd_msi_xml = Column(Text, nullable=False)

    # storage location
    storage_device_ip = Column(String)
    storage_device_ip_alias = Column(String) # might be necessary
    storage_share = Column(String, nullable=False)
    path_type = Column(String, nullable=False, comment='type of the path (e.g., POSIX-Path)')


class S2_Processed_Metadata(Base):
    
    __tablename__ = 'sentinel2_processed_metadata'

    # scene and tile id
    scene_id = Column(String, nullable=False, primary_key=True)
    product_uri = Column(String, nullable=False, primary_key=True)

    # spatial resolution and used resampling method
    spatial_resolution = Column(Float, nullable=False)
    resampling_method = Column(String, nullable=False)

    # was the scene merged to because of blackfill
    scene_was_merged = Column(Boolean, nullable=False, default=False)

    # storage address, filenames
    storage_device_ip = Column(String, nullable=False)
    storage_device_ip_alias = Column(String, nullable=False) # Linux
    storage_share = Column(String, nullable=False)
    bandstack = Column(String, nullable=False)
    scl = Column(String)
    preview = Column(String, nullable=False)
    path_type = Column(String, nullable=False, comment='type of the path (e.g., POSIX-Path)')

    __table_args__ = (
        ForeignKeyConstraint(['scene_id', 'product_uri'],
                             ['sentinel2_raw_metadata.scene_id', 'sentinel2_raw_metadata.product_uri'],
                             onupdate="CASCADE",
                             ondelete="CASCADE"
        ),
    )


def create_tables() -> None:
    """
    creates all Sentinel-2 related tables in the current
    <DEFAULT_SCHEMA>.
    """
    try:
        Base.metadata.create_all(bind=engine)
        for table in Base.metadata.tables.keys():
            logger.info(f'Created table {table}')
    except Exception as e:
        raise Exception(f'Could not create table: {e}')


if __name__ == '__main__':

    create_tables()
