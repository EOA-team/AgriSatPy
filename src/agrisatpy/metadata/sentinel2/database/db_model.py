'''
Created on Jul 12, 2021

@author: Lukas Graf (D-USYS; ETHZ)
'''

from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Float, Time, Date, Integer, Text, ARRAY
from geoalchemy2 import Geometry

from agrisatpy.config import get_settings


Settings = get_settings()
logger = Settings.logger

metadata = MetaData(schema=Settings.DEFAULT_SCHEMA)
Base = declarative_base(metadata=metadata)

engine = create_engine(Settings.DB_URL, echo=Settings.ECHO_DB)

Base = declarative_base()

class S2_Raw_Metadata(Base):

    __tablename__ = 'sentinel2_raw_metadata'

    # scene and tile id
    scene_id = Column(String, nullable=False, primary_key=True)
    product_uri = Column(String, nullable=False, primary_key=True)
    tile_id = Column(String, nullable=False)
    l1c_tile_id = Column(String)
    tile = Column(String, nullable=False)

    # processing level, orbit and spacecraft
    processing_level = Column(String, nullable=False)
    sensing_orbit_number = Column(Integer, nullable=False)
    spacecraft_name = Column(String, nullable=False)
    sensing_orbit_direction = Column(String, nullable=False)

    # temporal information
    sensing_time = Column(Time, nullable=False)
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
    storage_device_ip = Column(String, nullable=False)
    storage_share = Column(String, nullable=False)
    path_type = Column(String, nullable=False, comment='type of the path (e.g., POSIX-Path)')


class S2_Processed_Metadata(Base):
    
    __tablename__ = 'sentinel2_processed_metadata'

    # scene and tile id
    scene_id = Column(String, nullable=False, primary_key=True)
    product_uri = Column(String, nullable=False, primary_key=True)

    # spatial resolution and used resampling method
    spatial_resolution = Column(Float, nullable=False)
    interpolation_method = Column(String, nullable=False)

    # spectral bands contained
    spectral_bands = Column(ARRAY(String), nullable=False)


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
