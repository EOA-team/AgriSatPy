'''
Created on Jul 8, 2021

@author: graflu
'''

from typing import List
import logging
from pathlib import Path
from os.path import join
from pydantic import BaseSettings
from functools import lru_cache
from datetime import datetime


class Settings(BaseSettings):
    """
    The AgriSatPy setting class. Allows to modify default
    settings and behavior of the package using a .env file
    or environmental variables
    """
    # sat archive definitions
    SUBDIR_PIXEL_CSVS: str = 'tables_w_pixelvalues'
    SUBDIR_RGB_PREVIEWS: str = 'rgb_previews'
    SUBDIR_SCL_FILES: str = 'scene_classification'

    RESAMPLED_METADATA_FILE: str = 'metadata.csv'

    PROCESSING_LEVELS: List[str] = ['L1C', 'L2A']
    
    # define date format
    DATE_FMT_INPUT: str = '%Y-%m-%d'
    DATE_FMT_FILES: str = '%Y%m%d'

    # define DHUS username and password
    DHUS_USER: str = ''
    DHUS_PASSWORD: str = ''

    # define CREODIAS username and password
    CREODIAS_USER: str = ''
    CREODIAS_PASSWORD: str = ''

    # metadata base connection details
    DB_USER: str = 'postgres'
    DB_PW: str = '12345'
    DB_HOST: str = 'localhost'
    DB_PORT: str = '5432'
    DB_NAME: str = 'metadata_db'
    DB_URL: str = f'postgresql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    
    DEFAULT_SCHEMA: str = 'cs_sat_s1'
    ECHO_DB: bool = False
    
    # define logger
    CURRENT_TIME: str = datetime.now().strftime('%Y%m%d-%H%M%S')
    LOGGER_NAME: str = 'AgriSatPy'
    LOG_FORMAT: str = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    LOG_FILE: str = join(Path.home(), f'{CURRENT_TIME}_{LOGGER_NAME}.log')
    LOGGING_LEVEL: int = logging.INFO

    # processing checks
    PROCESSING_CHECK_FILE_NO_BF: str = f'successful_scenes_noblackfill.txt'
    PROCESSING_CHECK_FILE_BF: str = f'successful_scenes_blackfill.txt'
    

    logger: logging.Logger = logging.getLogger(LOGGER_NAME)

    def get_logger(self):
        """
        returns a logger object with stream and file handler
        """
        self.logger.setLevel(self.LOGGING_LEVEL)
        # create file handler which logs even debug messages
        fh: logging.FileHandler = logging.FileHandler(self.LOG_FILE)
        fh.setLevel(self.LOGGING_LEVEL)
        # create console handler with a higher log level
        ch: logging.StreamHandler = logging.StreamHandler()
        ch.setLevel(self.LOGGING_LEVEL)
        # create formatter and add it to the handlers
        formatter: logging.Formatter = logging.Formatter(self.LOG_FORMAT)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    # env files are encoded utf-8, only
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        arbitrary_types_allowed = True


@lru_cache()
def get_settings():
    """
    loads package settings
    """
    s = Settings()
    s.get_logger()
    return s
