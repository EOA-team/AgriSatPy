'''
Created on Jul 8, 2021

@author: graflu
'''

import logging
from pathlib import Path
from os.path import join
from pydantic import BaseModel
from functools import lru_cache


class Settings(BaseModel):
    """
    The AgriSatPy setting class. Allows to modify default
    settings and behaviour of the package using a .env file
    or environmental variables
    """
    # sat archive definitions
    SUBDIR_PIXEL_CSVS: str = 'tables_w_pixelvalues'
    SUBDIR_RGB_PREVIEWS: str = 'rgb_previews'
    SUBDIR_SCL_FILES: str = 'scene_classification'
    
    # define date format
    DATE_FMT_INPUT: str = '%Y-%m-%d'
    DATE_FMT_FILES: str = '%Y%m%d'
    
    # define logger
    LOGGER_NAME: str = 'AgriSatPy'
    LOG_FORMAT: str = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    LOG_FILE: str = join(Path.home(), f'{LOGGER_NAME}.log')
    LOGGING_LEVEL: int = logging.INFO

    def get_logger(self):
        """
        returns a logger object with stream and file handler
        """
        logger: logging.Logger = logging.getLogger(self.LOGGER_NAME)
        logger.setLevel(self.LOGGING_LEVEL)
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
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    # env files are encoded utf-8, only
    class Config:
        env_file_encoding = 'utf-8'
        arbitrary_types_allowed = True


@lru_cache()
def get_settings():
    """
    loads package settings
    """
    return Settings()