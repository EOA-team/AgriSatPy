'''
Created on Jul 19, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

from functools import wraps

from agrisatpy.config import get_settings


Settings = get_settings()


class ProcessingLevelNotFound(Exception):
    pass

def check_processing_level(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            processing_level = args[1]
        if kwargs != {}:
            processing_level = kwargs.get('processing_level', str)

        if not processing_level in Settings.PROCESSING_LEVELS:
            raise ProcessingLevelNotFound(
                f'{processing_level} is not part of {Settings.PROCESSING_LEVELS}')
        return f(*args, **kwargs)

    return wrapper


def string_to_path(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if isinstance('', str):
            pass
        # TODO
        
        