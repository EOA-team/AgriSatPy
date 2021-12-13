'''
Created on Jul 19, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

from functools import wraps

from agrisatpy.config import get_settings
from agrisatpy.utils.exceptions import UnknownProcessingLevel
from agrisatpy.utils.exceptions import BandNotFoundError


Settings = get_settings()


def check_processing_level(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        processing_level = ''
        if len(args) > 0:
            processing_level = args[1]
        if kwargs != {}:
            processing_level = kwargs.get('processing_level', processing_level)

        if not processing_level in Settings.PROCESSING_LEVELS:
            raise UnknownProcessingLevel(
                f'{processing_level} is not part of {Settings.PROCESSING_LEVELS}')
        return f(*args, **kwargs)

    return wrapper


def check_band_names(f):
    """checks if passed band name(s) are available"""
    @wraps(f)
    def wrapper(self, *args, **kwargs):

        band_names = None
        # RGB and False-Color are white-listed
        white_list = ['RGB', 'False-Color']

        if len(args) > 0:
            # band name(s) are always provided as first argument
            band_names = args[0]
        if kwargs != {}:
            # check for band_name and band_names key word argument
            band_names = kwargs.get('band_name', band_names)
            if band_names is None:
                band_names = kwargs.get('band_names', band_names)

        # check if band aliases is enabled
        if self._has_bandaliases:
            # check if passed band names are actual band names or their alias
            if isinstance(band_names, str):
                band_name = band_names
                # passed band name is alias
                if band_name in self._band_aliases.values():
                    band_name = [k for k, v in self._band_aliases.items() if v == band_name][0]
                    if len(args) > 0:
                        arg_list = list(args)
                        arg_list[0] = band_name
                        args = tuple(arg_list)
                    if kwargs != {} and 'band_name' in kwargs.keys():
                        kwargs.update({'band_name': band_name})
            elif isinstance(band_names, list):
                # check if passed band names are aliases
                new_band_names = []
                for band_name in band_names:
                    if band_name in self._band_aliases.values():
                        band_name = [k for k, v in self._band_aliases.items() if v == band_name][0]
                    # band name must be in band names if not an alias
                    else:
                        if band_name not in self.get_bandnames() and band_name not in white_list:
                            raise BandNotFoundError(f'{band_name} not found in data dict')
                    new_band_names.append(band_name)
                if len(args) > 0:
                    arg_list = list(args)
                    arg_list[0] = new_band_names
                    args = tuple(arg_list)
                if kwargs != {} and 'band_names' in kwargs.keys():
                    kwargs.update({'band_names': new_band_names})

        # if no band aliasing is enabled the passed name must be in band names
        else:
            if isinstance(band_names, str):
                if not band_names in self.get_bandnames() and band_names not in white_list:
                    raise BandNotFoundError(f'{band_names} not found in data dict')

        return f(self, *args, **kwargs)

    return wrapper


