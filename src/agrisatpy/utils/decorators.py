'''
Function and method decorators used to validate passed arguments.
'''

from functools import wraps
from rasterio.coords import BoundingBox

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
        if len(args) == 0 and len(kwargs) == 0:
            return f(self, *args, **kwargs)
        
        if len(args) > 0:
            # band name(s) are always provided as first argument
            band_names = args[0]
        if kwargs != {} and band_names is None:
            # check for band_name and band_names key word argument
            band_names = kwargs.get('band_name', band_names)
            if band_names is None:
                band_names = kwargs.get('band_selection', band_names)

        # check if band aliases is enabled
        if self.has_band_aliases:
            # check if passed band names are actual band names or their alias
            if isinstance(band_names, str):
                band_name = band_names
                if band_name not in self.band_names:
                    # passed band name is alias
                    if band_name in self.band_aliases:
                        band_idx = self.band_aliases.index(band_name)
                        band_name = self.band_names[band_idx]
                        if len(args) > 0:
                            arg_list = list(args)
                            arg_list[0] = band_name
                            args = tuple(arg_list)
                        if kwargs != {} and 'band_name' in kwargs.keys():
                            kwargs.update({'band_name': band_name})
                    else:
                        raise BandNotFoundError(
                            f'{band_names} not found in collection'
                        )
            elif isinstance(band_names, list):
                # check if passed band names are aliases
                if set(band_names).issubset(self.band_names):
                    new_band_names = band_names
                else:
                    new_band_names = []
                    for band_name in band_names:
                        try:
                            new_band_names.append(self[band_name].alias)
                        # band name must be in band names if not an alias
                        except Exception:
                            raise BandNotFoundError(
                                f'{band_name} not found in collection'
                            )
                if len(args) > 0:
                    arg_list = list(args)
                    arg_list[0] = new_band_names
                    args = tuple(arg_list)
                if kwargs != {} and 'band_selection' in kwargs.keys():
                    kwargs.update({'band_selection': new_band_names})

        # if no band aliasing is enabled the passed name must be in band names
        else:
            if isinstance(band_names, str):
                if not band_names in self.band_names:
                    raise BandNotFoundError(
                        f'{band_names} not found in collection'
                    )
            elif isinstance(band_names, list):
                if not set(band_names).issubset(self.band_names):
                    raise BandNotFoundError(
                        f'{band_names} not found in collection'
                    )

        return f(self, *args, **kwargs)

    return wrapper


def check_metadata(f):
    """validates if passed image metadata items are valid"""
    @wraps(f)
    def wrapper(self, *args, **kwargs):

        meta_key, meta_values = None, None

        if len(args) > 0:
            meta_key = args[0]
            meta_values = args[1]
        if kwargs != {}:
            if meta_key is None:
                meta_key = kwargs.get('metadata_key', meta_key)
            if meta_values is None:
                meta_values = kwargs.get('metadata_values', meta_values)

        # check different entries
        # image metadata
        if meta_key == 'meta':
            meta_keys = ['driver', 'dtype', 'nodata', 'width', 'height', 'count', 'crs', 'transform']
            if set(list(meta_values.keys())) !=  set(meta_keys):
                raise Exception('The passed meta-dict is invalid')
        # bounds
        elif meta_key == 'bounds':
            if not type(meta_values) == BoundingBox:
                raise Exception('The passed bounds are not valid.')

        return f(self, *args, **kwargs)

    return wrapper
