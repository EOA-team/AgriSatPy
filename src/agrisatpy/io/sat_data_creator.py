'''
The Sat_Data_Creator class allows to create (satellite) image raster data
with geo-referenzation from numpy.arrays.
'''

from typing import Optional

from agrisatpy.io import SatDataHandler
from agrisatpy.utils.decorators import check_meta


class Sat_Data_Creator(SatDataHandler):
    """
    class for creating new SatDataHandler-like objects.

    :attribute is_bandstack:
        if False, allows bands to have different spatial resolutions and
        extends. True by default.
    """
    def __init__(
            self,
            is_bandstack: Optional[bool] = True,
            *args,
            **kwargs
        ):
        SatDataHandler.__init__(self, *args, **kwargs)
        self._from_bandstack = is_bandstack


    @check_meta
    def set_meta(
            self,
            meta: dict,
            band_name: Optional[str] = None
        ) -> None:
        """
        Adds image metadata to the current object. Image metadata is an essential
        pre-requisite for writing image data to raster files.

        IMPORTANT: Overwrites image metadata if already existing!

        :param meta:
            image metadata dict
        :param band_name:
            name of the band for which meta is added. If the current object
            is not a bandstack, specifying a band name is mandatory!
        """

        # check if meta is already populated
        if 'meta' not in self.data.keys():
            self.data['meta'] = {}

        # check if the data is band stack
        if self.is_bandstack:
            self.data['meta'] = meta
        else:
            if band_name is None:
                raise ValueError(
                    'Band name must be provided when not from bandstack'
                )
            self.data['meta'][band_name] = meta


    def set_bounds(
            self,
            bounds,
            band_name: Optional[str] = None
        ) -> None:
        """
        Adds image bounds to the current object. Image bounds are required for
        plotting.

        IMPORTANT: Overwrites image bounds if already existing!

        :param meta:
            image metadata dict
        :param band_name:
            name of the band for which meta is added. If the current object
            is not a bandstack, specifying a band name is mandatory!
        """

        # check if bounds is already populated
        if 'bounds' not in self.data.keys():
            self.data['bounds'] = {}
    
        if self.is_bandstack:
            self.data['bounds'] = bounds
        else:
            if band_name is None:
                raise ValueError(
                    'Band name must be provided when not from bandstack'
                )
            self.data['bounds'][band_name] = bounds


    def copy_geoinfo_from_reader(
            self,
            reader: SatDataHandler,
            band_name: Optional[str] = None
        ) -> None:
        """
        copies the meta and bounds geo information from another
        ``SatDataHandler`` object into the current object

        :param reader:
            reader object from which to copy the geo-info from
        :param band_name:
            name of the band to process (optional). Must be provided if
            ``is_bandstack=False``.
        """

        # copy from band-stack
        if self.is_bandstack:
            meta = reader.get_meta()
            band_name = reader.get_bandnames()[0]
            bounds = reader.get_band_bounds(
                band_name=band_name,
                return_as_polygon=False
            )
            self.add_meta(meta)
            self.add_bounds(bounds)
    
        # copy from non-band-stacks
        else:
            if band_name is None:
                raise ValueError(
                    'Band name must be provided when not from bandstack'
                )
            # meta
            try:
                meta = reader.get_meta(band_name)
            except Exception as e:
                raise Exception from e
            # bounds
            try:
                bounds = reader.get_band_bounds(
                    band_name=band_name,
                    return_as_polygon=False
                )
            except Exception as e:
                raise Exception from e

            self.add_meta(meta=meta, band_name=band_name)
            self.add_bounds(bounds=bounds, band_name=band_name)
