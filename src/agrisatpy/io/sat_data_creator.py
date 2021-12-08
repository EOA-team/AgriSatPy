'''
The Sat_Data_Creator class allows to create (satellite) image raster data
with geo-referenzation from numpy.arrays.
'''

from typing import Optional
from datetime import datetime

from agrisatpy.io import Sat_Data_Reader



class Sat_Data_Creator(Sat_Data_Reader):
    """
    class for creating new Sat_Data_Reader-like objects from numpy
    arrays. The class can be used, e.g., to create time series stacks.

    :attribute is_bandstack:
        if False, allows bands to have different spatial resolutions and
        extends. True by default.
    :attribute is_timeseries:
        if True, assumes that the single bands represent a single variable
        (e.g., spectral band) over time (i.e, from different image acquisition dates).
        If is_timeseries the data must have the same spatial resolution and extent
        (`ìs_bandstack=True``)
    """
    def __init__(
            self,
            is_bandstack: Optional[bool] = True,
            is_timeseries: Optional[bool] = False,
            *args,
            **kwargs
        ):
        Sat_Data_Reader.__init__(self, *args, **kwargs)
        self._from_bandstack = is_bandstack
        self._is_timeeries = is_timeseries


    def add_meta(self, meta):
        pass

    def copy_meta_from_reader(self, reader, band_selection):
        pass

    def add_observation(self, timestamp: datetime, band_name, data):
        pass

    def get_observation(self, band_names):
    