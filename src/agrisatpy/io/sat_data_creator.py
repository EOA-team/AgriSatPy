'''
The Sat_Data_Creator class allows to create (satellite) image raster data
with geo-referenzation from numpy.arrays.
'''

from typing import Optional

from agrisatpy.io import SatDataHandler



class Sat_Data_Creator(SatDataHandler):
    """
    class for creating new SatDataHandler-like objects.

    :attribute is_bandstack:
        if False, allows bands to have different spatial resolutions and
        extends. True by default.
    """
    pass