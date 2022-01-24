'''
Defines some static attributes of Sentinel-2 MSI.
'''

from enum import Enum
from agrisatpy.utils.constants import ProcessingLevels


# available processing levels
class ProcessingLevels(Enum):
    L1C = 'LEVEL1C'
    L2A = 'LEVEL2A'

# Sentinel-2 processing levels as defined in the metadatabase
ProcessingLevelsDB = {
    'L1C' : 'Level-1C',
    'L2A' : 'Level-2A'
}

# native spatial resolution of the S2 bands per processing level
band_resolution = {
    ProcessingLevels.L1C: {
        'B01': 60,
        'B02': 10,
        'B03': 10,
        'B04': 10,
        'B05': 20,
        'B06': 20,
        'B07': 20,
        'B08': 10,
        'B8A': 20,
        'B09': 60,
        'B10': 60,
        'B11': 20,
        'B12': 20
    },
    ProcessingLevels.L2A: {
        'B01': 60,
        'B02': 10,
        'B03': 10,
        'B04': 10,
        'B05': 20,
        'B06': 20,
        'B07': 20,
        'B08': 10,
        'B8A': 20,
        'B09': 60,
        'B10': 60,
        'B11': 20,
        'B12': 20,
        'SCL': 20
    }    
}

s2_band_mapping = {
        'B01': 'ultra_blue',
        'B02': 'blue',
        'B03': 'green',
        'B04': 'red',
        'B05': 'red_edge_1',
        'B06': 'red_edge_2',
        'B07': 'red_edge_3',
        'B08': 'nir_1',
        'B8A': 'nir_2',
        'B09': 'nir_3',
        'B10': 'swir_0',
        'B11': 'swir_1',
        'B12': 'swir_2',
        'SCL': 'scl'
}

# S2 data is stored as uint16
s2_gain_factor = 0.0001

# scene classification layer (Sen2Cor)
class SCL_Classes(object):
    """
    class defining all possible SCL values and their meaning
    (SCL=Sentinel-2 scene classification)
    Class names follow the official ESA documentation available
    here:
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    (last access on 27.05.2021)
    """
    @classmethod
    def values(cls):
        values = {
            0 : 'no_data',
            1 : 'saturated_or_defective',
            2 : 'dark_area_pixels',
            3 : 'cloud_shadows',
            4 : 'vegetation',
            5 : 'non_vegetated',
            6 : 'water',
            7 : 'unclassified',
            8 : 'cloud_medium_probability',
            9 : 'cloud_high_probability',
            10: 'thin_cirrus',
            11: 'snow'
            }
        return values

    @classmethod
    def colors(cls):
        """
        Scene Classification Layer colors trying to mimic the default
        color map from ESA
        """
        scl_colors = [
                'black',            # nodata
                'red',              # saturated or defective
                'dimgrey',          # dark area pixels
                'chocolate',        # cloud shadows
                'yellowgreen',      # vegetation
                'yellow',           # bare soil
                'blue',             # open water
                'gray',             # unclassified
                'darkgrey',         # clouds medium probability
                'gainsboro',        # clouds high probability
                'mediumturquoise',  # thin cirrus
                'magenta'           # snow
            ]
        return scl_colors
