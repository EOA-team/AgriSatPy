'''
Created on Nov 24, 2021

@author: graflu
'''

from enum import Enum

# available processing levels
class ProcessingLevels(Enum):
    L1C = 'LEVEL1C'
    L2A = 'LEVEL2A'


# native spatial resolution of the S2 bands
band_resolution = {
    'B01': 60,
    'B02': 10,
    'B03': 10,
    'B04': 10,
    'B05': 20,
    'B06': 20,
    'B07': 20,
    'B08': 10,
    'B8A': 10,
    'B09': 60,
    'B10': 60,
    'B11': 20,
    'B12': 20
        
}

s2_band_mapping = {
        'B02': 'blue',
        'B03': 'green',
        'B04': 'red',
        'B05': 'red_edge_1',
        'B06': 'red_edge_2',
        'B07': 'red_edge_3',
        'B08': 'nir_1',
        'B8A': 'nir_2',
        'B11': 'swir_1',
        'B12': 'swir_2'
}

# S2 data is stored as uint16
s2_gain_factor = 0.0001
