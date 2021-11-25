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