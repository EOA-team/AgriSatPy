'''
Created on Jul 14, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

import os
import glob
from typing import List
import numpy as np
import geopandas as gpd

class DataNotFoundError(Exception):
    pass


class SCL_Classes(object):
    """
    class defining all possible SCL values and their meaning
    (SCL=Sentinel-2 scene classification)
    Class names follow the official ESA documentation available
    here:
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    (last access on 27.05.2021)
    """
    def values(self):
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


def get_S2_bandfiles(in_dir: str) -> List[str]:
    '''
    returns all JPEG-2000 files (*.jp2) found in a directory

    :param search_dir:
        directory containing the JPEG2000 band files
    '''
    search_pattern = '*B*.jp2'
    return glob.glob(os.path.join(in_dir, search_pattern))


def get_S2_sclfile(in_dir: str) -> str:
    '''
    return the path to the S2 SCL (scene classification file) 20m resolution!

    :param search_dir 
        directory containing the SCL band files (jp2000 file).
    '''
    search_pattern = "*_SCL_20m.jp2"
    return glob.glob(os.path.join(in_dir, search_pattern))[0]


def buffer_fieldpolygons(in_gdf: gpd.GeoDataFrame,
                         buffer: float,
                         drop_multipolygons: bool=True
                         ) -> gpd.GeoDataFrame:
    '''
    creates a buffer for field polygons and returns a new geodataframe.

    :param fieldpolygons:
        geodataframe created from shapefile with polygons
    :paran buffer:
        buffer distance in metric units to create around original polygon geometries
    :param drop_multipolygons:
        keep only polygons and drop multi-polygons. The default is True.
    '''
    buffered = in_gdf.copy()
    # resolution is set to 16 to obtain high-quality edge regions
    buffered.geometry = buffered.geometry.buffer(buffer,resolution=16)
    
    # omit empty geometries after buffering
    buffered = buffered[~buffered.geometry.is_empty]

    # in case of inward buffering it might happen that single field polygons
    # are split into multipolygons that are too small for further processing
    if drop_multipolygons:
        buffered = buffered[buffered.geom_type != 'MultiPolygon']

    return buffered


def compute_parcel_stat(in_array: np.array,
                        nodata_value: int
                        ) -> dict:
    """
    calculates the percentage of pixels per SCL class within a parcel (field)
    polygon. 100% equals the number of pixels that are located within the
    polygon geometry (i.e., that were extracted before using rio.mask.mask)

    :param in_array:
        array with extract SCL (scene classification) values
    :param nodata_value:
        value that shall be interpreted as nodata
    """
    
    scl_classes = SCL_Classes().values()
    # get number of pixels in the current polygon (all SCL values but nodata_value)
    num_pixels = np.count_nonzero(in_array[in_array != nodata_value])
    # load SCL values and count their occurence in the raster
    statistics = dict.fromkeys(scl_classes.values(), np.nan)
    # check if num_pixels is greater than zero
    if num_pixels <= 0:
        return statistics
    for class_key, class_value in scl_classes.items():
        class_occurence = in_array[in_array == class_key].shape[0]
        statistics[class_value] = (class_occurence / num_pixels) * 100.
        
    statistics["n_pixels_tot"] = num_pixels
    
    return statistics

