'''
Created on Jul 14, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

import os
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
import rasterio.mask

from typing import List
from typing import Optional
from typing import Tuple
from pathlib import Path

from agrisatpy.config.sentinel2 import Sentinel2
from agrisatpy.config import get_settings

S2 = Sentinel2()
Settings = get_settings()
logger = Settings.logger


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


def buffer_fieldpolygons(in_gdf: gpd.GeoDataFrame,
                         buffer: float,
                         drop_multipolygons: Optional[bool]=True
                         ) -> gpd.GeoDataFrame:
    '''
    creates a buffer for field polygons and returns a new geodataframe.

    :param fieldpolygons:
        geodataframe created from shapefile with polygons
    :paran buffer:
        buffer distance in metric units to create around original polygon geometries
    :param drop_multipolygons:
        keep only polygons and drop multi-polygons. The default is True.
    :return buffered:
        geodataframe with buffered geometries
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
    :return statistics:
        dictionary with extracted SCL statistic per field parcel
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


def raster2table(
        in_file: Path,
        buffer: float, 
        id_column: str,       
        in_file_polys: Optional[str]='',
        in_gdf_polys: Optional[gpd.GeoDataFrame]=None,     
        out_colnames: Optional[List[str]]=None,
        **kwargs
    ) -> Tuple[pd.DataFrame]:
    '''
    Function to extract raster pixel values for a set of vector features
    (Polygons) from any raster data (single and multi-band).

    :param in_file: 
        Path to the raster file from which to extract pixel values.
        We recommend to use GeoTiff files.
    :param buffer:
        Value to buffer the individual field polygons by (negative for inward buffer).
    :param id_column:
        COlumn name that contains the ID for each individual field polygon.
    :param in_file_polys: 
        Path to the shapefile containing the Polygons (can be other filetype as well).
    :param in_file_polys:
        ESRI shapefile containung 1 to N (field) polygons. Alternatively, a
        geopandas GeoDataFrame (e.g., from a database query) can be passed.
    :param in_gdf_polys:
        instead of a file with field parcel geometries, a geo-dataframe can
        be passed directly.
    :param out_colnames:
        optional list of column names for the resulting dataframe. Must equal the
        number of raster bands. Otherwise, the information is taken from the
        raster band description (if available).
    :param **kwargs:
        nodata = kwargs.get('nodata_refl', 64537)
    :return out_DF:
        Extracted Pixel values including X & Y Coordinates, EPSG code,
        and the ID of the polygons
    '''

    # check if files exist first
    if not in_file.exists():
        raise DataNotFoundError(f'Could not find {in_file}')

    # open input .tiff file
    bandstack = rio.open(in_file)

    # check for user-defined nodata values for reflectance
    nodata_refl = kwargs.get('nodata', S2.NODATA_REFLECTANCE)

    # read in bandlist
    if out_colnames is None:
        bandlist = list(bandstack.descriptions)
    else:
        bandlist = out_colnames

    # read the shapefile that contains the polyons
    try:
        if in_file_polys != '':
            bbox_parcels = gpd.read_file(in_file_polys)
        else:
            bbox_parcels = in_gdf_polys.copy()
    except Exception as e:
        raise DataNotFoundError(f'Could not read field parcel geoms: {e}')

    # ========================== check CRS ==========================

    # get CRS of satellite data by taking the 1st .JP2 file
    sat_crs = bandstack.crs
    # convert bbox to S2 CRS before buffering to be in the correct coordinate system
    bbox_s2_crs = bbox_parcels.to_crs(sat_crs)

    # calculate the buffer
    bbox_parcels_buffered = buffer_fieldpolygons(
        in_gdf=bbox_s2_crs, 
        buffer=buffer
    )

    # ========================== loop over IDs ==========================
    full_DF = []
   
    for idx in bbox_parcels_buffered.index:

        # unfortunately, geopandas still does not support iterrows() on geometries...
        shape = bbox_parcels_buffered.loc[[idx]]
        logger.info(f"Extracting field parcel with ID {shape[id_column]}")

    # ========================== Loop over bands! ==========================
        flat_band_rflt_per_ID = []
        try:
            out_band, out_transform = rio.mask.mask(
                bandstack,
                shape.geometry,
                crop=True, 
                all_touched=True, # IMPORTANT!
                nodata = nodata_refl
            )
        except Exception as e:
            logger.warning(f'Couldnot clip feature {shape[id_column]}: {e}')
            # if the feature could not be clipped (e.g., because it is not located
            # within the extent of the raster) flag the feature and continue with the next
            continue
       
        for idx in range(len(bandlist)):
            # flatten spectral values from 2d to 1d along columns (order=F(ortran))
            flat_band_n = out_band[idx, :, :].flatten(order='F')
            flat_band_rflt_per_ID.append(flat_band_n)
      
        # coerce to DF
        per_ID_df = pd.DataFrame(flat_band_rflt_per_ID).transpose()
        # add bandnames
        per_ID_df.columns = bandlist
        
        # ========== Get coordinates ==========
        # out_transform[0] = resolution in x direction (resolution_x)
        # out_transform[4] = resolution in y direction (resolution y)
        # out_transform[2] = upper left x coordinate (ulx)
        # out_transform[5] = upper left y coordinate (uly)
        resolution_x  = out_transform[0]
        resolution_y = out_transform[4]
        ulx = out_transform[2]
        uly = out_transform[5]

        # get rows and columns of extracted spatial subset of the image 
        maxcol = out_band.shape[2]
        maxrow = out_band.shape[1]

        # get coordinates of every item in out_image
        max_x_coord = ulx + maxcol * resolution_x
        max_y_coord = uly + maxrow * resolution_y
        x_coords = np.arange(ulx, max_x_coord, resolution_x)
        y_coords = np.arange(uly, max_y_coord, resolution_y)

        # flatten x coordinates along the y-axis
        flat_x_coords = np.repeat(x_coords, maxrow)
        # flatten y coordinates along the x-axis
        flat_y_coords = np.tile(y_coords, maxcol)
    
        # add coordinates
        per_ID_df["x_coord"] = flat_x_coords
        per_ID_df["y_coord"] = flat_y_coords

        # add field ID
        per_ID_df[id_column] = shape[id_column].values[0]
        # add CRS in form of EPSG code
        per_ID_df["epsg"] = bbox_parcels_buffered.crs.to_epsg()
        # append to full DF holding all field_IDs
        full_DF.append(per_ID_df)

    # convert full_DF from list to dataframe
    out_DF = pd.concat(full_DF)
    # drop no-data pixels (all reflectance values equal zero)
    out_DF = out_DF.loc[(out_DF[bandlist] != nodata_refl).all(axis=1)]

    return out_DF

if __name__ == '__main__':

    in_file = Path('/mnt/ides/Lukas/04_Work/DEM/hillshade_eschikon.tif')
    in_file_polys = Path('/mnt/ides/Lukas/04_Work/ESCH_2021/ZH_Polygons_2020_ESCH_EPSG32632.shp')
    buffer = 0
    id_column = 'GIS_ID'
    out_colnames = ['hillshade']
    out_df, _ = raster2table(
        in_file=in_file,
        buffer=buffer,
        id_column=id_column,
        in_file_polys=in_file_polys,
        out_colnames=out_colnames
    )
    out_df.to_csv('/mnt/ides/Lukas/04_Work/DEM/test.csv')
