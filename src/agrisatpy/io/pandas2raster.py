# -*- coding: utf-8 -*-
"""
Coerce extracted parcel-wise CSV pixel (i.e., regularly-gridded) values to geoTiff raster file.

There are four options:
    1) write all field parcels into a single geoTiff file (makes sense if all fields are
       spatially adjoint)
    2) write each field parcel into a single geoTiff file (one file per parcel; makes sense if the
       fields are spatially widely scattered)
    3) same as 1) but with the option to exclude one or more field parcels
    4) same as 2) but with the option to exclude one or more field parcels

This script is meant to be generic, i.e., it should be able to handle ANY kind of raster data!
"""

# TODO: update if necessary!

import os
import sys
import pandas as pd
import rasterio as rio
import numpy as np
from typing import List
from typing import  Tuple
from typing import TypeVar
from typing import Optional
from typing import Union
from pathlib import Path
from rasterio.profiles import Profile

from agrisatpy.config import get_settings

Settings = get_settings()
logger = Settings.logger


def extract_epsg(
        df: pd.DataFrame,
        col_crs: str='CRS_epsg'
    ) -> int:
    """
    returns the EPSG code from a Dataframe that has a EPSG column.
    Assumes that the EPSG is the same for the entire dataframe.

    :param df:
        dataframe from which to extract the EPSG
    :param col_crs:
        name of the column with the EPSG code (Def: 'CRS_epsg')
    :return:
        integer EPSG code
    """
    return df[col_crs].unique()[0]


def construct_profile(
        epsg_int: int,
        ulx: Union[int,float],
        uly: Union[int,float],
        target_resolution: Union[int,float],
        cols: int,
        rows: int, 
        bands: int
    ) -> dict:
    """
    constructs a dict-like structure (profile) required by rasterio
    to georeference the output geoTiffs files

    :param epsg_int:
        EPSG code denoting the spatial reference system
    :param ulx:
        upper left corner x coordinate (in CRS specified by EPSG)
    :param uly:
        upper left corner y coordinate (in CRS specified by EPSG)
    :param resolution:
        pixel size of the resulting image
    :param cols:
        number of columns (width of the resulting image)
    :param rows:
        number of rows (height of the resulting image)
    :param bands:
        number of the bands in the output image
    :return profile:
        profile required for geo-referencing the image data
    """
    profile = {}
    # define the Affine projection parameters first
    transformation = rio.Affine(target_resolution,
                                0.0,
                                ulx,
                                0.0,
                                -target_resolution,
                                uly)
    profile['transform'] = transformation
    profile['width'] = cols
    profile['height'] = rows
    profile['count'] = bands
    # set dtype per default to float64 to handle every kind of data
    profile['dtype'] = np.float64
    profile['crs'] = rio.crs.CRS(init=f'epsg:{epsg_int}')
    return profile


def polygon2raster(
        in_df: pd.DataFrame,
        epsg_int: int, 
        column_selection: List[str],
        target_resolution: Union[int,float],
        colname_x: Optional[str]='x_coord',
        colname_y: Optional[str]='y_coord'
    ) -> Tuple[np.array, Profile]:
    """
    helper method to convert a polygon or collection of polygons
    into a 3-dim raster (shape: bands, rows, cols) with proper geolocation
    information as rasterio compliant profile dictionary

    :param in_df:
        dataframe with x and y coordinates and values
    :param epsg_int:
        EPSG code denoting the spatial reference system
    :param target_resolution:
        spatial resolution (i.e, pixel size) of the output image
    :param column_selection:
        columns of the dataframe which shall be written to geoTiff (as bands)
    :param colname_x:
        name of the dataframe column with the x coordinates. The default
        is 'x_coord'.
    :param colname_y:
        name of the dataframe column with the y coordinates. The default
        is 'y_coord'.
    :return (img_arr, profile):
        returns the image array and the profile for proper geo-localisation
    """
    # get array of polygon values
    # get upper left X/Y coordinates
    ulx = in_df[colname_x].min()
    uly = in_df[colname_y].max()
    # get lower right X/Y coordinates to span the img matrix
    lrx = in_df[colname_x].max()
    lry = in_df[colname_y].min()

    # caluclate max rows along x and y axis
    max_x_coord = int((lrx - ulx) / target_resolution) + 1
    max_y_coord = int((uly - lry) / target_resolution) + 1
    
    # create index lists for coordinates
    x_indices = np.arange(ulx, lrx+target_resolution, step = target_resolution, dtype = int)
    y_indices = np.arange(uly, lry-target_resolution, step = -target_resolution, dtype = int)
    
    # unflatten the DF along the selected columns (e.g. loop over cols)
    bands = len(column_selection)
    img_arr = np.ones(shape=(bands, max_y_coord, max_x_coord)) * np.nan
    for band_index, col in enumerate(column_selection):

        for idx in range(in_df.shape[0]):
            
            x_index = np.where(x_indices == in_df.x_coord.iloc[idx])[0][0]
            y_index = np.where(y_indices == in_df.y_coord.iloc[idx])[0][0]
            
            img_arr[band_index, y_index, x_index] = in_df[col].iloc[idx]

    # generate the output
    # define profile to allow for correct geo-location
    profile = construct_profile(epsg_int=epsg_int,
                                ulx=ulx,
                                uly=uly,
                                target_resolution=target_resolution,
                                cols=max_x_coord,
                                rows=max_y_coord,
                                bands=bands)
    return (img_arr, profile)


def write_image(
        out_file: Path,
        profile: Profile,
        out_array: np.array,
        column_selection: List[str]
    ) -> None:
    """
    helper function to write a 3-d array with shape (bands,rows,cols) to a georeferenced
    image using rasterio. The image band names are the same as the selected columns
    in the original dataframe. It is assumed that profile already has the correct (i.e.,
    updated properties in terms of image size and geolocalisation)

    :param out_file:
        name of the image file to write
    :param profile:
        rasterio dataset profile
    :param out_array:
        numpy array containing the image values. Shape must be (bands,rows,cols)
    :param column_selection:
        list of columns in the original dataframe that were extracted. The
        image bands are named after them
    """
    # open dataset writer and write array to image
    with rio.open(out_file, 'w', **profile) as dst:
        dst.write(out_array)

        # update band descriptions
        for idx in range(len(column_selection)):
            dst.set_band_description(idx+1, column_selection[idx])


def pandas2raster(
        in_df: pd.DataFrame,
        out_dir: str,
        column_selection: List[str],
        target_resolution: Union[int,float],
        product_name: str,
        id_column: Optional[str]='polygon_ID',
        polygon_selection: Optional[List[TypeVar('T')]]=[],
        single_out_file: Optional[bool]=False,
    ) -> None:
    """
    converts a pandas dataframe with field polygons (identified by the id_column
    variable) to a one or more geoTiff file(s) using a user-defined list of dataframe columns.
    Each dataframe column is written as band to the geoTiff file in the order
    they appear in the input list.
    
    ATTENTION:
        Function overwrites .tiffs with different settings. It is recommended to
        create different output folders for different polygon selections/subsets.

    :param in_df:
        dataframe with data to be converted to geoTiffs
    :param out_dir:
        directory where to store the geoTiffs
    :param column_selection:
        columns of the dataframe which shall be written to geoTiff (as bands)
    :param target_resolution:
        spatial resolution of the output raster
    :param product_name:
        prefix for the image files to be generated (e.g. 'ESCH_20200402_S2A_')
    :param id_column:
        column name that contains the unique field ID (Def.: polygon_ID)
    :param polygon_selection:
        if only a subset of the available fields should be written to output
        then a list of those polygon IDs can be passed. If left empty (default)
        then all available fields are written to output
    :param single_out_file:
        if True will write all polygons into a single geoTiff file (Def.: False)
    """
    # check which field IDs are available
    uid_list = in_df[id_column].unique()

    # check which fields were selected. If the polygon_selection list is empty
    # all fields will be written to output
    if len(polygon_selection) == 0:
        polygon_selection = uid_list

    # extract the CRS
    crs = extract_epsg(in_df)

    # store the column names in the filename
    col_str  = '-'.join(column_selection)

    # write each field into a single geoTiff file (default)
    if not single_out_file:
        # loop over field ids and extract the selected column values
        for uid in uid_list:
        
            # cycle if the field ID should not be written to output
            if uid not in polygon_selection: continue
    
            logger.info(f'Converting raster data for polygon with ID: {uid}')
    
            # define name of the output
            out_file_polygon = out_dir.joinpath(
                f'{product_name}_fieldpolygon_{uid}_{col_str}.tiff'
            )
    
            # filter for uid
            polygon_in_df = in_df[in_df[id_column] == uid]

            # convert point-wise pixel values (i.e., rows in dataframe) to 3-d numpy array
            try:
                img_arr, profile = polygon2raster(
                    in_df=polygon_in_df,
                    epsg_int=crs,
                    target_resolution=target_resolution,
                    column_selection=column_selection
                )
            except Exception as e:
                logger.error(f'Could not convert data for polygon with ID {uid}: {e}')
                continue

            # write image to file
            try:
                write_image(
                    out_file=out_file_polygon,
                    profile=profile,
                    out_array=img_arr,
                    column_selection=column_selection
                )
            except Exception as e:
                logger.error(f'Could not write raster data to file (polygon ID {uid}): {e}')
                continue
            

    # write all files into a single geoTiff file
    else:
        
        polygon_in_df = in_df[in_df[id_column].isin(polygon_selection)]
        out_file_polygon = os.path.join(out_dir,
                                       f'{product_name}_fieldpolygons_all_{col_str}.tiff')
        try:
            img_arr, profile = polygon2raster(in_df=polygon_in_df,
                                              epsg_int=crs,
                                              target_resolution=target_resolution,
                                              column_selection=column_selection)
        except Exception as e:
            logger.error(f'Could not convert data from dataframe to numpy array: {e}')
            sys.exit()

        # write image to file
        try:
            write_image(out_file=out_file_polygon,
                        profile=profile,
                        out_array=img_arr,
                        column_selection=column_selection)
        except Exception as e:
            logger.error(f'Failed to write {out_file_polygon}: {e}')
            sys.exit()
