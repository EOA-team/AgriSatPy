'''
Created on Aug 10, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

import os
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from lxml import etree
from scipy.interpolate import interp2d
from agrisatpy.config import get_settings

logger = get_settings().logger


def search_mtd_xml(
        raw_archive_path: Path,
        entry: pd.Series
    ) -> str:
    """
    searches for MTD_TL.xml of a Sentinel-2 scene and returns the full
    path to it

    :param raw_archive_path:
        directory containing all Sentinel-2 scenes in .SAFE structure
    :param entry:
        pandas series containing the main metadata about the scene,
        specifically the PRODUCT_URI
    """
    dot_safe_dir = raw_archive_path.joinpath(entry.PRODUCT_URI)

    # in case the scene comes from Mundi the .SAFE is missing
    if not dot_safe_dir.exists():
        dot_safe_dir = raw_archive_path.joinpath(
            os.path.splitext(entry.PRODUCT_URI)[0]
        )
    search_expression = dot_safe_dir.joinpath(
        'GRANULE' + os.sep + '*' + os.sep + 'MTD_TL.xml'
    )
    return glob.glob(str(search_expression))[0]


def get_grid_values_from_xml(
        tree_node,
        xpath_str
    ) -> float:
    '''
    Receives a XML tree node and a XPath parsing string
    and search for children matching the string.
    Then, extract the VALUES in <values> v1 v2 v3 </values>
    <values> v4 v5 v6 </values> format as numpy array
    Loop through the arrays to compute the mean.

    Function written by Mauricio Cordeiro, Université Paul Sabatier, Toulouse
    April 2016
    
    https://towardsdatascience.com/how-to-implement-sunglint-detection-for-sentinel-2-images-in-python-using-metadata-info-155e683d50
    '''
    node_list = tree_node.xpath(xpath_str)

    arrays_lst = []
    for node in node_list:
        values_lst = node.xpath('.//VALUES/text()')
        values_arr = np.array(list(map(lambda x: x.split(' '), values_lst))).astype('float')
        arrays_lst.append(values_arr)

    return np.nanmean(arrays_lst, axis=0)


def get_s2_pixel_angles(
        pixels: gpd.GeoDataFrame,
        meta_df: pd.DataFrame,
        raw_archive_path: Path
    ) -> gpd.GeoDataFrame:
    """
    Function to extract the illumination and vieiwing angles from
    Sentinel-2 scenes on a per pixel basis (rather than taking a
    scene-wide average value) from the MTD_TL.xml file for a given
    date using a function provided by Mauricio Cordeiro (2016).

    Knowledge about the angles is important for potential BRDF
    correction and the calculation of the plant phenology index (PPI,
    requires sun azimuth angle, only)

    The extracted angles will be stored in new columns named
    sun_zenith_angle, sun_azimuth_angle, view_zenith_angle and
    view_azimuth_angle and are stored in degrees (deg).

    :param pixels:
        geodataframe containing the pixel in projection of the S2 tile.
        Assumes that the pixels are from a single S2 tile, only
    :param meta_df:
        basic metadata extracted using AgriSatPy
    :param raw_archive_path:
        path where the actual Sentinel-2 scenes are stored in .SAFE
        folder structure
    """
    # loop over dates in the pixels df
    dates = pixels.date.unique()

    # define columns for storing pixel-based angle values
    pixels['sun_zenith_angle'] = np.nan
    pixels['sun_azimuth_angle'] = np.nan
    pixels['viewing_zenith_angle'] = np.nan
    pixels['viewing_azimuth_angle'] = np.nan

    # extract pixel x and y coordinate
    pixels['x_coord'] = pixels.geometry.x
    pixels['y_coord'] = pixels.geometry.y

    res_list = []
    for pixel_date in list(dates):

        meta_date = meta_df[meta_df.SENSING_DATE == str(pixel_date)]
        pixels_date = pixels[pixels.date == pixel_date].copy()

        # check if the scene is splitted into two datasets
        # the sun angles are the same for both datasets, but not the
        # viewing angles
        is_split = False
        if meta_date.shape[0] == 1:
            entry = meta_date.iloc[0]
        else:
            # use the first dataset for the sun angles
            is_split = True
            entry = meta_date.iloc[0]
        
        # construct the path to the metadata xml and parse it
        try:
            meta_xml = search_mtd_xml(
                raw_archive_path=raw_archive_path,
                entry=entry
            )
            tree = etree.parse(meta_xml)
        except Exception as e:
            logger.error(f'Could not read metadata xml: {e}')

        root = tree.getroot()

        # get the angles
        sun_zenith = get_grid_values_from_xml(root,
                                              './/Sun_Angles_Grid/Zenith'
        )
        sun_azimuth = get_grid_values_from_xml(root,
                                               './/Sun_Angles_Grid/Azimuth'
        )

        view_zenith = get_grid_values_from_xml(root,
                                               './/Viewing_Incidence_Angles_Grids/Zenith'
        )
        view_azimuth = get_grid_values_from_xml(root,
                                                './/Viewing_Incidence_Angles_Grids/Azimuth'
        )

        # if the scene is splitted into two datasets, retrieve the viewing angles from
        # the second dataset as well and combine them by averaging them
        if is_split:
            entry = meta_date.iloc[1]
            
            try:
                meta_xml = search_mtd_xml(
                    raw_archive_path=raw_archive_path,
                    entry=entry
                )
                tree = etree.parse(meta_xml)
            except Exception as e:
                print(e)
            root = tree.getroot()

            view_zenith2 = get_grid_values_from_xml(root,
                                                    './/Viewing_Incidence_Angles_Grids/Zenith'
            )
            view_azimuth2 = get_grid_values_from_xml(root,
                                                     './/Viewing_Incidence_Angles_Grids/Azimuth'
            )
            view_zenith = np.nanmean([view_zenith, view_zenith2], axis=0)
            view_azimuth = np.nanmean([view_azimuth, view_azimuth2], axis=0)

        # in case of the viewing angles check for nans (outside the actual image date)
        # and fill by them by the mean of non-nan values
        # -> scipy interp2 cannot handle nans in the data
        if np.isnan(view_azimuth).any():
            view_azimuth[np.isnan(view_azimuth)] = np.nanmean(view_azimuth)
        if np.isnan(view_zenith).any():
            view_zenith[np.isnan(view_zenith)] = np.nanmean(view_zenith)

        # calculate the coordinates in x and y direction using the upper left
        # corner and knowing that the spacing of the angle values is 5000 m
        # NOTE: From the documentation it is not really clear where the origin of
        # the angle arrays is located in terms of x and y, but the s2tbx source code
        # seems to use the ULX and ULY of the 10m bands
        row_step = 5000
        col_step = 5000
        x_coords = np.arange(meta_date.ULX.values[0],
                             meta_date.ULX.values[0] + sun_zenith.shape[1] * col_step,
                             col_step).ravel()
        y_coords = np.arange(meta_date.ULY.values[0],
                             meta_date.ULY.values[0] + sun_zenith.shape[0] * row_step,
                             row_step).ravel()

        # calculate the pixel values using bilinear interpolation
        f_sza = interp2d(x_coords, y_coords, sun_zenith, kind='linear')
        f_saa = interp2d(x_coords, y_coords, sun_azimuth, kind='linear')
        f_vza = interp2d(x_coords, y_coords, view_zenith, kind='linear')
        f_vaa = interp2d(x_coords, y_coords, view_azimuth, kind='linear')

        # find the closes x and y coordinates for each pixel and perform the
        # bilinear interpolation
        pixels_date['sun_zenith_angle'] = pixels_date.apply(
            lambda x, f=f_sza: f(x.x_coord, x.y_coord)[0], axis=1
        )
        pixels_date['sun_azimuth_angle'] = pixels_date.apply(
            lambda x, f=f_saa: f(x.x_coord, x.y_coord)[0], axis=1
        )
        pixels_date['view_zenith_angle'] = pixels_date.apply(
            lambda x, f=f_vza: f(x.x_coord, x.y_coord)[0], axis=1
        )
        pixels_date['view_azimuth_angle'] = pixels_date.apply(
            lambda x, f=f_vaa: f(x.x_coord, x.y_coord)[0], axis=1
        )
        res_list.append(pixels_date)

    return pd.concat(res_list)
