#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:09:42 2019

Updated on Fri Jun 25 2021

@author: graflu

This module contains functions to extract relevant
scene-specific Sentinel-2 metadata supporting
L1C and L2A (sen2core-derived) processing level
"""

import os
import glob
import numpy as np
from datetime import datetime
from xml.dom import minidom
from pyproj import Transformer
from pathlib import Path
import pandas as pd

from agrisatpy.config import get_settings

logger = get_settings().logger

class UnknownProcessingLevel(Exception):
    pass


def parse_MTD_TL(in_file: Path
                ) -> dict:
    """
    Parses the MTD_TL.xml metadata file provided by ESA.This metadata
    XML is usually placed in the GRANULE subfolder of a ESA-derived
    S2 product and named 'MTD_TL.xml'.

    The 'MTD_TL.xml' is available for both processing levels (i.e.,
    L1C and L2A). The function is able to handle both processing
    sources and returns some entries available in L2A processing level,
    only, as None type objects.
    
    The function extracts the most important metadata from the XML and
    returns a dict with those extracted entries.
    
    :param in_file:
        filepath of the scene metadata xml
    :return metadata:
        dict with extracted metadata entries
    """
    # parse the xml file into a minidom object
    xmldoc = minidom.parse(in_file)
    
    # now, the values of some relevant tags can be extracted:
    metadata = dict()
    
    # get tile ID of L2A product and its corresponding L1C counterpart
    tile_id_xml = xmldoc.getElementsByTagName('TILE_ID')
    tile_id = tile_id_xml[0].firstChild.nodeValue
    scene_id = tile_id.split('.')[0]
    metadata['SCENE_ID'] = scene_id

    # check if the scene is L1C or L2A
    is_l1c = False
    try:
        l1c_tile_id_xml = xmldoc.getElementsByTagName('L1C_TILE_ID')
        l1c_tile_id = l1c_tile_id_xml[0].firstChild.nodeValue
        l1c_tile_id = l1c_tile_id.split('.')[0]
        metadata['L1C_TILE_ID'] = l1c_tile_id
    except Exception:
        logger.info(f'{scene_id} is L1C processing level')
        is_l1c = True

    # sensing time (acquisition time)
    sensing_time_xml = xmldoc.getElementsByTagName('SENSING_TIME')
    sensing_time = sensing_time_xml[0].firstChild.nodeValue
    metadata['SENSING_TIME'] = sensing_time
    metadata['SENSING_DATE'] = datetime.strptime(
        sensing_time.split('T')[0],'%Y-%m-%d').date()

    # number of rows and columns for each resolution -> 10, 20, 60 meters
    nrows_xml = xmldoc.getElementsByTagName('NROWS')
    ncols_xml = xmldoc.getElementsByTagName('NCOLS')
    resolutions = ['_10m', '_20m', '_60m']
    # order: 10, 20, 60 meters spatial resolution
    for ii in range(3):
        nrows = nrows_xml[ii].firstChild.nodeValue
        ncols = ncols_xml[ii].firstChild.nodeValue
        metadata['NROWS' + resolutions[ii]] = int(nrows)
        metadata['NCOLS' + resolutions[ii]] = int(ncols)

    # EPSG-code
    epsg_xml = xmldoc.getElementsByTagName('HORIZONTAL_CS_CODE')
    epsg = epsg_xml[0].firstChild.nodeValue
    metadata['EPSG'] = int(epsg.split(':')[1])

    # Upper Left Corner coordinates -> is the same for all three resolutions
    ulx_xml = xmldoc.getElementsByTagName('ULX')
    uly_xml = xmldoc.getElementsByTagName('ULY')
    ulx = ulx_xml[0].firstChild.nodeValue
    uly = uly_xml[0].firstChild.nodeValue
    metadata['ULX'] = float(ulx)
    metadata['ULY'] = float(uly)
    # endfor

    # extract the mean zenith and azimuth angles
    # the sun angles come first followed by the mean angles per band
    zenith_angles = xmldoc.getElementsByTagName('ZENITH_ANGLE')
    metadata['SUN_ZENITH_ANGLE'] = float(zenith_angles[0].firstChild.nodeValue)

    azimuth_angles = xmldoc.getElementsByTagName('AZIMUTH_ANGLE')
    metadata['SUN_AZIMUTH_ANGLE'] = float(azimuth_angles[0].firstChild.nodeValue)

    # get the mean zenith and azimuth angle over all bands
    sensor_zenith_angles = [float(x.firstChild.nodeValue) for x in zenith_angles[1::]]
    metadata['SENSOR_ZENITH_ANGLE'] = np.mean(np.asarray(sensor_zenith_angles))

    sensor_azimuth_angles = [float(x.firstChild.nodeValue) for x in azimuth_angles[1::]]
    metadata['SENSOR_AZIMUTH_ANGLE'] = np.mean(np.asarray(sensor_azimuth_angles))
    
    # extract scene relevant data about nodata values, cloud coverage, etc.
    cloudy_xml = xmldoc.getElementsByTagName('CLOUDY_PIXEL_PERCENTAGE')
    cloudy = cloudy_xml[0].firstChild.nodeValue
    metadata['CLOUDY_PIXEL_PERCENTAGE'] = float(cloudy)

    degraded_xml = xmldoc.getElementsByTagName('DEGRADED_MSI_DATA_PERCENTAGE')
    degraded = degraded_xml[0].firstChild.nodeValue
    metadata['DEGRADED_MSI_DATA_PERCENTAGE'] = float(degraded)

    # the other tags are available in L2A processing level, only
    if not is_l1c:
        nodata_xml = xmldoc.getElementsByTagName('NODATA_PIXEL_PERCENTAGE')
        nodata = nodata_xml[0].firstChild.nodeValue
        metadata['NODATA_PIXEL_PERCENTAGE'] = float(nodata)
    
        darkfeatures_xml = xmldoc.getElementsByTagName('DARK_FEATURES_PERCENTAGE')
        darkfeatures = darkfeatures_xml[0].firstChild.nodeValue
        metadata['DARK_FEATURES_PERCENTAGE'] = float(darkfeatures)
    
        cs_xml = xmldoc.getElementsByTagName('CLOUD_SHADOW_PERCENTAGE')
        cs = cs_xml[0].firstChild.nodeValue
        metadata['CLOUD_SHADOW_PERCENTAGE'] = float(cs)
    
        veg_xml = xmldoc.getElementsByTagName('VEGETATION_PERCENTAGE')
        veg = veg_xml[0].firstChild.nodeValue
        metadata['VEGETATION_PERCENTAGE'] = float(veg)
    
        noveg_xml = xmldoc.getElementsByTagName('NOT_VEGETATED_PERCENTAGE')
        noveg = noveg_xml[0].firstChild.nodeValue
        metadata['NOT_VEGETATED_PERCENTAGE'] = float(noveg)
    
        water_xml = xmldoc.getElementsByTagName('WATER_PERCENTAGE')
        water = water_xml[0].firstChild.nodeValue
        metadata['WATER_PERCENTAGE'] = float(water)
    
        unclass_xml = xmldoc.getElementsByTagName('UNCLASSIFIED_PERCENTAGE')
        unclass = unclass_xml[0].firstChild.nodeValue
        metadata['UNCLASSIFIED_PERCENTAGE'] = float(unclass)
    
        cproba_xml = xmldoc.getElementsByTagName('MEDIUM_PROBA_CLOUDS_PERCENTAGE')
        cproba = cproba_xml[0].firstChild.nodeValue
        metadata['MEDIUM_PROBA_CLOUDS_PERCENTAGE'] = float(cproba)
    
        hcproba_xml = xmldoc.getElementsByTagName('HIGH_PROBA_CLOUDS_PERCENTAGE')
        hcproba = hcproba_xml[0].firstChild.nodeValue
        metadata['HIGH_PROBA_CLOUDS_PERCENTAGE'] = float(hcproba)
    
        thcirrus_xml = xmldoc.getElementsByTagName('THIN_CIRRUS_PERCENTAGE')
        thcirrus = thcirrus_xml[0].firstChild.nodeValue
        metadata['THIN_CIRRUS_PERCENTAGE'] = float(thcirrus)
        
        # try catch because of version differences in xml file
        try:
            ccover_xml = xmldoc.getElementsByTagName('CLOUD_COVERAGE_PERCENTAGE')
            ccover = ccover_xml[0].firstChild.nodeValue
            metadata['CLOUD_COVERAGE_PERCENTAGE'] = ccover
        except IndexError:
            pass

        snowice_xml = xmldoc.getElementsByTagName('SNOW_ICE_PERCENTAGE')
        snowice = snowice_xml[0].firstChild.nodeValue
        metadata['SNOW_ICE_PERCENTAGE'] = float(snowice)

    # calculate the scene footprint in geographic coordinates
    metadata['geom'] = get_scene_footprint(sensor_data=metadata)

    return metadata


def parse_MTD_MSI(in_file: str
                 ) -> dict:
    """
    parses the MTD_MSIL1C or MTD_MSIL2A metadata file that is delivered with
    ESA Sentinel-2 L1C and L2A products, respectively.
    
    The file is usually placed directly in the .SAFE root folder of an
    unzipped Sentinel-2 L1C or L2A scene.

    The extracted metadata is returned as a dict.

    :param in_file:
        filepath of the scene metadata xml
    """
    # parse the xml file into a minidom object
    xmldoc = minidom.parse(in_file)

    # define tags to extract
    tag_list = ['PRODUCT_URI', 'PROCESSING_LEVEL', 'SENSING_ORBIT_NUMBER',
                'SPACECRAFT_NAME', 'SENSING_ORBIT_DIRECTION']

    metadata = dict.fromkeys(tag_list)

    for tag in tag_list:
        xml_elem = xmldoc.getElementsByTagName(tag)
        metadata[tag] = xml_elem[0].firstChild.data

    # extract solar irradiance for the single bands
    bands = ['B01', 'BO2', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09',
             'B10', 'B11', 'B12']
    sol_irrad_xml = xmldoc.getElementsByTagName('SOLAR_IRRADIANCE')
    for idx, band in enumerate(bands):
        metadata[f'SOLAR_IRRADIANCE_{band}'] = float(sol_irrad_xml[idx].firstChild.nodeValue)

    # S2 tile
    metadata['TILE'] = metadata['PRODUCT_URI'].split('_')[5]

    return metadata
    

def get_scene_footprint(sensor_data: dict
                        ) -> str:
    """
    get the footprint (geometry) of a scene by calculating its
    extent using the original UTM coordinates of the scene.
    The obtained footprint is then converted to WGS84 geographic
    coordinates and returned as Extended Well-Known-Text (EWKT)
    string.

    :param sensor_data:
        dict with ULX, ULY, NROWS_10m, NCOLS_10m, EPSG entries
        obtained from the MTD_TL.xml file
    :return wkt:
        extended well-known-text representation of the scene
        footprint
    """
    dst_crs = 'epsg:4326'
    # get the EPSG-code
    epsg = sensor_data['EPSG']
    src_crs = f'epsg:{epsg}'
    # the pixelsize is set to 10 m
    pixelsize = 10.
    
    # use per default the 10m-representation
    ulx = sensor_data['ULX']            # upper left x
    uly = sensor_data['ULY']            # upper left y
    nrows = sensor_data['NROWS_10m']    # number of rows
    ncols = sensor_data['NCOLS_10m']    # number of columns
        
    # calculate the other image corners (upper right, lower left, lower right)
    urx = ulx + (ncols - 1) * pixelsize # upper right x
    ury = uly                           # upper right y
    llx = ulx                           # lower left x
    lly = uly - (nrows + 1) * pixelsize # lower left y
    lrx = urx                           # lower right x
    lry = lly                           # lower right y

    # transform coordinates to WGS84
    transformer = Transformer.from_crs(src_crs, dst_crs)
    uly, ulx = transformer.transform(xx=ulx, yy=uly)
    ury, urx = transformer.transform(xx=urx, yy=ury)
    lly, llx = transformer.transform(xx=llx, yy=lly)
    lry, lrx = transformer.transform(xx=lrx, yy=lry)

    wkt = f'SRID=4326;'
    wkt += f'POLYGON(({ulx} {uly},{urx} {ury},{lrx} {lry},{llx} {lly},{ulx} {uly}))'

    return wkt


def parse_s2_scene_metadata(in_dir: Path
                            ) -> dict:
    """
    wrapper function to extract metadata from ESA Sentinel-2
    scenes. It returns a dict with the metadata most important
    to characterize a given Sentinel-2 scene.

    The function works on both, L1C and L2A (sen2cor-based)
    processing levels. The amount of metadata, however, is
    reduced in the case of L1C since no scene classification
    information is available.

    NOTE: In order to identify scenes and their processing level
    correctly, L2A scenes must have '_MSIL2A_' occuring somewhere
    in the filepath. For L1C, it must be '_MSIL1C_'.

    :param in_dir:
        directory containing the L1C or L2A Sentinel-2 scene
    :return mtd_msi:
        dict with extracted metadata items
    """
    
    # depending on the processing level (supported: L1C and
    # L2A) metadata has to be extracted slightly differently
    # because of different file names and storage locations
    if in_dir.find('_MSIL2A_') > 0:
        # scene is L2A
        mtd_msil2a_xml = str(next(Path(in_dir).rglob('MTD_MSIL2A.xml')))
        mtd_msi = parse_MTD_MSI(in_file=mtd_msil2a_xml)
        # TODO: test if that works
        with open(mtd_msil2a_xml) as xml_file:
            mtd_msi = mtd_msi['mtd_msi_xml'] = xml_file.read()

    elif in_dir.find('_MSIL1C_') > 0:
        # scene is L1C
        mtd_msil1c_xml = str(next(Path(in_dir).rglob('MTD_MSIL1C.xml')))
        mtd_msi = parse_MTD_MSI(in_file=mtd_msil1c_xml)

    else:
        raise UnknownProcessingLevel(
            f'{in_dir} seems not be a valid Sentinel-2 scene')

    mtd_tl_xml = str(next(Path(in_dir).rglob('MTD_TL.xml')))
    mtd_msi.update(parse_MTD_TL(in_file=mtd_tl_xml))

    return mtd_msi


def loop_s2_archive(in_dir: str
                    ) -> pd.DataFrame:
    """
    wrapper function to loop over an entire archive (i.e., collection) of
    Sentinel-2 scenes in either L1C or L2A processing level or a mixture
    thereof.

    The function returns a pandas dataframe for all found entries in the
    archive (i.e., directory). Each row in the dataframe denotes one scene.

    :param in_dir:
        directory containing the Sentinel-2 data (L1C and/or L2A
        processing level). Sentinel-2 scenes are assumed to follow ESA's
        .SAFE naming convention and structure
    :return:
        dataframe with metadata of all scenes handled by the function
        call
    """
    # search for .SAFE subdirectories identifying the single scenes
    s2_scenes = glob.glob(os.path.join(in_dir, '*.SAFE'))
    n_scenes = len(s2_scenes)
    if n_scenes == 0:
        raise UnknownProcessingLevel(
            f'No .SAFE sub-directories where found in {in_dir}')

    # loop over the scenes
    metadata_scenes = []
    for idx, s2_scene in enumerate(s2_scenes):
        print(f'Extracting metadata of {os.path.basename(s2_scene)} ({idx+1}/{n_scenes})')
        try:
            mtd_scene = parse_s2_scene_metadata(in_dir=s2_scene)
            mtd_scene['filepath'] = s2_scene
        except Exception as e:
            logger.error(f'Extraction of metadata failed {s2_scene}: {e}')
            continue
        metadata_scenes.append(mtd_scene)

    # convert to pandas dataframe and return
    return pd.DataFrame.from_dict(metadata_scenes)
