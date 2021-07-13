'''
Created on Jul 13, 2021

@author:     Gregor Perich, Lukas Graf (D-USYS, ETHZ)

@purpose:    Extracts pixel values (i.e., spectra) from single band files
             or bandstacked (and resampled) Sentinel-2 files of (agricultural)
             field parcels.
             Works on L1C and L2A processing level. In the latter case, also
             a descriptive statistics of the scene classification layer (SCL)
             is calculated per parcel and sensing date.
'''

import os
import glob
from typing import List, Tuple
import rasterio as rio
import rasterio.mask
import geopandas as gpd
import numpy as np
import pandas as pd


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
    buffered['geometry'] = buffered['geometry'].buffer(buffer,resolution=16)
    
    # omit empty geometries after buffering
    buffered = buffered[~buffered["geometry"].is_empty]

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


def S2singlebands2table(in_dir: str,
                        in_file_polys: str,
                        buffer: float, 
                        id_column: str,
                        product_date = str,
                        filter_clouds: bool = True,
                        is_L2A: bool=True,
                        **kwargs
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    extracts spectral values of all bands found in a Sentinel-2 granule folder
    (each band is stored as a separate JPEG-2000 file) for a series of field
    polygons. To allow for extracting pure field pixels (i.e, without border
    effects and, thus, mixed spectral signals) a buffer distance can be specified
    in the metric spatial unit of the Sentinel-2 data (i.e., meters). To obtain
    pure field spectra, negative distance values are necessary (e.g., -10.).
    In addition, the SCL (Sentinel-2 Scene Classification) information is extracted and
    stored per pixel. It is possible to filter out specific classes to obtain, e.g.,
    only cloud-free pixels. For each field polygon, the share of each single SCL
    class is computed and stored in a second dataframe which is also returned

    IMPORTANT: All spectral bands must necessarily have the same spatial resolution!
    IMPORTANT: Currently, all geometries that are of type MultiPolygon are dropped
    
    The extracted spectral values are stored in a geopandas geodataframe
    where each row denotes a pixel. The geolocation information of each pixel
    is saved by storing its x and y coordinate (same CRS as the Sentinel-2 data)
    in the pandas dataframe. This allows reconstructing images from the flattened
    list of pixels. Moreover, the ID of each polygon is stored so that all pixels
    belonging to a polygon (i.e, field parcel) can be easily identified.
    
    For S2 scenes w/o any non-cloudy pixel, no data is written to a CSV
    
    :param in_dir:
        directory where the spectral bands are stored as JP2 files
    :param in_file_polys:
        ESRI shapefile containung 1 to N (field) polygons
    :param buffer:
        buffer distance. Set to zero if no buffer shall be created
    :param id_column:
        name of the column with the unqiue polygon (parcel) ID
    :param product_date:
        scene acquisition product_date (derived from metadata = ingestiondate)
        in YYYYMMDD format
    :param cloudfilter:
        should clouds be filtered out or not? Defaults to SCL 
        cloud classes. Accepts kwarg ("scl_classes") for manual input
    :param is_L2A:
        specifies that the data is in is_L2A processing level and the scene classification
        layer (SCL) is available. If False it is assumed that the data is in L1C
        level (thus, SCL is omitted).
    :kwargs:
        scl_2_filterout = kwargs.get('scl_classes', [0, 1, 3, 8, 9, 10, 11])
        nodata_refl = kwargs.get('nodata_refl', 64537)
        nodata_scl = kwargs.get('nodata_scl', 254)
    """

    # ========================== Get list of files ==========================
    # get a list of files containing the single spectral bands
    jp2_files = get_S2_bandfiles(in_dir=in_dir)
    # get the file with SCL (scene classification)
    if is_L2A: scl_file = get_S2_sclfile(in_dir = in_dir)

    # ==========================     Check kwargs   ==========================
    # check for user-defined SCL filtration
    # default is: clouds, snow and ice, cloud shadow and nodata classes
    if not is_L2A: filter_clouds = False  # cloud filtering does not work for L1C level
    if filter_clouds:
        scl_2_filterout = kwargs.get('scl_classes', [0, 1, 3, 8, 9, 10, 11])

    # check for user-defined nodata values for reflectance and SCL
    # nodata value for Sentinel-2 reflectance
    # TODO: check if 64537 is a valid nodata value for reflectance values
    nodata_refl = kwargs.get('nodata_refl', 64537)
    # nodata for scene classification
    nodata_scl = kwargs.get('nodata_scl', 254)

    bandlist = []
    for filename in jp2_files:
        if is_L2A:
            bandname = filename.split(os.sep)[-1].split(".")[0].split("_")[-2]
        else:
            bandname = filename.split(os.sep)[-1].split(".")[0].split("_")[-1]
        bandlist.append(bandname) 

    # read the shapefile that contains the polyons
    bbox_parcels = gpd.read_file(in_file_polys)

    # calculate the buffer
    bbox_parcels_buffered = buffer_fieldpolygons(in_gdf = bbox_parcels, 
                                                 buffer = buffer)

    # ========================== check CRS ==========================

    # get CRS of satellite data by taking the 1st .JP2 file
    sat_crs = rio.open(jp2_files[0]).crs
    # convert buffered bbox to S2 CRS
    bbox_buffered_s2_crs = bbox_parcels_buffered.to_crs(sat_crs)

    # assert that the crs of the buffered polygons is the same as of the inputs
    assert bbox_parcels_buffered.geometry.crs == bbox_parcels.geometry.crs, \
            'CRS mismatch between buffer and input'

    # ========================== loop over IDs ==========================
    full_DF = []
    if is_L2A: parcel_statistics = []  # SCL is available for is_L2A level, only

    for feature in bbox_buffered_s2_crs.iterrows():

        print(f"Calculating ID no. {feature[1][id_column]}")

    # ========================== Loop over bands! ==========================
        flat_band_rflt_per_ID = []
        couldnot_clip = False
        for bandcounter, band_file in enumerate(jp2_files):
            
            # open file 
            with rio.open(band_file, 'r') as src:
                        
                # feature is of type tuple; feature[1] contains the values
                # cast into GDF & transpose for whatever reason
                shape = gpd.GeoDataFrame(feature[1]).transpose()

                try:
                    out_band, out_transform = rio.mask.mask(src,
                                                            shape["geometry"],
                                                            crop = True, 
                                                            all_touched = True, # IMPORTANT!
                                                            nodata = nodata_refl
                                                            )
                except Exception as e:
                    print(f'Couldnot clip feature {feature[1][id_column]}: {e}')
                    # if the feature could not be clipped (e.g., because it is not located
                    # within the extent of the raster) flag the feature and continue with the next
                    couldnot_clip = True
                    break

                # flatten spectral values from 2d to 1d along columns (order=F(ortran))
                flat_band_n = out_band.flatten(order='F')
                flat_band_rflt_per_ID.append(flat_band_n)

        if couldnot_clip:
            continue

        # coerce to DF
        per_ID_df = pd.DataFrame(flat_band_rflt_per_ID).transpose()
        # add bandnames
        per_ID_df.columns = bandlist
        
        # ========== Get coordinates (calculate only once!) ==========
        if bandcounter == len(jp2_files)-1:
        
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
            
            # ======== Get SCL class per pixel ==========
            # works only for is_L2A processing level
            if is_L2A:
                with rio.open(scl_file) as src:
                    shape = gpd.GeoDataFrame(feature[1]).transpose()
                    
                    out_scl, out_transform = rio.mask.mask(src,
                                                            shape["geometry"],
                                                            crop=True, 
                                                            all_touched=True,
                                                            nodata=nodata_scl
                                                            )
    
                    # compute share of SCL classes for the current polygon
                    stats = compute_parcel_stat(in_array=out_scl,
                                                nodata_value=nodata_scl)
    
                    stats[id_column] = feature[1][id_column]
                    parcel_statistics.append(stats)
    
                    # also flatten the SCL values to store them per pixel
                    flat_scl_class = out_scl.flatten(order='F')
                
                # add SCL to ID_df out of rio scope
                per_ID_df["SCL_class"] = flat_scl_class

        # add field ID
        per_ID_df[id_column] = feature[1][id_column]
        # add CRS in form of EPSG code
        per_ID_df["CRS_epsg"] = bbox_buffered_s2_crs.crs.to_epsg()
        # append to full DF holding all field_IDs
        full_DF.append(per_ID_df)

    # coerce parcel statistic to Dataframe (is_L2A only)
    if is_L2A:

        stat_DF = pd.DataFrame(parcel_statistics)
        stat_DF[id_column] = stat_DF[id_column]

    else:
        
        stat_DF = None

    # stat_DF = stat_DF.drop(id_column, axis=1)

    # convert full_DF from list to dataframe
    out_DF = pd.concat(full_DF)
    # drop no-data pixels (all reflectance values equal zero)
    out_DF = out_DF.loc[(out_DF[bandlist] != nodata_refl).all(axis=1)]

    # ========= filter out clouds based on SCL ================
    # works for is_L2A data only
    if is_L2A:
        out_DF = out_DF.loc[out_DF["SCL_class"] != nodata_scl]
    if filter_clouds and is_L2A:
        out_DF = out_DF.loc[~out_DF["SCL_class"].isin(scl_2_filterout)]
    
    # Append sensing product_date of S2 image
    # YYYY, MM, DD = 2020, 4, 14  # debug
    out_DF["date"] = product_date
    if is_L2A: stat_DF["date"] = product_date

    return out_DF, stat_DF


def S2bandstack2table(in_file: str,
                      in_file_scl: str, 
                      in_file_polys: str,
                      buffer: float, 
                      id_column: str,
                      product_date = str,
                      filter_clouds: bool = True,
                      is_sentinel: bool = True,
                      **kwargs
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    For a multiband (stacked .tiff) S2 scene: Extract pixel values for each
    polygon ID of the provided polygons

    Parameters
    ----------
    in_file : str
        Path to the multiband stacked .tiff file.
    in_file_scl : str
        Path to the SCL file.
    in_file_polys : str
        Path to the shapefile containing the Polygons (can be other filetype as well).
    buffer : float
        Value to buffer the individual field polygons by (negative for inward buffer).
    id_column : str
        COlumn name that contains the ID for each indiv. field polygon.
    product_date : TYPE, optional
        Ingestion product_date of the S2 scene. The default is str.
    filter_clouds : bool, optional
        Should the SCL scene be used for masking cloudy pixels on polygon level. 
        The default is True.
    is_sentinel : bool, optional
        If satellite data other than is_sentinel-2 is used, set to false to ignore
        all SCL logic. The default is True.
    **kwargs : TYPE
        scl_2_filterout = kwargs.get('scl_classes', [0, 1, 3, 8, 9, 10, 11])
        nodata_refl = kwargs.get('nodata_refl', 64537)
        nodata_scl = kwargs.get('nodata_scl', 254).

    Returns
    -------
    out_DF : TYPE
        Extracted Pixel values including X & Y Coordinates, EPSG code, ingestionproduct_date, 
        Polygon_ID, SCL classes plus the reflectance of each 10 S2 bands as int
    stat_DF : TYPE
        SCL statistic for each S2 ingestionproduct_date and polygon ID.

    '''

    # open stacked .tiff file
    bandstack = rio.open(in_file)
    
    # get the file with SCL (scene classification)
    if is_sentinel: scl_filepath = in_file_scl

    # ==========================     Check kwargs   ==========================
    # check for user-defined SCL filtration
    # default is: clouds, snow and ice, cloud shadow and nodata classes
    if not is_sentinel: filter_clouds = False  # cloud filtering does not work for L1C level
    if filter_clouds:
        scl_2_filterout = kwargs.get('scl_classes', [0, 1, 3, 8, 9, 10, 11])

    # check for user-defined nodata values for reflectance and SCL
    # nodata value for is_sentinel-2 reflectance
    # TODO: check if 64537 is a valid nodata value for reflectance values
    nodata_refl = kwargs.get('nodata_refl', 64537)
    # nodata for scene classification
    nodata_scl = kwargs.get('nodata_scl', 254)

    # read in bandlist
    bandlist = list(bandstack.descriptions)

    # read the shapefile that contains the polyons
    bbox_parcels = gpd.read_file(in_file_polys)

    # calculate the buffer
    bbox_parcels_buffered = buffer_fieldpolygons(in_gdf = bbox_parcels, 
                                                 buffer = buffer)

    # ========================== check CRS ==========================

    # get CRS of satellite data by taking the 1st .JP2 file
    sat_crs = bandstack.crs
    # convert buffered bbox to S2 CRS
    bbox_buffered_s2_crs = bbox_parcels_buffered.to_crs(sat_crs)

    # assert that the crs of the buffered polygons is the same as of the inputs
    assert bbox_parcels_buffered.geometry.crs == bbox_parcels.geometry.crs, \
            'CRS mismatch between buffer and input'

    # ========================== loop over IDs ==========================
    full_DF = []
    if is_sentinel: parcel_statistics = []  # SCL is available for is_sentinel level, only
   
    for feature in bbox_buffered_s2_crs.iterrows():

        print(f"Calculating ID no. {feature[1][id_column]}")
        
        if feature[1][id_column] == '30430':
            print('a')

    # ========================== Loop over bands! ==========================
        flat_band_rflt_per_ID = []
        shape = gpd.GeoDataFrame(feature[1]).transpose()

        try:
            out_band, out_transform = rio.mask.mask(bandstack,
                                                    shape["geometry"],
                                                    crop = True, 
                                                    all_touched = True, # IMPORTANT!
                                                    nodata = nodata_refl
                                                    )
        except Exception as e:
            print(f'Couldnot clip feature {feature[1][id_column]}: {e}')
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
        
        # ======== Get SCL class per pixel ==========
        # works only for is_sentinel processing level
        if is_sentinel:
            with rio.open(scl_filepath) as src:
                shape = gpd.GeoDataFrame(feature[1]).transpose()
                
                out_scl, out_transform = rio.mask.mask(src,
                                                        shape["geometry"],
                                                        crop=True, 
                                                        all_touched=True,
                                                        nodata=nodata_scl
                                                        )

                # compute share of SCL classes for the current polygon
                stats = compute_parcel_stat(in_array=out_scl,
                                            nodata_value=nodata_scl)

                stats[id_column] = feature[1][id_column]
                parcel_statistics.append(stats)

                # also flatten the SCL values to store them p er pixel
                flat_scl_class = out_scl.flatten(order='F')
            
            # add SCL to ID_df out of rio scope
                per_ID_df["SCL_class"] = flat_scl_class

        # add field ID
        per_ID_df[id_column] = feature[1][id_column]
        # add CRS in form of EPSG code
        per_ID_df["CRS_epsg"] = bbox_buffered_s2_crs.crs.to_epsg()
        # append to full DF holding all field_IDs
        full_DF.append(per_ID_df)
    
    # coerce parcel statistic to Dataframe (is_sentinel-2 only)
    if is_sentinel:

        stat_DF = pd.DataFrame(parcel_statistics)
        stat_DF[id_column] = stat_DF[id_column]

    else:
        
        stat_DF = None

    # convert full_DF from list to dataframe
    out_DF = pd.concat(full_DF)
    # drop no-data pixels (all reflectance values equal zero)
    out_DF = out_DF.loc[(out_DF[bandlist] != nodata_refl).all(axis=1)]

    # ========= filter out clouds based on SCL ================
    # works for is_sentinel-2 data only
    if is_sentinel:
        out_DF = out_DF.loc[out_DF["SCL_class"] != nodata_scl]
    if filter_clouds and is_sentinel:
        out_DF = out_DF.loc[~out_DF["SCL_class"].isin(scl_2_filterout)]
    
    # Append sensing product_date of S2 image
    out_DF["date"] = product_date
    if is_sentinel: stat_DF["date"] = product_date

    return out_DF, stat_DF
