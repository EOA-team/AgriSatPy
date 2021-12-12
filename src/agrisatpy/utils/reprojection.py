'''
Functions for reprojecting vector and raster data from one spatial
coordinate reference system into another one.
'''

import numpy as np
import rasterio as rio
import geopandas as gpd

from rasterio import Affine
from shapely.geometry import box
from pathlib import Path
from typing import Union
from typing import Tuple
from geopandas import GeoDataFrame


def check_aoi_geoms(
        in_file_aoi: Path,
        fname_sat: Path,
        full_bounding_box_only: bool
    ) -> GeoDataFrame:
    """
    Checks the provided AOI file. If necessary it reprojects
    the vector data in the reference system of the satellite raster
    data. If the full bounding box shall be used (e.g., the hull
    encompassing all provided vector geometries) it only returns
    this geometry.

    :param in_file_aoi:
        vector file (e.g., ESRI shapefile or geojson) defining geometry/ies
        (polygon(s)) for which to extract the Sentinel-2 data. Can contain
        one to many features.
    :param fname_sat:
        raster file with satellite data
    :param full_bounding_box_only:
        if set to False, will only extract the data for those geometry/ies
        defined in in_file_aoi. If set to False, returns the data for the
        full extent (hull) of all features (geometries) in in_file_aoi.
    :return:
        GeoDataFrame with one up to many vector geometries
    """
    
    # check for vector file defining AOI
    # read AOI into a geodataframe
    gdf_aoi = gpd.read_file(in_file_aoi)
    # check if the spatial reference systems match
    sat_crs = rio.open(fname_sat).crs
    # reproject vector data if necessary
    if gdf_aoi.crs != sat_crs:
        gdf_aoi.to_crs(sat_crs, inplace=True)

    # if the the entire bounding box shall be extracted
    # we need the hull encompassing all geometries in gdf_aoi
    if full_bounding_box_only:
        bbox = box(*gdf_aoi.total_bounds)
        gdf_aoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bbox))

    return gdf_aoi


def reproject_raster_dataset(
        raster: Union[Path, np.ndarray],
        **kwargs
    ) -> Tuple[Union[Path,np.ndarray], Affine]:
    """
    Re-projects a raster dataset into another spatial coordinate reference
    system by calling ``rasterio.warp.reproject``.

    :param raster:
        either a file-path to a raster dataset or a numpy array
        containing band data to reproject
    :param kwargs:
        kwargs required by ``rasterio.warp.reproject``. See rasterio's docs
        for more information.
    """

    try:
        dst, transform = rio.warp.reproject(
            source=raster,
            **kwargs
        )
    except Exception as e:
        raise Exception from e

    return dst, transform

