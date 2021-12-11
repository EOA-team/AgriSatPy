'''
Created on Dec 10, 2021

@author: graflu
'''

from pathlib import Path
from typing import NamedTuple
from typing import List
from collections import namedtuple
import rasterio as rio
from rasterio.merge import merge

from agrisatpy.io import Sat_Data_Reader


def _get_CRS_and_bounds(
        in_file: Path,
        **kwargs
    ) -> NamedTuple:
    """
    Returns the CRS and bounding box of a raster dataset
    """

    ds = Sat_Data_Reader()
    ds.read_from_bandstack(
        fname_bandstack=in_file,
        **kwargs
    )
    Geo_Info = namedtuple('Geo_Info', 'CRS bounds meta')

    # take CRS and bounds from the first band since it must be
    # the same for all bands
    band_name = ds.get_bandnames()[0]
    crs = ds.get_band_epsg(band_name)
    bounds = ds.get_band_bounds(band_name)
    meta = ds.get_meta(band_name)

    return Geo_Info(crs, bounds, meta)


def merge_datasets(
        datasets: List[Path],
        out_file: Path
    ):
    """
    Merges a list of raster datasets using the ``rasterio.merge`` module. The
    function can handle datasets in different coordinate systems by resampling
    the data into a common spatial reference system (todo)

    IMPORTANT: All datasets must have the same number of bands and data type!

    :param datasets:
        list of datasets (as path-like objects or opened raster datasets)
        to merge into a single raster
    :param out_file:
        name of the resulting raster dataset
    """

    # check the CRS and bounds of the datasets first
    crs_list = []
    bounds_list = []
    meta_list = []

    for dataset in datasets:
        geo_info = _get_CRS_and_bounds(in_file=dataset)
        crs_list.append(geo_info.CRS)
        bounds_list.append(geo_info.bounds)
        meta_list.append(geo_info.meta)

    # coordinate systems are not the same 
    if len(set(crs_list)) > 1:
        # TODO: implement this case -> requires re-projection of the data
        pass

    # use rasterio merge to get a new raster dataset
    try:
        merge(
            datasets=datasets,
            dst_path=out_file
        )
    except Exception as e:
        raise Exception(f'Could not merge datasets: {e}') 
