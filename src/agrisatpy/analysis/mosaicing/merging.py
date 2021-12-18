'''
Module for merging raster datasets.
'''


from pathlib import Path
from typing import NamedTuple
from typing import List
from typing import Optional
from typing import Union
from collections import namedtuple
from rasterio.merge import merge
from rasterio.crs import CRS

from agrisatpy.io import SatDataHandler
from agrisatpy.utils.reprojection import reproject_raster_dataset


def _get_CRS_and_bounds(
        in_file: Path,
        **kwargs
    ) -> NamedTuple:
    """
    Returns the CRS and bounding box of a raster dataset
    """

    ds = SatDataHandler()
    ds.read_from_bandstack(
        fname_bandstack=in_file,
        **kwargs
    )
    Geo_Info = namedtuple('Geo_Info', 'CRS bounds meta')

    # take CRS and bounds from the first band since it must be
    # the same for all bands
    band_name = ds.get_bandnames()[0]
    crs = ds.get_epsg(band_name)
    bounds = ds.get_bounds(band_name)
    meta = ds.get_meta(band_name)

    return Geo_Info(crs, bounds, meta)


def merge_datasets(
        datasets: List[Path],
        out_file: Path,
        target_crs: Optional[Union[int,CRS]] = None,
        **kwargs
    ):
    """
    Merges a list of raster datasets using the ``rasterio.merge`` module. The
    function can handle datasets in different coordinate systems by resampling
    the data into a common spatial reference system either provided in the function
    call or infered from the first dataset in the list.

    ATTENTION:
        All datasets must have the same number of bands and data type!

    :param datasets:
        list of datasets (as path-like objects or opened raster datasets)
        to merge into a single raster
    :param out_file:
        name of the resulting raster dataset
    :param target_crs:
        optional target spatial coordinate reference system in which the output
        product shall be generated. Must be passed as integer EPSG code or CRS
        instance.
    :param kwargs:
        kwargs to pass to ``rasterio.warp.reproject``
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

    # coordinate systems are not the same -> re-projection of raster datasets
    if len(set(crs_list)) > 1:
        # check if a target CRS is provided, otherwise use the CRS of the first
        # dataset item
        if target_crs is None:
            # use CRS from first dataset
            pass
        else:
            pass
    # all datasets have one coordinate system, check if it is the desired one
    else:
        if target_crs is not None:
            if crs_list[0] != target_crs:
                # re-projection into target CRS required
                pass

    # use rasterio merge to get a new raster dataset
    try:
        merge(
            datasets=datasets,
            dst_path=out_file
        )
    except Exception as e:
        raise Exception(f'Could not merge datasets: {e}') 
