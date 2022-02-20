'''
Module for merging raster datasets.
'''


from pathlib import Path
from shapely.geometry import Polygon
from typing import List, Optional, Tuple, Union
from rasterio.merge import merge
from rasterio.crs import CRS

from agrisatpy.core.band import GeoInfo
from agrisatpy.core.raster import RasterCollection
from agrisatpy.utils.reprojection import reproject_raster_dataset


def _get_CRS_and_bounds(
        in_file: Path,
        **kwargs
    ) -> Tuple[GeoInfo,Polygon]:
    """
    Returns the ``GeoInfo`` from a multi-band raster dataset

    :param in_file:
        raster datasets from which to extract the ``GeoInfo``
    :param kwargs:
        optional keyword-arguments to pass to
        `~agrisatpy.core.raster.RasterCollection.from_multi_band_raster`
    :returns:
        ``GeoInfo`` of the raster dataset
    """

    ds = RasterCollection.from_multi_band_raster(
        fpath_raster=in_file,
        **kwargs
    )
    geo_info = ds[ds.band_names[0]].geo_info
    bounds = ds[ds.band_names[0]].bounds
    return geo_info, bounds

def merge_datasets(
        datasets: List[Path],
        out_file: Optional[Path] = None,
        target_crs: Optional[Union[int,CRS]] = None,
        **kwargs
    ) -> Union[None, RasterCollection]:
    """
    Merges a list of raster datasets using the ``rasterio.merge`` module.
    
    The function can handle datasets in different coordinate systems by resampling
    the data into a common spatial reference system either provided in the function
    call or infered from the first dataset in the list.

    ATTENTION:
        All datasets must have the same number of bands and data type!

    :param datasets:
        list of datasets (as path-like objects or opened raster datasets)
        to merge into a single raster
    :param out_file:
        name of the resulting raster dataset (optional). If None (default)
        returns a new ``SatDataHandler`` instance otherwise writes the data
        to disk as new raster dataset.
    :param target_crs:
        optional target spatial coordinate reference system in which the output
        product shall be generated. Must be passed as integer EPSG code or CRS
        instance.
    :param kwargs:
        kwargs to pass to ``rasterio.warp.reproject``
    :returns:
        depending on the kwargs passed either `None` (if output is written to file directly)
        or a `RasterCollection` instance if the operation takes place in memory
    """

    # check the CRS and bounds of the datasets first
    crs_list = []
    # bounds_list = []
    # meta_list = []

    for dataset in datasets:
        geo_info, bounds = _get_CRS_and_bounds(in_file=dataset)
        crs_list.append(geo_info.epsg)
        # bounds_list.append(geo_info.bounds)
        # meta_list.append(geo_info.meta)

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
    dst_kwds = {
        'QUALITY': '100',
        'REVERSIBLE': 'YES'
    }
    try:
        out_file, _ = merge(
            datasets=datasets,
            dst_path=out_file,
            dst_kwds=dst_kwds,
            **kwargs
        )
    except Exception as e:
        raise Exception(f'Could not merge datasets: {e}')

    if out_file is not None:
        return

    # create new SatDataHandler instance from merged datasets
    return RasterCollection.from_multi_band_raster(fpath_raster=out_file)
