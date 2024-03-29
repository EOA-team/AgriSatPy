'''
Module for merging raster datasets.
'''

import os
import geopandas as gpd
import uuid

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from rasterio.merge import merge
from rasterio.crs import CRS

import agrisatpy
from agrisatpy.config import get_settings
from agrisatpy.core.band import Band, GeoInfo
from agrisatpy.core.raster import RasterCollection
from agrisatpy.core.scene import SceneProperties

Settings = get_settings()

def _get_crs_and_attribs(
        in_file: Path,
        **kwargs
    ) -> Tuple[GeoInfo, List[Dict[str, Any]]]:
    """
    Returns the ``GeoInfo`` from a multi-band raster dataset

    :param in_file:
        raster datasets from which to extract the ``GeoInfo`` and
        attributes
    :param kwargs:
        optional keyword-arguments to pass to
        `~agrisatpy.core.raster.RasterCollection.from_multi_band_raster`
    :returns:
        ``GeoInfo`` and metadata attributes of the raster dataset
    """

    ds = RasterCollection.from_multi_band_raster(
        fpath_raster=in_file,
        **kwargs
    )
    geo_info = ds[ds.band_names[0]].geo_info
    attrs = [ds[x].get_attributes() for x in ds.band_names]
    return geo_info, attrs

def merge_datasets(
        datasets: List[Path],
        out_file: Optional[Path] = None,
        target_crs: Optional[Union[int,CRS]] = None,
        vector_features: Optional[Union[Path, gpd.GeoDataFrame]] = None,
        scene_properties: Optional[SceneProperties] = None,
        band_options: Optional[Dict[str,Any]] = None,
        sensor: Optional[str] = None,
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
    :param vector_features:
        optional vector features to clip the merged dataset to (full bounding box).
    :param scene_properties:
        optional scene properties to set to the resulting merged dataset
    :param band_options:
        optional sensor-specific band options to pass to the sensor's
        ``RasterCollection`` constructor
    :param sensor:
        if the data is from a sensor explicitly supported by AgriSatPy such as
        Sentinel-2 the raster data is loaded into a sensor-specific collection
    :param kwargs:
        kwargs to pass to ``rasterio.warp.reproject``
    :returns:
        depending on the kwargs passed either `None` (if output is written to file directly)
        or a `RasterCollection` instance if the operation takes place in memory
    """
    # check the CRS and attributes of the datasets first
    crs_list = []
    attrs_list = []
    for dataset in datasets:
        geo_info, attrs = _get_crs_and_attribs(in_file=dataset)
        crs_list.append(geo_info.epsg)
        attrs_list.append(attrs)
        
    if target_crs is None:
        # use CRS from first dataset
        target_crs = crs_list[0]
    # coordinate systems are not the same -> re-projection of raster datasets
    if len(set(crs_list)) > 1:
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
        res = merge(
            datasets=datasets,
            dst_path=out_file,
            dst_kwds=dst_kwds,
            **kwargs
        )
        if res is not None:
            out_ds, out_transform = res[0], res[1]
    except Exception as e:
        raise Exception(f'Could not merge datasets: {e}')

    # when out_file was provided the merged data is written to file directly
    if out_file is not None:
        return
    # otherwise, create new SatDataHandler instance from merged datasets
    # add scene properties if available
    raster = RasterCollection(scene_properties=scene_properties)
    n_bands = out_ds.shape[0]
    # take attributes of the first dataset
    attrs = attrs_list[0]
    geo_info = GeoInfo(
        epsg=target_crs,
        ulx=out_transform.c,
        uly=out_transform.f,
        pixres_x=out_transform.a,
        pixres_y=out_transform.e
    )
    for idx in range(n_bands):
        band_attrs = attrs[idx]
        nodata = band_attrs.get('nodatavals')
        if isinstance(nodata, tuple):
            nodata = nodata[0]
        is_tiled = band_attrs.get('is_tiled')
        scale = band_attrs.get('scales')
        if isinstance(scale, tuple):
            scale = scale[0]
        offset = band_attrs.get('offsets')
        if isinstance(offset, tuple):
            offset = offset[0]
        description = band_attrs.get('descriptions')
        if isinstance(description, tuple):
            description = description[0]
        unit = band_attrs.get('units')
        if isinstance(unit, tuple):
            unit = unit[0]
        raster.add_band(
            band_constructor=Band,
            band_name=f'B{idx+1}',
            values=out_ds[idx,:,:],
            geo_info=geo_info,
            is_tiled=is_tiled,
            scale=scale,
            offset=offset,
            band_alias=description,
            unit=unit
        )

    # clip raster collection if required to vector_features
    if vector_features is not None:
        tmp_dir = Settings.TEMP_WORKING_DIR
        fname_tmp = tmp_dir.joinpath(f'{uuid.uuid4()}.tif')
        raster.to_rasterio(fname_tmp)
        if sensor is None:
            expr = 'RasterCollection'
        else:
            expr = f'agrisatpy.core.sensors.{sensor.lower()}.{sensor[0].upper() + sensor[1::]}()'
        if band_options is None:
            band_options = {}
        raster = eval(f'''{expr}.from_multi_band_raster(
            fpath_raster=fname_tmp,
            vector_features=vector_features,
            full_bounding_box_only=False,
            **band_options
        )''')
        # set scene properties if available
        if scene_properties is not None:
            raster.scene_properties = scene_properties
        os.remove(fname_tmp)

    return raster
