"""
Tests for the RasterCollection class
"""

import datetime
import pytest

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from agrisatpy.core.band import GeoInfo
from agrisatpy.core.band import Band
from agrisatpy.core.raster import SceneProperties
from agrisatpy.core.raster import RasterCollection
from agrisatpy.utils.exceptions import BandNotFoundError


def test_ndarray_constructor():
    """
    basic test with ``np.ndarray`` backend in raster collection
    """

    handler = RasterCollection()
    assert handler.empty, 'RasterCollection is not empty'
    assert handler.scene_properties.acquisition_time == datetime.datetime(2999,1,1)
    assert len(handler) == 0, 'there should not be any items so far'
    assert handler.is_bandstack() is None, 'cannot check for bandstack without bands'

    # add band to empty handler
    epsg = 32633
    ulx, uly = 300000, 5100000
    pixres_x, pixres_y = 10, -10
    geo_info = GeoInfo(epsg=epsg,ulx=ulx,uly=uly,pixres_x=pixres_x,pixres_y=pixres_y)
    zeros = np.zeros((100,100))
    band_name_zeros = 'zeros'
    handler.add_band(Band, band_name=band_name_zeros, values=zeros, geo_info=geo_info)
    assert not handler.empty, 'handler is still empty'
    assert 'zeros' in handler.band_names, 'band not found'

    # test with band instance from np.ndarray
    band_name = 'random'
    color_name = 'blue'
    values = np.random.random(size=(100,120))

    handler = RasterCollection(
        band_constructor=Band,
        band_name=band_name,
        values=values,
        band_alias=color_name,
        geo_info=geo_info
    )

    assert len(handler) == 1, 'wrong number of bands in collection'
    assert isinstance(handler[band_name], Band), 'not a proper band in collection'
    assert handler[band_name].band_name == band_name, 'incorrect band name'
    assert handler.band_names == [band_name], 'wrong number of band names'
    assert handler.band_aliases != [''], 'band aliases were not set'

    # make sure the collection is protected properly
    with pytest.raises(TypeError):
        handler.collection = 'ttt'

    with pytest.raises(ValueError):
        handler.collection = dict()

    # add a second band
    zeros = np.zeros_like(values)
    band_name_zeros = 'zeros'
    handler.add_band(Band, band_name=band_name_zeros, values=zeros, geo_info=geo_info)

    # incorrect constructor call (duplicate band name)
    with pytest.raises(KeyError):
        handler.add_band(
            Band,
            band_name=band_name_zeros,
            values=zeros,
            geo_info=geo_info
        )

    # mask the second band based on the first one
    masked = handler.mask(mask='random', mask_values=[0.15988288, 0.38599023])
    assert masked.band_names == handler.band_names, 'band names not passed on correctly'
    assert masked['zeros'].is_masked_array, 'array should have mask now'
    assert masked.scene_properties.acquisition_time == \
        handler.scene_properties.acquisition_time, 'scene properties got lost'

    # drop one of the bands again
    handler.drop_band('random')
    assert 'random' not in handler.band_names, 'band name still exists after dropping it'
    with pytest.raises(KeyError):
        handler['random']

    # drop non-existing band
    with pytest.raises(BandNotFoundError):
        handler.drop_band('test')

def test_rasterio_constructor_single_band(get_bandstack):
    """
    RasterCollection from rasterio dataset (single band)
    """

    handler = RasterCollection()

    # add a band from rasterio
    fpath_raster = get_bandstack()
    band_idx = 1
    handler.add_band(Band.from_rasterio, fpath_raster=fpath_raster, band_idx=band_idx)

    assert set(handler.band_names) == {'B1'}, \
        'band names not set properly in collection'
    assert len(handler.band_names) == 1, 'wrong number of bands in collection'

    # bands should fulfill the band stack criterion
    assert handler.is_bandstack(), 'single band not fulfill bandstack criterion'
    assert handler['B1'].geo_info.epsg == 32632, 'wrong EPSG code inserted'
    assert handler['B1'].geo_info.pixres_x == 10, 'wrong pixel size in x direction'
    assert handler['B1'].geo_info.pixres_y == -10, 'wrong pixel size in y direction'

def test_scale():
    """scaling of raster band"""
    handler = RasterCollection()

    # add band to empty handler
    epsg = 32633
    ulx, uly = 300000, 5100000
    pixres_x, pixres_y = 10, -10
    geo_info = GeoInfo(epsg=epsg,ulx=ulx,uly=uly,pixres_x=pixres_x,pixres_y=pixres_y)
    ones = np.ones((100,100), dtype='uint16')
    band_name_ones = 'ones'
    scale = 100
    offset = 2
    handler.add_band(
        Band,
        band_name=band_name_ones,
        values=ones,
        geo_info=geo_info,
        scale=scale,
        offset=offset
    )

    before_scaling = handler.get_values()
    handler.scale(inplace=True)
    after_scaling = handler.get_values()
    assert (after_scaling == (scale * (before_scaling + offset))).all(), 'values not scaled correctly'
    assert (after_scaling == 300).all(), 'wrong value after scaling'

def test_rasterio_constructor_multi_band(get_bandstack):    
    """
    RasterCollection from rasterio dataset (multiple bands)
    """
    # read multi-band geoTiff into new handler
    fpath_raster = get_bandstack()
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster
    )

    assert gTiff_collection.band_names == \
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'], \
        'wrong list of band names in collection'
    assert gTiff_collection.is_bandstack(), 'collection must be bandstacked'
    gTiff_collection['B2'].crs == 32632, 'wrong CRS'

    # read multi-band geoTiff into new handler with custom destination names
    colors = ['blue', 'green', 'red', 'red_edge_1', 'red_edge_2', 'red_edge_3', \
              'nir_1', 'nir_2', 'swir_1', 'swir_2']
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster,
        band_names_dst=colors
    )
    assert gTiff_collection.band_names == colors, 'band names not set properly'
    gTiff_collection.calc_si('NDVI', inplace=True)
    assert 'NDVI' in gTiff_collection.band_names, 'SI not added to collection'
    assert gTiff_collection['NDVI'].ncols == gTiff_collection['red'].ncols, \
        'wrong number of columns in SI'
    assert gTiff_collection['NDVI'].nrows == gTiff_collection['red'].nrows, \
        'wrong number of rows in SI'
    assert -1 < np.nanmin(gTiff_collection['NDVI'].values) < 0, 'minimum NDVI should be <-1 and <0'
    assert 0 < np.nanmax(gTiff_collection['NDVI'].values) < 1, 'maximum NDVI should be <0 and <1'

    gdf = gTiff_collection.to_dataframe(['NDVI', 'swir_2'])
    assert set(['NDVI', 'swir_2']).issubset(gdf.columns), 'bands not added as GeoDataFrame columns'
    assert (gdf.geometry.type == 'Point').all(), 'all geometries must be Points'
    assert gdf.shape[0] == gTiff_collection['blue'].nrows * gTiff_collection['blue'].ncols, \
        'wrong number of pixels converted to GeoDataFrame'

    # with band aliases
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster,
        band_aliases=colors
    )
    assert gTiff_collection.has_band_aliases, 'band aliases must exist'

def test_reprojection(get_bandstack):
    """Reprojecting data in raster collection into a different CRS"""
    # read multi-band geoTiff into new handler with custom destination names
    colors = ['blue', 'green', 'red', 'red_edge_1', 'red_edge_2', 'red_edge_3', \
              'nir_1', 'nir_2', 'swir_1', 'swir_2']
    fpath_raster = get_bandstack()
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster,
        band_names_dst=colors
    )
    # try reprojection of raster bands to geographic coordinates
    reprojected = gTiff_collection.reproject(target_crs=4326)
    assert reprojected.band_names == gTiff_collection.band_names, 'band names not the same'

    # reproject inplace
    gTiff_collection.reproject(target_crs=4326, inplace=True)
    assert gTiff_collection.get_band('blue').crs == reprojected.get_band('blue').crs, \
        'CRS not updated properly'

    # plot RGB
    fig_rgb = reprojected.plot_multiple_bands(band_selection=['red', 'green', 'blue'])
    assert isinstance(fig_rgb, plt.Figure), 'not a matplotlib figure'

def test_resampling(datadir,get_bandstack):
    """Resampling into a different pixel size"""
    # read multi-band geoTiff into new handler with custom destination names
    colors = ['blue', 'green', 'red', 'red_edge_1', 'red_edge_2', 'red_edge_3', \
              'nir_1', 'nir_2', 'swir_1', 'swir_2']
    fpath_raster = get_bandstack()
    # resample all bands to 5m
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster,
        band_aliases=colors
    )
    resampled = gTiff_collection.resample(
        target_resolution=5
    )
    assert resampled['green'].geo_info.pixres_x == 5, 'resolution was not changed'
    assert resampled['green'].ncols == 2206, 'wrong number of columns'
    assert resampled['green'].nrows == 2200, 'wrong number of rows'

    fpath_out = datadir.joinpath('test.jp2')
    resampled.to_rasterio(fpath_out)
    assert fpath_out.exists(), 'output-file not created'
