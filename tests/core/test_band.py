"""
Tests for `~agrisatpy.core.Band`
"""

import cv2
import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import zarr

from shapely.geometry import Polygon

from agrisatpy.core.band import Band
from agrisatpy.core.band import GeoInfo
from agrisatpy.core.band import WavelengthInfo

@pytest.fixture
def get_test_band(get_bandstack, get_polygons):
    """Fixture returning Band object from rasterio"""
    def _get_test_band():
        fpath_raster = get_bandstack()
        vector_features = get_polygons()
    
        band = Band.from_rasterio(
            fpath_raster=fpath_raster,
            band_idx=1,
            band_name_dst='B02',
            vector_features=vector_features,
            full_bounding_box_only=False,
            nodata=0
        )
        return band
    return _get_test_band

def test_base_constructors():
    """
    test base constructor calls
    """

    epsg = 32633
    ulx = 300000
    uly = 5100000
    pixres_x, pixres_y = 10, -10
    
    geo_info = GeoInfo(
        epsg=epsg,
        ulx=ulx,
        uly=uly,
        pixres_x=pixres_x,
        pixres_y=pixres_y
    )
    assert isinstance(geo_info.as_affine(), rio.Affine), 'wrong Affine type'

    # invalid EPSG code
    epsg = 0
    with pytest.raises(ValueError):
        geo_info = GeoInfo(
        epsg=epsg,
        ulx=ulx,
        uly=uly,
        pixres_x=pixres_x,
        pixres_y=pixres_y
    )

    band_name = 'test'
    values = np.zeros(shape=(2,4))

    band = Band(band_name=band_name, values=values, geo_info=geo_info)
    assert type(band.bounds) == Polygon, 'band bounds must be a Polygon'
    assert not band.has_alias, 'when color name is not set, the band has no alias'
    assert band.band_name == band_name, 'wrong name for band'

    assert band.values[0,0] == 0., 'wrong value for band data'

    assert band.meta['height'] == band.nrows, 'wrong raster height in meta'
    assert band.meta['width'] == band.ncols, 'wrong raster width in meta'
    assert band.is_ndarray, 'must be of type ndarray'
    assert band.crs.is_epsg_code, 'EPSG code not valid' 

    zarr_values = zarr.zeros((10,10), chunks=(5,5), dtype='float32')
    band = Band(band_name=band_name, values=zarr_values, geo_info=geo_info)
    assert band.is_zarr

def test_band_from_rasterio(get_test_band, get_bandstack):
    """
    Tests instance and class methods of the `Band` class
    """
    band = get_test_band()
    
    assert band.geo_info.epsg == 32632, 'wrong EPSG code'
    assert band.band_name == 'B02', 'wrong band name'
    assert band.is_masked_array, 'array has not been masked'
    assert band.alias is None, 'band alias was not set'
    assert set(band.coordinates.keys()) == {'x', 'y'}, 'coordinates not set correctly'
    assert band.nrows == band.values.shape[0], 'number of rows does not match data array'
    assert band.ncols == band.values.shape[1], 'number of rows does not match data array'
    assert band.coordinates['x'].shape[0] == band.ncols, \
        'number of x coordinates does not match number of columns'
    assert band.coordinates['y'].shape[0] == band.nrows, \
        'number of y coordinates does not match number of rows'
    assert band.values.min() == 19, 'wrong minimum returned from data array'
    assert band.values.max() == 9504, 'wrong maximum returned from data array'
    assert band.geo_info.ulx == 475420.0, 'wrong upper left x coordinate'
    assert band.geo_info.uly == 5256840.0, 'wrong upper left y coordinate'

    band_bounds_mask = band.bounds
    assert band_bounds_mask.type == 'Polygon'
    assert band_bounds_mask.exterior.bounds[0] == band.geo_info.ulx, \
        'upper left x coordinate does not match'
    assert band_bounds_mask.exterior.bounds[3] == band.geo_info.uly, \
        'upper left y coordinate does not match'

    assert not band.values.mask.all(), 'not all pixels should be masked'

    fig = band.plot(colorbar_label='Surface Reflectance')
    assert fig is not None, 'no figure returned'

    # try some cases that must fail
    # reading from non-existing file
    with pytest.raises(rio.errors.RasterioIOError):
        Band.from_rasterio(
            fpath_raster='not_existing_file.tif'
        )
    # existing file but reading wrong band index
    fpath_raster = get_bandstack()
    with pytest.raises(IndexError):
        Band.from_rasterio(
            fpath_raster=fpath_raster,
            band_idx=22,
        )

def test_reprojection(datadir, get_test_band):
    """reprojection into another CRS"""
    # define test data sources
    
    band = get_test_band()
    # reproject to Swiss Coordinate System (EPSG:2056)
    reprojected = band.reproject(target_crs=2056)

    assert reprojected.crs == 2056, 'wrong EPSG after reprojection'
    assert reprojected.is_masked_array, 'mask must not be lost after reprojection'
    assert np.round(reprojected.geo_info.ulx) == 2693130.0, 'wrong upper left x coordinate'
    assert np.round(reprojected.geo_info.uly) == 1257861.0, 'wrong upper left y coordinate'
    fpath_out = datadir.joinpath('reprojected.jp2')
    reprojected.to_rasterio(fpath_out)
    with rio.open(fpath_out, 'r') as src:
        meta = src.meta
    # make sure everything was saved correctly to file
    assert meta['crs'] == reprojected.crs
    assert meta['height'] == reprojected.nrows
    assert meta['width'] == reprojected.ncols
    assert meta['transform'] == reprojected.transform

def test_bandstatistics(get_test_band):

    band = get_test_band()
    # get band statistics
    stats = band.reduce(method=['mean', 'min', 'max'])
    mean_stats = band.reduce(method='mean')
    assert mean_stats['mean'] == stats['mean'], 'miss-match of metrics'
    assert stats['min'] == band.values.min(), 'minimum not calculated correctly'
    assert stats['max'] == band.values.max(), 'maximum not calculated correctly'

    # convert to GeoDataFrame
    gdf = band.to_dataframe()
    assert (gdf.geometry.type == 'Point').all(), 'wrong geometry type'
    assert set(gdf.columns) == {'geometry', 'B02'}, 'wrong column labels'
    assert gdf.shape[0] == 29674, 'wrong number of pixels converted'
    assert gdf.B02.max() == stats['max'], 'band statistics not the same after conversion'
    assert gdf.B02.min() == stats['min'], 'band statistics not the same after conversion'
    assert gdf.B02.mean() == stats['mean'], 'band statistics not the same after conversion'

def test_to_xarray(get_test_band):
    band = get_test_band()
    # convert to xarray
    xarr = band.to_xarray()
    assert xarr.x.values[0] == band.geo_info.ulx + 0.5*band.geo_info.pixres_x, \
        'pixel coordinate not shifted to center of pixel in xarray'
    assert xarr.y.values[0] == band.geo_info.uly + 0.5*band.geo_info.pixres_y, \
        'pixel coordinate not shifted to center of pixel in xarray'
    assert (xarr.values == band.values.astype(float)).all(), \
        'array values changed after conversion to xarray'
    assert np.count_nonzero(~np.isnan(xarr.values)) == band.values.compressed().shape[0], \
        'masked values were not set to nan correctly'
    assert xarr.shape[1] == band.nrows and xarr.shape[2] == band.ncols, \
        'wrong number of rows and columns in xarray'

def test_resampling(get_test_band):
    band = get_test_band()
    # resample to 20m spatial resolution using bi-cubic interpolation
    resampled = band.resample(
        target_resolution=20,
        interpolation_method=cv2.INTER_CUBIC
    )
    assert resampled.geo_info.pixres_x == 20, 'wrong pixel size after resampling'
    assert resampled.geo_info.pixres_y == -20, 'wrong pixel size after resampling'
    assert resampled.geo_info != band.geo_info, 'geo info must not be the same'
    assert resampled.ncols < band.ncols, 'spatial resolution should decrease'
    assert resampled.nrows < band.ncols, 'spatial resolution should decrease'
    assert resampled.is_masked_array, 'mask should be preserved'

    # resample to 5m inplace
    old_shape = (band.nrows, band.ncols)
    band.resample(
        target_resolution=5,
        inplace=True
    )
    assert band.nrows == 588, 'wrong number of rows after resampling'
    assert band.ncols == 442, 'wrong number of columns after resampling'
    assert band.is_masked_array, 'mask should be preserved'

    # resample back to 10m and align to old shape
    band.resample(
        target_resolution=10,
        target_shape=old_shape,
        interpolation_method=cv2.INTER_CUBIC,
        inplace=True
    )

    assert (band.nrows, band.ncols) == old_shape, 'resampling to target shape did not work'

def test_masking(datadir, get_test_band, get_bandstack, get_points3):
    """masking of band data"""
    band = get_test_band()
    mask = np.ndarray(band.values.shape, dtype='bool')
    mask.fill(True)

    # try without inplace
    masked_band = band.mask(mask=mask, inplace=False)
    assert isinstance(masked_band, Band), 'wrong return type'
    assert (masked_band.values.mask == mask).all(), 'mask not applied correctly'
    assert (masked_band.values.data == band.values.data).all(), 'data not preserved correctly'

    band.mask(mask=mask, inplace=True)
    assert band.values.mask.all(), 'not all pixels masked'

    # test scaling -> nothing should happen at this stage
    values_before_scaling = band.values
    band.scale_data()
    assert (values_before_scaling.data == band.values.data).all(), 'scaling must not have an effect'

    # read data with AOI outside of the raster bounds -> should raise a ValueError
    fpath_raster = get_bandstack()
    vector_features_2 = get_points3()
    with pytest.raises(ValueError):
        band = Band.from_rasterio(
            fpath_raster=fpath_raster,
            band_idx=1,
            band_name_dst='B02',
            vector_features=vector_features_2,
            full_bounding_box_only=False
        )

    # write band data to disk
    fpath_out = datadir.joinpath('test.jp2')
    band.to_rasterio(fpath_raster=fpath_out)

    assert fpath_out.exists(), 'output dataset not written'
    band_read_again = Band.from_rasterio(fpath_out)
    assert (band_read_again.values == band.values.data).all(), \
        'band data not the same after writing'

def test_read_pixels(get_bandstack, get_test_band, get_polygons, get_points3):
    # read single pixels from raster dataset
    fpath_raster = get_bandstack()
    vector_features = get_polygons()

    pixels = Band.read_pixels(
        fpath_raster=fpath_raster,
        vector_features=vector_features
    )
    assert 'B1' in pixels.columns, 'extracted band data not found'
    gdf = gpd.read_file(vector_features)
    assert pixels.shape[0] == gdf.shape[0], 'not all geometries extracted'
    assert pixels.geometry.type.unique().shape[0] == 1, 'there are too many different geometry types'
    assert pixels.geometry.type.unique()[0] == 'Point', 'wrong geometry type'

    # compare against results from instance method
    band = get_test_band()
    pixels_inst = band.get_pixels(vector_features)
    assert (pixels.geometry == pixels_inst.geometry).all(), \
        'pixel geometry must be always the same'
    assert band.band_name in pixels_inst.columns, 'extracted band data not found'

    # try features outside of the extent of the raster
    vector_features_2 = get_points3()
    pixels = Band.read_pixels(
        fpath_raster=fpath_raster,
        vector_features=vector_features_2
    )
    assert (pixels.B1 == 0).all(), 'nodata not set properly to features outside of raster extent'

    # read with full bounding box (no masking just spatial sub-setting)
    band = Band.from_rasterio(
        fpath_raster=fpath_raster,
        band_idx=1,
        band_name_dst='B02',
        vector_features=vector_features,
        full_bounding_box_only=True
    )

    assert not band.is_masked_array, 'data should not be masked'
    assert band.is_ndarray, 'band data should be ndarray'

    mask = np.ndarray(band.values.shape, dtype='bool')
    mask.fill(True)
    mask[100:120,100:200] = False
    band.mask(mask=mask, inplace=True)
    assert band.is_masked_array, 'band must now be a masked array'
    assert not band.values.mask.all(), 'not all pixel should be masked'
    assert band.values.mask.any(), 'some pixels should be masked'

    resampled = band.resample(target_resolution=5)
    assert band.geo_info.pixres_x == 10, 'resolution of original band should not change'
    assert band.geo_info.pixres_y == -10, 'resolution of original band should not change'
    assert resampled.geo_info.pixres_x == 5, 'wrong x pixel resolution'
    assert resampled.geo_info.pixres_y == -5, 'wrong y pixel resolution'
    assert resampled.bounds == band.bounds, 'band bounds should be the same after resampling'

def test_from_vector(get_polygons):
    vector_features = get_polygons()

    # read data from vector source
    epsg = 32632
    gdf = gpd.read_file(vector_features)
    ulx = gdf.geometry.total_bounds[0]
    uly = gdf.geometry.total_bounds[-1]
    pixres_x, pixres_y = 10, -10
    
    geo_info = GeoInfo(
        epsg=epsg,
        ulx=ulx,
        uly=uly,
        pixres_x=pixres_x,
        pixres_y=pixres_y
    )
    band = Band.from_vector(
        vector_features=vector_features,
        geo_info=geo_info,
        band_name_src='GIS_ID',
        band_name_dst='gis_id',
    )
    bounds = band.bounds

    assert band.band_name == 'gis_id', 'wrong band name inserted'
    assert band.values.dtype == 'float32', 'wrong data type for values'
    assert band.geo_info.pixres_x == 10, 'wrong pixel size in x direction'
    assert band.geo_info.pixres_y == -10, 'wrong pixel size in y direction'
    assert band.geo_info.ulx == ulx, 'wrong ulx coordinate'
    assert band.geo_info.uly == uly, 'wrong uly coordinate'
    assert band.geo_info.epsg == 32632, 'wrong EPSG code'

    # with custom datatype
    band = Band.from_vector(
        vector_features=vector_features,
        geo_info=geo_info,
        band_name_src='GIS_ID',
        band_name_dst='gis_id',
        dtype_src='uint16',
        snap_bounds=bounds
    )
    assert band.values.dtype == 'uint16', 'wrong data type'

    # test with point features
    point_gdf = gpd.read_file(vector_features)
    point_gdf.geometry = point_gdf.geometry.apply(lambda x: x.centroid)

    band_from_points = Band.from_vector(
        vector_features=point_gdf,
        geo_info=geo_info,
        band_name_src='GIS_ID',
        band_name_dst='gis_id',
        dtype_src='uint32'
    )
    assert band_from_points.values.dtype == 'uint32', 'wrong data type'
    assert band_from_points.reduce(method='max')['max'] == \
        point_gdf.GIS_ID.values.astype(int).max(), 'miss-match in band statistics'
