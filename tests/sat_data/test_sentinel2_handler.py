
import cv2
import pytest
import requests
import numpy as np

from rasterio.coords import BoundingBox
from pathlib import Path
from shapely.geometry import Polygon
from matplotlib.figure import Figure
from datetime import date

from agrisatpy.io.sentinel2 import Sentinel2Handler
from agrisatpy.utils.exceptions import BandNotFoundError, InputError


def test_read_from_bandstack_l1c():
    pass


def test_read_from_safe_l1c(get_s2_safe_l1c):
    """handling of Sentinel-2 data in L1C processing level from .SAFE archives"""

    in_dir = get_s2_safe_l1c()

     # read without AOI file
    reader = Sentinel2Handler()
    band_selection = ['B04', 'B05', 'B8A']
    reader.read_from_safe(
        in_dir=in_dir,
        band_selection=band_selection
    )

    # check if the object can be considered a band stack -> should not be the case
    assert not reader.check_is_bandstack(), 'data is labelled as band-stack but it should not'

    # check scene properties
    acquisition_time = reader.scene_properties.get('acquisition_time')
    assert acquisition_time.date() == date(2019,7,25), 'acquisition date is wrong'
    assert reader.scene_properties.get('sensor') == 'MSI', 'wrong sensor'
    assert reader.scene_properties.get('platform') == 'S2B', 'wrong platform'
    assert reader.scene_properties.get('processing_level').value == 'LEVEL1C', 'wrong processing level'

    # check band list
    bands = reader.get_bandnames()
    assert len(bands) == len(band_selection), 'number of bands is wrong'
    assert 'scl' not in bands, 'SCL band cannot be available for L1C data'


def test_read_from_bandstack_l2a(datadir, get_bandstack, get_polygons):
    """
    handling Sentinel-2 bandstacked geoTiff files derived from AgriSatPy's
    default processing pipeline using masks
    """

    fname_bandstack = get_bandstack()
    fname_polygons = get_polygons()

    handler = Sentinel2Handler()

    # attempt to read from .SAFE archive but single geoTiff provided -> must fail
    with pytest.raises(Exception):
        handler.read_from_safe(in_dir=fname_bandstack)

    # read data for field parcels, only
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons
    )

    # check data types
    assert handler.get_band('blue').dtype == float, 'wrong  data type for spectral band, expected float'
    assert isinstance(handler.get_band('blue'), np.ma.core.MaskedArray), 'expected masked array'

    assert handler.check_is_bandstack() == True, 'expected band-stacked object'

    # check bands, SCL must be available
    assert 'scl' in handhttps://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2019_2585-1130/swissalti3d_2019_2585-1130_2_2056_5728.tifler.get_bandnames(), 'scene classification layer not read'
    assert handler.get_band('scl').dtype == 'uint8', 'wrong datatype for SCL'
    assert len(handler.get_bandnames()) == 11, 'wrong number of bands read'
    assert len(handler.get_bandaliases()) == 11, 'band aliases not provided correctly'
    assert (handler.get_band('B02').data == handler.get_band('blue').data).all(), 'band aliasing not working'

    # check attributes
    assert len(handler.get_attrs()['scales']) == 11, 'wrong number of bands in attributes'
    assert len(handler.get_attrs()['descriptions']) == 11, 'wrong number of bands in attributes'
    assert set(handler.get_attrs()['descriptions']) == set(handler.get_bandaliases().values())

    # check conversion to xarray
    xds = handler.to_xarray()

    assert xds.crs == 32632, 'wrong CRS in xarray'
    assert len(xds.attrs.keys()) > 0, 'no image attributes set'
    assert isinstance(xds.blue.data, np.ndarray), 'image data not converted to xarray'
    assert handler.get_band('B02').data[100,100].astype(float) == xds.blue.data[100,100], 'incorrect values after conversion'
    assert (np.isnan(xds.blue.data)).any(), 'xarray band has no NaNs although it should'

    # read data from bandstack without applying int to float conversion
    band_selection = ['B02', 'B03', 'B04']

    handler = Sentinel2Handler()
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons,
        full_bounding_box_only=True,
        int16_to_float=False,
        band_selection=band_selection
    )

    assert handler.get_band('blue').dtype == 'uint16', 'wrong data type for spectral band data'
    assert handler.get_band('scl').dtype == 'uint8', 'wrong data type for SCL'



def test_read_from_safe_with_mask_l2a(datadir, get_s2_safe_l2a, get_polygons, get_polygons_2):
    """handling Sentinel-2 data from .SAFE archives (masking)"""

    in_dir = get_s2_safe_l2a()
    in_file_aoi = get_polygons()

    # read using polygons outside of the tile extent -> should fail
    handler = Sentinel2Handler()
    with pytest.raises(Exception):
        handler.read_from_safe(
            in_dir=in_dir,
            in_file_aoi=in_file_aoi
        )

    # read using polygons overlapping the tile extent
    in_file_aoi = get_polygons_2()

    handler.read_from_safe(
        in_dir=in_dir,
        in_file_aoi=in_file_aoi
    )

    assert not handler.check_is_bandstack(), 'data read from SAFE archive cannot be a bandstack'

    # to_xarray should fail because of different spatial resolutions
    with pytest.raises(ValueError):
        handler.to_xarray()

    # as well as the calculation of the TCARI-OSAVI ratio
    with pytest.raises(Exception):
        handler.calc_vi('TCARI_OSAVI')

    # but calculation of NDVI should work because it requires the 10m bands only
    handler.calc_vi('NDVI')
    assert 'NDVI' in handler.get_bandnames(), 'NDVI not added to handler'
    assert 'NDVI' in handler.get_attrs().keys(), 'NDVI not added to dataset attributes'

    # stacking bands should fail because of different spatial resolutions
    with pytest.raises(InputError):
        handler.get_bands()

    # dropping all 20m should then allow stacking operations and conversion to xarray
    bands_to_drop = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'scl']
    for band_to_drop in bands_to_drop:
        handler.drop_band(band_to_drop)

    assert len(handler.get_bandnames()) == 5, 'too many bands left'
    assert handler.check_is_bandstack(), 'data should now fulfill bandstack criteria'

    xds = handler.to_xarray()

    # make sure attributes were set correctly in xarray
    assert len(xds.attrs['scales']) == len(handler.get_bandnames()), 'wrong number of bands in attributes'


def test_ignore_scl(datadir, get_s2_safe_l2a, get_polygons_2):
    """ignore the SCL on reading"""

    in_dir = get_s2_safe_l2a()
    in_file_aoi = get_polygons_2()

    # read using polygons outside of the tile extent -> should fail
    handler = Sentinel2Handler()
    handler.read_from_safe(
        in_dir=in_dir,
        in_file_aoi=in_file_aoi,
        read_scl=False
    )
    assert 'scl' not in handler.get_bandnames(), 'SCL band should not be available'

    # read with weird band ordering
    band_selection = ['B08','B06']
    handler = Sentinel2Handler()
    handler.read_from_safe(
        in_dir=in_dir,
        in_file_aoi=in_file_aoi,
        band_selection=band_selection,
        read_scl=False
    )
    assert 'scl' not in handler.get_bandnames(), 'SCL band should not be available'
    assert handler.get_bandnames() == ['nir_1', 'red_edge_2'], 'wrong order of bands'
    with pytest.raises(BandNotFoundError):
        handler.get_meta('scl')


def test_band_selections(datadir, get_s2_safe_l2a, get_polygons, get_polygons_2,
                         get_bandstack):
    """testing invalid band selections"""

    in_dir = get_s2_safe_l2a()
    in_file_aoi = get_polygons_2()

    # attempt to read no-existing bands
    handler = Sentinel2Handler()
    with pytest.raises(IndexError):
        handler.read_from_safe(
            in_dir=in_dir,
            in_file_aoi=in_file_aoi,
            band_selection=['B02','B13']
        )

    fname_bandstack = get_bandstack()
    in_file_aoi = get_polygons()
    handler = Sentinel2Handler()
    with pytest.raises(InputError):
        handler.read_from_bandstack(
            fname_bandstack=fname_bandstack,
            band_selection=['B02','B13']
        )


def test_read_from_safe_l2a(datadir, get_s2_safe_l2a):
    """handling Sentinel-2 data from .SAFE archives (no masking)"""

    in_dir = get_s2_safe_l2a()

    # read without AOI file
    reader = Sentinel2Handler()
    band_selection = ['B02', 'B03', 'B04', 'B08', 'B8A']
    reader.read_from_safe(
        in_dir=in_dir,
        band_selection=band_selection
    )

    # check if the object can be considered a band stack -> should not be the case
    assert not reader.check_is_bandstack(), 'data is labelled as band-stack but it should not'

    # check scene properties
    acquisition_time = reader.scene_properties.get('acquisition_time')
    assert acquisition_time.date() == date(2019,5,24), 'acquisition date is wrong'
    assert reader.scene_properties.get('sensor') == 'MSI', 'wrong sensor'
    assert reader.scene_properties.get('platform') == 'S2A', 'wrong platform'
    assert reader.scene_properties.get('processing_level').value == 'LEVEL2A', 'wrong processing level'

    # check band list
    bands = reader.get_bandnames()
    assert len(bands) == len(band_selection), 'number of bands is wrong'
    assert 'scl' in bands, 'expected SCL band'

    # check band aliases
    aliases = reader.get_bandaliases()
    assert len(aliases) == len(bands), 'number of aliases does not match number of bands'
    assert set(aliases.keys()) == set(bands), 'wrong band alias mapping'

    # check single band data
    blue = reader.get_band('B02')

    assert type(blue) in (np.ndarray, np.array), 'wrong datatype for band'
    assert len(blue.shape) == 2, 'band array is not 2-dimensional'
    assert blue.shape == (10980, 10980), 'wrong shape of band array'
    assert blue.dtype == float, 'band data was not type-casted to float'
    assert blue.min() >= 0., 'reflectance data must not be smaller than zero'

    scl = reader.get_band('scl')
    assert type(scl) in (np.ndarray, np.array), 'wrong datatype for band'
    assert scl.dtype == 'uint8', 'SCL has dtype uint8'
    assert scl.shape == (5490,5490), 'wrong shape of band array'
    assert scl.max() < 12, 'invalid value for scl'
    assert scl.min() >= 0, 'invalid value for scl'

    spatial_res = reader.get_spatial_resolution()
    spatial_res_blue = reader.get_spatial_resolution('blue')

    assert type(spatial_res) == dict, 'expected dictionary for spatial resolutions'
    assert len(spatial_res) == len(band_selection)

    assert spatial_res['blue'] == spatial_res_blue, 'spatial resolution not stored correctly'
    assert spatial_res['blue'] == spatial_res['green'] == spatial_res['red'] == spatial_res['nir_1']
    assert spatial_res['blue'] != spatial_res['scl'], 'scl should have different spatial resolution'

    assert spatial_res['blue'].x == 10, 'blue should have a spatial resolution of 10m'
    assert spatial_res['blue'].x == -spatial_res['blue'].y, 'signs of x and y should be different'
    assert spatial_res['scl'].x == 20, 'scl should have a spatial resolution of 20m before resampling'
    assert spatial_res['nir_2'].x == 20, 'B8A should have a spatial resolution of 20m before resampling'

    # get non-exisiting bands
    with pytest.raises(BandNotFoundError):
        non_existing_band = reader.get_band('B01')

    # check band coordinates
    coords_blue = reader.get_coordinates('blue')
    assert coords_blue['y'].shape[0] == blue.shape[0], 'miss-match between number of y coordinates and rows'
    assert coords_blue['x'].shape[0] == blue.shape[1], 'miss-match between number of x coordinates and columns'

    # check dataset metadata

    ###### attributes
    attrs = reader.get_attrs()
    assert type(attrs) == dict, 'expected dictionary for attributes'
    assert len(attrs) == len(band_selection), 'expected an entry for each band'
    assert set(attrs['blue'].keys()) == {'is_tiled', 'nodatavals', 'scales', 'offsets'}

    attrs_blue = reader.get_attrs('blue')
    attrs_b02 = reader.get_attrs('B02')

    assert attrs['blue'] == attrs['green'] == attrs['red'] == attrs['nir_1'], 'miss-match between bands'
    assert attrs['blue'] == attrs['scl'], 'SCL and spectral bands must have same attributes'

    assert attrs_blue == attrs_b02, 'band aliasing returned different band attributes'
    assert attrs_blue == attrs['blue'], 'wrong band attributes returned'

    ###### meta-dict
    meta = reader.get_meta()
    assert type(meta) == dict, 'expected dictionary for meta-dict'
    assert len(meta) == len(band_selection), 'expected an entry for each band'
    assert set(meta['blue'].keys()) == {'driver', 'dtype', 'nodata', 'width', 'height', 'count', 'crs', 'transform'}

    meta_blue = reader.get_meta('blue')
    meta_b02 = reader.get_meta('B02')

    assert meta_blue == meta_b02, 'band aliasing returned different band attributes'
    assert meta_blue == meta['blue'], 'wrong band attributes returned'

    assert meta['blue'] == meta['green'] == meta['red'] == meta['nir_1'], 'miss-match between bands'
    assert meta['blue'] != meta['scl'], 'spectral bands and scl should not have same meta'

    ##### bounds
    bounds = reader.get_bounds()
    assert type(bounds) == dict, 'expected dictionary for bounds'
    assert len(bounds) == len(band_selection), 'expected an entry for each band'
    assert type(bounds['blue']) == Polygon, 'expected polygon'

    bounds_bounds = reader.get_bounds(return_as_polygon=False)
    assert type(bounds_bounds['blue']) == BoundingBox, 'expected bounding box'

    bounds_blue = reader.get_bounds('blue')
    assert bounds_blue == bounds['blue']

    bounds_bounds_blue = reader.get_bounds('blue', return_as_polygon=False)
    assert bounds_bounds_blue == bounds_bounds['blue']

    # check the RGB
    fig_rgb = reader.plot_rgb()
    assert type(fig_rgb) == Figure, 'plotting of RGB bands failed'

    # and the false-color near-infrared
    fig_nir = reader.plot_false_color_infrared()
    assert type(fig_nir) == Figure, 'plotting of false color NIR failed'

    # check the scene classification layer
    fig_scl = reader.plot_scl()
    assert type(fig_scl) == Figure, 'plotting of SCL failed'

    reader.resample(
        target_resolution=10,
        resampling_method=cv2.INTER_CUBIC,
        bands_to_exclude=['scl']
    )

    # scl should not have changed
    assert reader.get_band('scl').shape == (5490,5490), 'SCL was resampled although excluded'

    # but B8A should have 10m resolution now
    assert reader.get_band('B8A').shape == (10980,10980), 'B8A was not resampled although selected'
    assert reader.get_spatial_resolution('B8A').x == 10, 'meta B8A was not updated to 10m'

    # Vegetation Index calculation
    reader.calc_vi(vi='NDVI')
    assert type(reader.get_band('NDVI')) in (np.array, np.ndarray), 'VI is not an array'
    assert reader.get_band('NDVI').shape == reader.get_band('red').shape == reader.get_band('nir_1').shape, 'shapes do not fit'
    assert 'NDVI' in reader.get_bandnames(), 'NDVI not found as band name'

    # add custom band
    band_to_add = np.zeros_like(blue)
    reader.add_band(band_name='test', band_data=band_to_add)
    assert (reader.get_band('test') == band_to_add).all(), 'band was not added correctly'
    assert 'test' in reader.get_bandnames(), 'band "test" not found in reader entries'

    # check cloud masking using SCL
    cloudy_pixels = reader.get_cloudy_pixel_percentage()
    assert 0 <= cloudy_pixels <= 100, 'cloud pixel percentage must be between 0 and 100%'

    # check blackfill (there is some but not the entire scene is blackfilled)
    assert not reader.is_blackfilled(), 'blackfill detection did not work out - to many false positives'
    blackfill_mask = reader.get_blackfill('blue')
    assert blackfill_mask.dtype == bool, 'A boolean mask is required for the blackfill'
    assert 0 < np.count_nonzero(blackfill_mask) < np.count_nonzero(~blackfill_mask)

    # try masking using the SCL classes. Since SCL is not resampled yet this should fail
    with pytest.raises(Exception):
        reader.mask_clouds_and_shadows(bands_to_mask=['blue'])

    # resample SCL and try masking again
    reader.resample(target_resolution=10, pixel_division=True)

    assert reader.get_band('scl').shape == blue.shape, 'SCL resampling failed'
    reader.mask_clouds_and_shadows(bands_to_mask=['blue'])
    reader.mask_clouds_and_shadows(bands_to_mask=['B03', 'B04', 'NDVI'])

    # since the test since contains some clouds the bands should slightly differ after masking
    assert not (reader.get_band('blue') == blue).all(), 'cloud masking had no effect'

    # drop a band
    reader.drop_band('test')

    assert 'test' not in reader.get_bandnames(), 'band "test" still available although dropped'
    with pytest.raises(Exception):
        reader.get_band('test')
    with pytest.raises(Exception):
        reader.get_meta('test')

    # re-project a band to another UTM zone (33)
    blue_meta_utm32 = reader.get_meta('blue')
    blue_bounds_utm32 = reader.get_meta('blue')
    reader.reproject_bands(
        target_crs=32633,
        blackfill_value=0,
    )

    assert reader.check_is_bandstack(), 'data should fulfill bandstack criteria but doesnot'

    assert reader.get_epsg('blue') == 32633, 'projection was not updated'
    assert reader.get_meta('blue') != blue_meta_utm32, 'meta was not updated'
    assert reader.get_bounds('blue') != blue_bounds_utm32, 'bounds were not updated'

    # try writing bands to output file
    reader.write_bands(
        out_file=datadir.joinpath('scl.tif'),
        band_names=['scl']
    )

    assert datadir.joinpath('scl.tif').exists(), 'output raster file not found'

    # try converting data to xarray
    xds = reader.to_xarray()

    assert xds.crs == 32633, 'EPSG got lost in xarray dataset'
    dims = dict(xds.dims)
    assert list(dims.keys()) == ['y', 'x'], 'wrong coordinate keys'
    assert tuple(dims.values()) == reader.get_band('blue').shape, 'wrong shape of array in xarray dataset'



