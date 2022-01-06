import pytest
import numpy as np
import geopandas as gpd

from agrisatpy.io import SatDataHandler
from agrisatpy.utils.exceptions import InputError
from agrisatpy.utils.exceptions import BandNotFoundError
from agrisatpy.utils.exceptions import DataExtractionError


def test_add_bands(datadir, get_bandstack):
    """tests adding and removing bands from a handler"""

    fname_bandstack = get_bandstack()
    handler = SatDataHandler()
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack
    )
    old_scales = len(handler.get_attrs()['scales'])
    assert old_scales == 10, 'too few entries in scales attribute'

    # add a band of same shape as those bands already available
    band_to_add = np.zeros_like(handler.get_band('B02'))
    handler.add_band(
        band_name='test',
        band_data=band_to_add,
        snap_band='B02'
    )

    assert 'test' in handler.get_bandnames(), 'band name not added'
    assert handler.check_is_bandstack(), 'handler should still fulfill the bandstack criteria'
    # check meta, bounds and attribs
    assert handler.get_meta() == handler.get_meta('test'), 'meta not correct'
    assert handler.get_bounds() == handler.get_bounds('test'), 'bounds not correct'
    assert handler.get_attrs('test')['descriptions'] == tuple(['TEST']), 'description not set'
    assert handler.get_attrs('test')['scales'] == tuple([1]), 'scale not set'
    assert handler.get_attrs('B02')['scales'] == handler.get_attrs('test')['scales'], 'scales not taken from reference'
    assert len(handler.get_attrs()['scales']) == old_scales + 1, 'attributes not updated'

    # update with raster not matching snap band

    # update with non-existing snap raster

    # add band without snap band
    


def test_read_pixels(datadir, get_bandstack, get_points, get_polygons):
    """tests reading pixel values from bandstack at point locations"""

    fname_bandstack = get_bandstack()
    fname_points = get_points()

    # pixel reading refers to a classmethod, therefore no constructor call is required
    gdf = SatDataHandler.read_pixels(
        point_features=fname_points,
        raster=fname_bandstack
    )

    assert len(gdf.columns) == 12, 'wrong number of columns in dataframe'
    assert not gdf.empty, 'no pixels were read'
    assert isinstance(gdf, gpd.GeoDataFrame), 'not a geodataframe'
    assert 'B02' in gdf.columns, 'band not found'
    assert not (gdf['B02'].isna()).any(), 'nans encountered'
    assert gdf.crs == 32632, 'wrong EPSG code'

    # read only a subset of bands
    gdf2 = SatDataHandler.read_pixels(
        point_features=fname_points,
        raster=fname_bandstack,
        band_selection=['B03','B11']
    )

    assert len(gdf2.columns) == 4, 'wrong number of columns in dataframe'
    assert not gdf2.empty, 'no pixels were read'
    assert 'B02' not in gdf2.columns, 'band found although it was not selected for extraction'
    assert not (gdf2['B03'].isna()).any(), 'nans encountered'
    assert gdf2.crs == gdf.crs, 'wrong EPSG code'
    assert (gdf['B03'] == gdf2['B03']).all(), 'band data not the same'
    assert (gdf['B11'] == gdf2['B11']).all(), 'band data not the same'

    # try to read using wrong geometry type -> must fail
    fname_polygons = get_polygons()
    with pytest.raises(ValueError):
        gdf = SatDataHandler.read_pixels(
            point_features=fname_polygons,
            raster=fname_bandstack,
            band_selection=['B03','B11']
        )

    # TODO: points partly outside of raster

    # TODO: points complete outside of raster



def test_add_band_from_shp(datadir, get_bandstack, get_polygons, get_polygons_2):
    """tests adding a band from a shapefile (rasterization)"""

    fname_bandstack = get_bandstack()
    fname_polygons = get_polygons()

    handler = SatDataHandler()
    # read data for field parcels, only (masked array)
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons
    )
    handler.add_bands_from_vector(
        in_file_vector=fname_polygons,
        snap_band='B02'
    )

    assert len(handler.get_bandnames()) == 12, 'wrong number of bands'
    assert 'GIS_ID' in handler.get_bandnames(), 'band GIS_ID not rasterized from vector file'
    assert 'NUTZUNGSCO' in handler.get_bandnames(), 'band NUTZUNGSCO not rasterized from vector file'
    assert 'NUTZUNG' not in handler.get_bandnames(), 'character attribute NUTZUNG rasterized'

    # make sure rasterized matches snap band exactly
    gis_id1 = handler.get_band('GIS_ID')
    assert np.count_nonzero(~np.isnan(gis_id1)) > 0, 'band contains Nodata, only'
    snap_band = handler.get_band('B02')
    # the number of nodata pixels must be the same
    snap_band_compressed_shape = snap_band.compressed().shape[0]
    assert np.count_nonzero(~np.isnan(gis_id1)) == snap_band_compressed_shape, 'band contains wrong number of pixels'

    # read data using entire AOI extent
    handler = SatDataHandler()
    # read data for field parcels, only (masked array)
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons,
        full_bounding_box_only=True
    )
    handler.add_bands_from_vector(
        in_file_vector=fname_polygons,
        snap_band='B02'
    )

    assert len(handler.get_bandnames()) == 12, 'wrong number of bands'
    assert 'GIS_ID' in handler.get_bandnames(), 'band GIS_ID not rasterized from vector file'
    assert 'NUTZUNGSCO' in handler.get_bandnames(), 'band NUTZUNGSCO not rasterized from vector file'
    assert 'NUTZUNG' not in handler.get_bandnames(), 'character attribute NUTZUNG rasterized'

    # make sure the rasterized band is still the same as before (minor difference occure because
    # of the masking algorithm)
    gis_id2 = handler.get_band('GIS_ID')
    assert np.count_nonzero(~np.isnan(gis_id2)) > 0, 'band contains Nodata, only'
    assert np.count_nonzero(gis_id1 == gis_id2) == snap_band_compressed_shape, 'rasterized band differs although it should not'
    snap_band = handler.get_band('B02')
    # now the number of npdata pixels must differ
    assert np.count_nonzero(~np.isnan(gis_id2)) < snap_band.compressed().shape[0]

    # test conversion to dataframe
    gdf = handler.to_dataframe()
    assert gdf.GIS_ID.dtype == 'float32', 'wrong float data type'

    # test conversion to xarray
    xds = handler.to_xarray()
    assert xds.GIS_ID.data.dtype == 'float32', 'wrong float data type'

    # try non-existing snap band -> must fail
    handler = SatDataHandler()
    # read data for field parcels, only (masked array)
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons,
        full_bounding_box_only=True
    )
    with pytest.raises(BandNotFoundError):
        handler.add_bands_from_vector(
            in_file_vector=fname_polygons,
            snap_band='blue'
        )

    # try unsupported data type -> must fail
    with pytest.raises(ValueError):
        handler.add_bands_from_vector(
            in_file_vector=fname_polygons,
            snap_band='B02',
            default_float_type='uint16'
        )

    # set another blackfill (i.e., no data) value
    handler.add_bands_from_vector(
            in_file_vector=fname_polygons,
            snap_band='B02',
            blackfill_value=-99999.
    )

    gis_id = handler.get_band('GIS_ID')
    assert np.count_nonzero(gis_id != -99999.) == snap_band_compressed_shape, 'blackfill value not set to all no data pixels'
    assert np.count_nonzero(gis_id == np.nan) == 0, 'np.nans encounter although they should be set to user-defined value'

    # try with different attribute selections
    handler.drop_band('GIS_ID')
    handler.drop_band('NUTZUNGSCO')

    handler.add_bands_from_vector(
        in_file_vector=fname_polygons,
        snap_band='B02',
        attribute_selection=['GIS_ID']
    )
    assert 'GIS_ID' in handler.get_bandnames(), 'attribute not rasterized and added'
    assert 'NUTZUNGSCO' not in handler.get_bandnames(), 'attribute rasterized although not selected'

    # invalid attribute selection
    with pytest.raises(AttributeError):
        handler.add_bands_from_vector(
            in_file_vector=fname_polygons,
            snap_band='B02',
            attribute_selection=['foo']
        )

    # try with vector features in different CRS
    gpd_test = gpd.read_file(fname_polygons)
    gpd_test.to_crs(4326, inplace=True)
    fname_polygons_wgs84 = datadir.joinpath('polygons_wgs84.shp')
    gpd_test.to_file(fname_polygons_wgs84)

    handler.add_bands_from_vector(
        in_file_vector=fname_polygons_wgs84,
        snap_band='B02',
        attribute_selection=['NUTZUNGSCO']
    )
    assert 'NUTZUNGSCO' in handler.get_bandnames(), 'attribute not rasterized and added'
    nutzungsco = handler.get_band('NUTZUNGSCO')
    assert np.count_nonzero(~np.isnan(nutzungsco)) == snap_band_compressed_shape, 'band contains wrong number of pixels'
    assert handler.get_epsg('NUTZUNGSCO') == 32632, 'wrong EPSG code'

    # try with vector features not overlapping the image data
    fname_polygons_outside = get_polygons_2()
    with pytest.raises(DataExtractionError):
        handler.add_bands_from_vector(
            in_file_vector=fname_polygons_outside,
            snap_band='B02'
        )


def test_conversion_to_gpd(datadir, get_bandstack, get_polygons):
    """tests the conversion of raster band data to a geopandas GeoDataFrame"""

    # test from bandstack with multiple bands masked to the extent of polygons
    # spatial resolutions are all the same
    fname_bandstack = get_bandstack()
    fname_polygons = get_polygons()

    handler = SatDataHandler()

    # read data for field parcels, only (masked array)
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons
    )

    # conversion using the default settings
    gdf = handler.to_dataframe()

    assert gdf.crs == 32632, 'geodataframe has wrong CRS'
    assert gdf.shape[1] == 11, 'geodataframe has wrong number of bands'
    assert gdf.shape[0] == 29674, 'geodataframe has wrong number of pixels'
    assert (gdf.geom_type == 'Point').all(), 'geodataframe is not of type Point geometry'
    assert 'geometry' in gdf.columns, 'no geometry column found'
    assert gdf.dtypes['geometry'] == 'geometry', 'geometry column has no proper geom datatpye'
    assert all(elem in gdf.columns for elem in handler.get_bandnames()), 'band names not passed'
    assert all(gdf.dtypes[handler.get_bandnames()] == 'uint16'), 'band data has wrong datatype'

    # conversion using only one band
    gdf2 = handler.to_dataframe(band_names=['B02'])

    assert gdf2.shape[1] == 2, 'too many columns'
    assert gdf2.shape[0] == 29674, 'geodataframe has wrong number of pixels'
    assert 'B02' in gdf2.columns, 'selected band name not used as column name'
    assert all(gdf['B02'] == gdf2['B02']), 'band data not inserted correctly into geodataframe'

    # conversion using two bands
    gdf3 = handler.to_dataframe(band_names=['B03','B02'])

    assert gdf3.shape[1] == 3, 'too many columns'
    assert gdf3.shape[0] == 29674, 'geodataframe has wrong number of pixels'
    assert all(gdf['B02'] == gdf3['B02']), 'band data not inserted correctly into geodataframe'
    assert all(gdf['B03'] == gdf3['B03']), 'band data not inserted correctly into geodataframe'

    # read data for entire area of interest (will result in np.ndarrays instead of np.ma.MaskedArray)
    handler = SatDataHandler()

    # read data for field parcels, only (masked array)
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons,
        full_bounding_box_only=True
    )

    gdf = handler.to_dataframe()

    band_shape = handler.get_band_shape('B02')
    num_pixels = band_shape.nrows * band_shape.ncols
    assert gdf.shape[0] == num_pixels, 'wrong number of pixels in dataframe'
    assert gdf.shape[1] == 11, 'wrong number of columns in dataframe'

    # check coordinates
    band_coords = handler.get_coordinates('B02', shift_to_center=False)
    assert (band_coords['x'] == gdf.geometry.x.unique()).all(), 'x coordinates distorted in dataframe'
    assert (band_coords['y'] == gdf.geometry.y.unique()).all(), 'y coordinates distored in dataframe'

    # shift pixel coordinates to center and check again
    gdf2 = handler.to_dataframe(band_names=['B02'], pixel_coordinates_centered=True)
    band_coords = handler.get_coordinates('B02', shift_to_center=True)
    assert (band_coords['x'] == gdf2.geometry.x.unique()).all(), 'x coordinates distorted in dataframe'
    assert (band_coords['y'] == gdf2.geometry.y.unique()).all(), 'y coordinates distored in dataframe'

    assert not (gdf.geometry.x == gdf2.geometry.x).any(), 'x coordinates not shifted'
    assert not (gdf.geometry.y == gdf2.geometry.y).any(), 'y coordinates not shifted'
    assert (gdf['B02'] == gdf['B02']).all(), 'band values distorted'


def test_read_from_bandstack_with_mask(datadir, get_bandstack, get_polygons):
    """reads from multi-band geoTiff file using a mask"""

    fname_bandstack = get_bandstack()
    fname_polygons = get_polygons()

    handler = SatDataHandler()

    # read data for field parcels, only
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons
    )

    assert handler.check_is_bandstack(), 'not recognized as bandstack although data should'
    assert isinstance(handler.get_band('B02'), np.ma.core.MaskedArray), 'band data was not masked'
    assert len(handler.get_bandnames()) == 10, 'wrong number of bands'
    assert len(handler.get_bandaliases()) == 0, 'band aliases available although they should not'

    # direct calculation of NDVI should fail because color names are not defined
    with pytest.raises(Exception):
        ndvi = handler.calc_vi('NDVI')

    assert handler.get_band('B12').dtype == 'uint16', 'wrong band data type'

    # attempt to access non-existing band
    with pytest.raises(BandNotFoundError):
        handler.get_band('notthere')

    # check coordinates
    coords = handler.get_coordinates('B03', shift_to_center=False)
    assert 'x' in coords.keys(), 'x coordinates not provided'
    assert 'y' in coords.keys(), 'y coordinates not provided'
    assert coords['x'].shape[0] == handler.get_band_shape('B8A').ncols, 'wrong number of image columns'
    assert coords['y'].shape[0] == handler.get_band_shape('B8A').nrows, 'wrong number of image rows'
    assert handler.get_band_shape('B8A').nrows == handler.get_band('B8A').shape[0], 'wrong array shape y dim'
    assert handler.get_band_shape('B8A').ncols == handler.get_band('B8A').shape[1], 'wrong array shape x dim'

    # check geo-localisation and image metadata
    assert coords['x'][0] == handler.get_meta()['transform'][2], 'upper left x coordinate shifted'
    assert coords['y'][0] == handler.get_meta()['transform'][5], 'upper left y coordinate shifted'
    assert coords['x'].shape[0] == handler.get_meta()['width'], 'mismatch in image width information'
    assert coords['y'].shape[0] == handler.get_meta()['height'], 'mismatch in image height information'
    assert handler.get_meta()['count'] == 10, 'mismatch in number of bands'

    # check attributes
    assert len(handler.get_attrs()['nodatavals']) == 10, 'attributes not set correctly'

    # check conversion to xarray dataset. The masked integer array should be converted to float
    # and masked pixels be set to NaN
    xds = handler.to_xarray()

    assert xds.crs == 32632, 'wrong CRS in xarray'
    assert len(xds.attrs.keys()) > 0, 'no image attributes set'
    assert isinstance(xds.B02.data, np.ndarray), 'image data not converted to xarray'
    assert handler.get_band('B02').data[100,100].astype(float) == xds.B02.data[100,100], 'incorrect values after conversion'
    assert (np.isnan(xds.B02.data)).any(), 'xarray band has no NaNs although it should'

    handler = SatDataHandler()

    # read data for field parcels plus the surrounding bounding box
    handler.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=fname_polygons,
        full_bounding_box_only=True
    )

    assert isinstance(handler.get_band('B02'), np.ndarray), 'band data was masked'

    # when converting to xarray no nans should appear
    xds = handler.to_xarray()

    assert not (np.isnan(xds.B02.data)).any(), 'NaNs encountered'

    # add a band with x,y shape smaller than bandstack -> must fail
    band_to_add = np.zeros((10,10))

    with pytest.raises(InputError):
        handler.add_band(band_name='test', band_data=band_to_add)

    # add a band with correct shape, should work
    band_to_add = np.zeros_like(handler.get_band('B02'))
    handler.add_band(
        band_name='test',
        band_data=band_to_add,
        snap_band='B02'
    )
    assert (handler.get_band('test') == band_to_add).all(), 'array not the same after adding it'
    assert handler.check_is_bandstack(), 'handler does not fulfill bandstack criteria any more'


@pytest.mark.parametrize(
    'url',
    ['https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2019_2585-1130/swissalti3d_2019_2585-1130_2_2056_5728.tif']
)
def test_from_cloudoptimized_geotiff(datadir, url):
    """
    tests reading and handling data from cloud-optimized geoTiff (digital elevation
    model tile from Swisstopo)
    """

    handler = SatDataHandler()
    handler.read_from_bandstack(
        fname_bandstack=url
    )

    assert len(handler.get_bandnames()) > 0, 'no data read'
    band_data = handler.get_band('B1')
    assert isinstance(band_data, np.ndarray), 'band data not read correctly'
    assert band_data.dtype == 'float32', 'expected floating point data'
    assert len(handler.get_attrs()) > 0, 'no attributes parsed'
    assert handler.get_attrs()['units'][0] == 'metre', 'physical unit attribute missing or wrong'
    meta = handler.get_meta()
    assert meta['crs'] == 2056, 'wrong CRS parsed'

    # reproject into UTM zone 32 using nearest neighbor (def.) and save file to disk
    handler.reproject_bands(
        target_crs=32632,
        blackfill_value=handler.get_attrs()['nodatavals'][0]
    )
    assert handler.get_epsg() == 32632, 'EPSG not updated'
    band_data_utm = handler.get_band('B1')
    assert band_data_utm.max() == band_data.max(), 'nearest neighbor interpolation must not change raster values'
    assert band_data.shape != band_data_utm.shape, 'shape of array should change after reprojection when no dst affine is provided'
    assert handler.get_band_shape('B1').nrows == handler.get_meta()['height'], 'mismatch between rows and image height'
    assert handler.get_band_shape('B1').ncols == handler.get_meta()['width'], 'mismatch between cols and image width'

    # convert to xarray
    xds = handler.to_xarray()
    assert (xds.B1.data == band_data_utm).all()
