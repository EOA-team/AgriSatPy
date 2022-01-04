import pytest
import numpy as np


from agrisatpy.io import SatDataHandler
from agrisatpy.utils.exceptions import InputError, BandNotFoundError


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
    handler.add_band(band_name='test', band_data=band_to_add)
    assert (handler.get_band('test') == band_to_add).all(), 'array not the same after adding it'


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
