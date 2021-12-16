
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
from agrisatpy.utils.exceptions import BandNotFoundError



@pytest.fixture()
def get_s2_safe_l2a():
    """
    Get Sentinel-2 testing data in L2A processing level. If not available yet
    download the data from the Menedely dataset link provided
    """

    def _get_s2_safe_l2a():

        testdata_dir = Path('../../data')
        testdata_fname = testdata_dir.joinpath('S2A_MSIL2A_20190524T101031_N0212_R022_T32UPU_20190524T130304.SAFE')
    
        # download URL
        url = 'https://data.mendeley.com/public-files/datasets/ckcxh6jskz/files/e97b9543-b8d8-436e-b967-7e64fe7be62c/file_downloaded'
    
        if not testdata_fname.exists():
        
            # download dataset
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(testdata_fname, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=5096):
                    fd.write(chunk)
        
            # unzip dataset
            unzip_datasets(download_dir=testdata_dir)
            
        return testdata_fname

    return _get_s2_safe_l2a
            

def test_read_from_safe_l2a(datadir, get_s2_safe_l2a):
    """reading Sentinel-2 data from .SAFE archives"""

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
