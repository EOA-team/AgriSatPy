
import cv2
import pytest
import rasterio as rio

from agrisatpy.io.sentinel2 import Sentinel2Handler
from agrisatpy.spatial_resampling.sentinel2 import resample_and_stack_s2


@pytest.mark.parametrize('resampling_method', [(cv2.INTER_CUBIC), (cv2.INTER_NEAREST_EXACT)])
def test_resample_and_stack_s2(datadir, get_s2_safe_l2a, pixel_division):
    """Tests the resample and band stack module from the pipeline"""

    in_dir = get_s2_safe_l2a()

    fnames_dict = resample_and_stack_s2(
        in_dir=in_dir,
        out_dir=datadir,
        pixel_division=pixel_division
    )

    assert len(fnames_dict) == 4, 'expected four dict items in case of L2A data'

    if pixel_division:
        assert 'pixel-division' in fnames_dict['bandstack'].name, 'wrong file naming'
        assert 'pixel-division' in fnames_dict['scl'].name, 'wrong file naming'
    else:
        assert 'cubic' in fnames_dict['bandstack'].name, 'wrong file naming'
        assert 'cubic' in fnames_dict['scl'].name, 'wrong file naming'

    assert fnames_dict['bandstack'].exists(), 'no output band stack'
    assert fnames_dict['scl'].exists(), 'no output SCL found'
    assert fnames_dict['rgb_preview'].exists(), 'no output RGB found'
    assert fnames_dict['scl_preview'].exists(), 'no output SCL preview found'

    # make sure output dataset contains data
    with rio.open(fnames_dict['bandstack']) as src:
        meta = src.meta
        descriptions = src.descriptions
        band_data = src.read(1)

    assert meta['crs'] == 32632, 'data has wrong projection'
    assert meta['dtype'] == 'uint16', 'data has wrong dtype'
    assert meta['count'] == 10, 'wrong number of spectral bands'
    assert band_data.shape == (10980,10980), 'band data has wrong shape'
    # make sure bands are in the right order
    assert list(descriptions) == ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

    # check SCL
    with rio.open(fnames_dict['scl']) as src:
        meta_scl = src.meta
        scl_data = src.read(1)

    assert scl_data.shape == band_data.shape, 'SCL data was not resampled to resolution of spectral bands'
    assert meta_scl['dtype'] == 'uint8', 'SCL data has wrong datatype'
    assert 0 <= scl_data.min() <= 11, 'invalid values for SCL data'
    assert 0 <= scl_data.max() <= 11, 'invalid values for SCL data'

    # check if data from blue band is still the same (not subject to resampling)
    handler = Sentinel2Handler()
    handler.read_from_safe(
        in_dir=in_dir,
        band_selection=['B02'],
        read_scl=False,
        int16_to_float=False
    )

    assert (handler.get_band('B02') == band_data).all(), 'band data is not the same although it should'
    