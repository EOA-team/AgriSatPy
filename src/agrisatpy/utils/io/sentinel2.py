'''
Created on Nov 24, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

import numpy as np
import rasterio as rio
import geopandas as gpd
import rasterio.mask

from rasterio.coords import BoundingBox
from shapely.geometry import box
from geopandas.geodataframe import GeoDataFrame
from pathlib import Path
from typing import Optional
from typing import Dict
from typing import List

from agrisatpy.utils.constants.sentinel2 import ProcessingLevels
from agrisatpy.utils.reprojection import check_aoi_geoms
from pickle import FALSE


s2_band_mapping = {
        'B02': 'blue',
        'B03': 'green',
        'B04': 'red',
        'B05': 'red_edge_1',
        'B06': 'red_edge_2',
        'B07': 'red_edge_3',
        'B08': 'nir',
        'B8A': 'nir_2',
        'B11': 'swir_1',
        'B12': 'swir_2'
}

# S2 data is stored as uint16
s2_gain_factor = 0.0001


def _check_band_selection(
        band_selection: List[str]
    ) -> Dict[str, str]:
    """
    Returns the band mapping dictionary including only the
    user-selected bands

    :param band_selection:
        list of selected bands
    :return:
        band mapping dict with selected bands
    """

    # check how many bands were selected
    band_selection_dict = dict((k, s2_band_mapping[k]) for k in band_selection)
    return dict.fromkeys(band_selection_dict.values())



def read_from_bandstack(
        fname_bandstack: Path,
        in_file_aoi: Optional[Path] = None,
        full_bounding_box_only: Optional[bool] = False,
        int16_to_float: Optional[bool] = True,
        band_selection: Optional[List[str]] = list(s2_band_mapping.keys())
    ) -> Dict[str, np.array]:
    """
    Reads Sentinel-2 spectral bands from a band-stacked geoTiff file
    using the band description to extract the required spectral band
    and store them in a dict with the required band names.

    :param fname_bandstack:
        file-path to the bandstacked geoTiff containing the Sentinel-2
        bands in a single spatial resolution
    :param in_file_aoi:
        vector file (e.g., ESRI shapefile or geojson) defining geometry/ies
        (polygon(s)) for which to extract the Sentinel-2 data. Can contain
        one to many features.
    :param full_bounding_box_only:
        if set to False, will only extract the data for those geometry/ies
        defined in in_file_aoi. If set to False, returns the data for the
        full extent (hull) of all features (geometries) in in_file_aoi.
    :param int16_to_float:
        if True (Default) converts the original UINT16 Sentinel-2 data
        to numpy floats ranging between 0 and 1. Set to False if you
        want to keep the UINT16 datatype and original value range.
    :param band_selection:
        selection of Sentinel-2 bands to read. Per default, all 10 and
        20m bands are processed. If you wish to read less, specify the
        band names accordingly, e.g., ['B02','B03','B04'] to read only the
        VIS bands.
    :return:
        dictionary with numpy arrays of the spectral bands as well as
        two entries denoting the geo-referencation information and bounding
        box in the projection of the input satellite data
    """

    # check which bands were selected
    s2_band_data = _check_band_selection(band_selection=band_selection)

    # check bounding box
    masking = False
    if in_file_aoi is not None:
        masking = True
        gdf_aoi = check_aoi_geoms(
            in_file_aoi=in_file_aoi,
            fname_sat=fname_bandstack,
            full_bounding_box_only=full_bounding_box_only
        )

    with rio.open(fname_bandstack, 'r') as src:
        # get geo-referencation information
        meta = src.meta
        # and bounds which are helpful for plotting
        bounds = src.bounds
        # read relevant bands and store them in dict
        band_names = src.descriptions
        for idx, band_name in enumerate(band_names):
            if band_name in band_selection:
                if not masking:
                    s2_band_data[s2_band_mapping[band_name]] = src.read(idx+1)
                else:
                    s2_band_data[s2_band_mapping[band_name]], out_transform = rio.mask.mask(
                        src,
                        gdf_aoi.geometry,
                        crop=True, 
                        all_touched=True, # IMPORTANT!
                        indexes=idx+1,
                        filled=False
                    )
                    # update meta dict to the subset
                    meta.update(
                        {
                            'height': s2_band_data[s2_band_mapping[band_name]].shape[0],
                            'width': s2_band_data[s2_band_mapping[band_name]].shape[1], 
                            'transform': out_transform
                         }
                    )
                    # and bounds
                    left = out_transform[2]
                    top = out_transform[5]
                    right = left + meta['width'] * out_transform[0]
                    bottom = top + meta['height'] * out_transform[4]
                    bounds = BoundingBox(left=left, bottom=bottom, right=right, top=top)
 
                # convert and rescale to float if selected
                if int16_to_float:
                    s2_band_data[s2_band_mapping[band_name]] = \
                        s2_band_data[s2_band_mapping[band_name]].astype(float) * \
                        s2_gain_factor

    # meta and bounds are saved as additional items of the dict
    meta.update(
        {'count': len(s2_band_data)}
    )
    s2_band_data['meta'] = meta
    s2_band_data['bounds'] = bounds

    return s2_band_data


def read_from_safe(
        fname_safe: Path,
        processing_level: ProcessingLevels,
        in_file_aoi: Optional[Path] = None,
        full_bounding_box_only: Optional[bool] = False,
        int16_to_float: Optional[bool] = True,
        band_selection: Optional[List[str]] = list(s2_band_mapping.keys())
    ) -> Dict[str, np.array]:
    """
    Reads Sentinel-2 spectral bands from a band-stacked geoTiff file
    using the band description to extract the required spectral band
    and store them in a dict with the required band names. The spectral
    bands are kept in the original spatial resolution.

    :param fname_safe:
        file-path to the .SAFE directory containing Sentinel-2 data in
        L1C or L2A processing level
    :param processing_level:
        specification of the processing level of the Sentinel-2 data
    :param spatial resolution:
        target spatial resolution. Sentinel-2 bands not having the
        target spatial resolution are resampled on the fly...
    :param in_file_aoi:
        vector file (e.g., ESRI shapefile or geojson) defining geometry/ies
        (polygon(s)) for which to extract the Sentinel-2 data. Can contain
        one to many features.
    :param full_bounding_box_only:
        if set to False, will only extract the data for those geometry/ies
        defined in in_file_aoi. If set to False, returns the data for the
        full extent (hull) of all features (geometries) in in_file_aoi.
    :param int16_to_float:
        if True (Default) converts the original UINT16 Sentinel-2 data
        to numpy floats ranging between 0 and 1. Set to False if you
        want to keep the UINT16 datatype and original value range.
    :param band_selection:
        selection of Sentinel-2 bands to read. Per default, all 10 and
        20m bands are processed. If you wish to read less, specify the
        band names accordingly, e.g., ['B02','B03','B04'] to read only the
        VIS bands.
    :return:
        dictionary with numpy arrays of the spectral bands as well as
        three entries denoting the geo-referencation information and bounding
        box in the projection of the input satellite data as well as the
        spatial resolution information of the extracted bands
    """

    # TODO
    pass

if __name__ == '__main__':

    in_file_aoi = Path('/mnt/ides/Lukas/04_Work/ESCH_2021/ZH_Polygons_2020_ESCH_EPSG32632.shp')

    # read bands from bandstack
    band_selection = ['B02','B03', 'B04']
    testdata = Path('/mnt/ides/Lukas/04_Work/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    band_dict = read_from_bandstack(
        fname_bandstack=testdata,
        in_file_aoi=in_file_aoi,
        band_selection=band_selection
    )

    # read bands from .SAFE directories
    # L1C case
    testdata = Path('/mnt/ides/Lukas/04_Work/ESCH_2021/S2A/S2A_MSIL1C_20180326T103021_N0206_R108_T31TGL_20180326T155240.SAFE')
    
    
