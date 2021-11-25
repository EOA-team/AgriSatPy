'''
Created on Nov 24, 2021

@author:    Lukas Graf (D-USYS, ETHZ)

@purpose:   This module enables reading one or more
            spectral bands from Sentinel-2 data in either band-stacked
            (i.e., AgriSatPy derived format) or from the ESA derived
            .SAFE archive structure supporting L1C and L2A processing
            level
'''

import numpy as np
import rasterio as rio
import rasterio.mask

from rasterio.coords import BoundingBox
from pathlib import Path
from typing import Optional
from typing import Dict
from typing import List

from agrisatpy.utils.constants.sentinel2 import ProcessingLevels
from agrisatpy.utils.constants.sentinel2 import s2_band_mapping
from agrisatpy.utils.constants.sentinel2 import s2_gain_factor
from agrisatpy.utils.reprojection import check_aoi_geoms
from agrisatpy.utils.sentinel2 import get_S2_bandfiles_with_res
from agrisatpy.utils.constants.sentinel2 import band_resolution
from agrisatpy.utils.io import Sat_Data_Reader



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



class S2_Band_Reader(Sat_Data_Reader):
    """
    Class for storing Sentinel-2 band data read from bandstacks or
    .SAFE archives (L1C and L2A level)
    """

    def __init__(self, *args, **kwargs):
        Sat_Data_Reader.__init__(self, *args, **kwargs)

    def from_bandstack(self) -> bool:
        """checks if the data was read from bandstack or .SAFE archive"""
        return self._from_bandstack

    def resample(self, target_resolution):
        """
        resamples data from .SAFE archive on the fly if required
        into a user-definded spatial resolution
        """
        # TODO
        pass

    def plot_band(self, band_name: str):
        """
        plots a custom band
        """
        # TODO
        pass

    def read_from_bandstack(
            self,
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

        The method populates the self.data attribute that is a
        dictionary with the following items:
            <name-of-the-band>: <np.array> denoting the spectral band data
            <meta>:<rasterio meta> denoting the georeferencation
            <bounds>: <BoundingBox> denoting the bounding box of the band data

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
        """
    
        # check which bands were selected
        self.data = _check_band_selection(band_selection=band_selection)
    
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
                        self.data[s2_band_mapping[band_name]] = src.read(idx+1)
                    else:
                        self.data[s2_band_mapping[band_name]], out_transform = rio.mask.mask(
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
                                'height': self.data[s2_band_mapping[band_name]].shape[0],
                                'width': self.data[s2_band_mapping[band_name]].shape[1], 
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
                        self.data[s2_band_mapping[band_name]] = \
                            self.data[s2_band_mapping[band_name]].astype(float) * \
                            s2_gain_factor
    
        # meta and bounds are saved as additional items of the dict
        meta.update(
            {'count': len(self.data)}
        )
        self.data['meta'] = meta
        self.data['bounds'] = bounds

        self._from_bandstack = True


    def read_from_safe(
            self,
            in_dir: Path,
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

        The method populates the self.data attribute that is a
        dictionary with the following items:
                <name-of-the-band>: <np.array> denoting the spectral band data
                <meta>:<rasterio meta> denoting the georeferencation per band
                <bounds>: <BoundingBox> denoting the bounding box of the data per band
    
        :param in_dir:
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
        """
    
        # check which bands were selected
        self.data = _check_band_selection(band_selection=band_selection)
    
        # determine which spatial resolutions are selected
        # (based on the native spatial resolution of Sentinel-2 bands)
        band_selection_spatial_res = [x[1] for x in band_resolution.items() if x[0] in band_selection]
        resolution_selection = list(np.unique(band_selection_spatial_res))
    
        # check processing level
        is_L2A = True
        if processing_level == ProcessingLevels.L1C:
            is_L2A = False
    
        # search band files depending on processing level and spatial resolution(s)
        band_df_safe = get_S2_bandfiles_with_res(
            in_dir=in_dir,
            resolution_selection=resolution_selection,
            is_L2A=is_L2A
        )
    
        # check bounding box
        masking = False
        if in_file_aoi is not None:
            masking = True
            gdf_aoi = check_aoi_geoms(
                in_file_aoi=in_file_aoi,
                fname_sat=band_df_safe.band_path.iloc[0],
                full_bounding_box_only=full_bounding_box_only
            )
    
        # loop over selected bands and read the data into a dict
        meta_bands = {}
        bounds_bands = {}
        for band_name in band_selection:
    
            # get entry from dataframe with file-path of band
            band_safe = band_df_safe[band_df_safe.band_name == band_name]
            band_fpath = band_safe.band_path.values[0]
            
            with rio.open(band_fpath, 'r') as src:
    
                meta = src.meta
                # and bounds which are helpful for plotting
                bounds = src.bounds
                
                if not masking:
                    self.data[s2_band_mapping[band_name]] = src.read(1)
                else:
                    self.data[s2_band_mapping[band_name]], out_transform = rio.mask.mask(
                        src,
                        gdf_aoi.geometry,
                        crop=True, 
                        all_touched=True, # IMPORTANT!
                        indexes=1,
                        filled=False
                    )
                    # update meta dict to the subset
                    meta.update(
                        {
                            'height': self.data[s2_band_mapping[band_name]].shape[0],
                            'width': self.data[s2_band_mapping[band_name]].shape[1], 
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
                        self.data[s2_band_mapping[band_name]] = \
                            self.data[s2_band_mapping[band_name]].astype(float) * \
                            s2_gain_factor
    
            # store georeferencation and bounding box per band
            meta_bands[s2_band_mapping[band_name]] = meta
            bounds_bands[s2_band_mapping[band_name]] = bounds
    
        self.data['meta'] = meta_bands
        self.data['bounds'] = bounds_bands

        self._from_bandstack = False
 

if __name__ == '__main__':

    in_file_aoi = Path('/mnt/ides/Lukas/04_Work/ESCH_2021/ZH_Polygons_2020_ESCH_EPSG32632.shp')

    # read bands from bandstack
    band_selection = ['B02','B03', 'B08']
    # band_selection = None
    testdata = Path('/mnt/ides/Lukas/04_Work/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    reader = S2_Band_Reader()
    reader.read_from_bandstack(
        fname_bandstack=testdata,
        in_file_aoi=in_file_aoi,
        band_selection=band_selection
    )
    # check reader.data

    # read bands from .SAFE directories
    # L1C case
    testdata = Path('/mnt/ides/Lukas/04_Work/ESCH_2021/S2A/ESCH/S2A_MSIL1C_20210615T102021_N0300_R065_T32TMT_20210615T122505.SAFE')
    processing_level = ProcessingLevels.L1C

    reader.read_from_safe(
        in_dir=testdata,
        processing_level=processing_level,
        in_file_aoi=in_file_aoi,
        band_selection=band_selection
    )

    # L2A testcase
    # TODO: resampling of data if required/selected -> resampling module
    testdata = Path('/mnt/ides/Lukas/04_Work/ESCH_2021/S2A/ESCH/S2A_MSIL2A_20210615T102021_N0300_R065_T32TMT_20210615T131659.SAFE')
    processing_level = ProcessingLevels.L2A

    reader.read_from_safe(
        in_dir=testdata,
        processing_level=processing_level,
        in_file_aoi=in_file_aoi,
        band_selection=band_selection
    )
