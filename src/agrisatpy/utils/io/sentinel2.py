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

from matplotlib.pyplot import Figure
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
from agrisatpy.utils.sentinel2 import get_S2_sclfile
from agrisatpy.utils.constants.sentinel2 import band_resolution
from agrisatpy.utils.io import Sat_Data_Reader
from agrisatpy.utils.exceptions import BandNotFoundError


class S2_Band_Reader(Sat_Data_Reader):
    """
    Class for storing Sentinel-2 band data read from bandstacks or
    .SAFE archives (L1C and L2A level) overwriting methods inherited
    from `~agrisatpy.utils.io.Sat_Data_Reader`.
    """

    def __init__(self, *args, **kwargs):
        Sat_Data_Reader.__init__(self, *args, **kwargs)

    @staticmethod
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

    def plot_scl(
            self,
            colormap: Optional[str]='Accent'
        ) -> Figure:
        """
        Wrapper around `agrisatpy.utils.io` to plot the Scene Classification
        Layer available from the L2A processing level. Raises an error if
        the band is not available

        :param colormap:
            matplotlib named colormap to use for visualization (Def: 'Accent')
        :return:
            matplotlib figure object with the SCL band data
            plotted as map
        """

        try:
            return self.plot_band(
                band_name='scl',
                colormap=colormap,
                discrete_values=True
            )
        except Exception as e:
            raise BandNotFoundError(f'Could not plot SCL: {e}')


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
        :return:
            dictionary with band names and corresponding band data as np.array.
            In addition, two entries in the dict provide information about the
            geo-localization ('meta') and the bounding box ('bounds') in image
            coordinates
        """
    
        # check which bands were selected
        self.data = self._check_band_selection(band_selection=band_selection)
    
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
            band_selection: Optional[List[str]] = list(s2_band_mapping.keys()),
            read_scl: Optional[bool] = True
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
        :param read_scl:
            if True (Default), reads the scene classification layer available in
            L2A processing level. If the processing level is lower (L1C) this
            option is ignored
        :return:
            dictionary with band names and corresponding band data as np.array.
            In addition, two entries in the dict provide information about the
            geo-localization ('meta') and the bounding box ('bounds') in image
            coordinates per band.
        """
    
        # check which bands were selected
        self.data = self._check_band_selection(band_selection=band_selection)
    
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

        # search SCL file if selected and processing level is L2A
        if is_L2A and read_scl:
            band_selection.append('SCL')
            scl_file = get_S2_sclfile(in_dir)
            # append to dataframe
            record = {
                'band_name': 'SCL',
                'band_path': str(scl_file),
                'band_resolution': 20
            }
            band_df_safe = band_df_safe.append(record, ignore_index=True)
    
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
                        # if SCL is selected, do not apply the conversion
                        if not band_name.upper() == 'SCL':
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

    # download test data (if not done yet)
    import cv2
    import requests
    from agrisatpy.downloader.sentinel2.utils import unzip_datasets

    reader = S2_Band_Reader()
    band_selection = ['B02', 'B03', 'B04', 'B05', 'B07', 'B08']
    in_file_aoi = Path('/home/graflu/git/agrisatpy/data/sample_polygons/BY_AOI_2019_MNI_EPSG32632.shp')

    # L2A testcase
    url = 'https://data.mendeley.com/public-files/datasets/ckcxh6jskz/files/e97b9543-b8d8-436e-b967-7e64fe7be62c/file_downloaded'
    
    testdata_dir = Path('/mnt/ides/Lukas/debug/S2_Data')
    testdata_fname = testdata_dir.joinpath('S2A_MSIL2A_20190524T101031_N0212_R022_T32UPU_20190524T130304.zip')
    testdata_fname_unzipped = Path(testdata_fname.as_posix().replace('.zip', '.SAFE'))

    if not testdata_fname_unzipped.exists():
        
        # download dataset
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(testdata_fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=5096):
                fd.write(chunk)

        # unzip dataset
        unzip_datasets(download_dir=testdata_dir)

    processing_level = ProcessingLevels.L2A

    reader.read_from_safe(
        in_dir=testdata_fname_unzipped,
        processing_level=processing_level,
        in_file_aoi=in_file_aoi,
        band_selection=band_selection
    )

    # check the RGB
    fig_rgb = reader.plot_rgb()

    # and the false-color near-infrared
    fig_nir = reader.plot_false_color_infrared()

    # check the scene classification layer
    fig_scl = reader.plot_scl()

    reader.resample(target_resolution=10, resampling_method=cv2.INTER_CUBIC, bands_to_exclude=['scl'])
    reader.calc_vi(vi='TCARI_OSAVI')

