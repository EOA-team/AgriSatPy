'''
Created on Nov 24, 2021

@author:    Lukas Graf

@purpose:   This module contains a set of commonly used Vegetation
            Indices (VIs). The formula are generic, i.e., they can be
            applied to different remote sensing platforms. 
'''

import numpy as np
import rasterio as rio
import rasterio.mask
import geopandas as gpd

from rasterio.coords import BoundingBox
from shapely.geometry import box
from pathlib import Path
from typing import Optional
from typing import Dict


class VegetationIndices(object):
    """generic vegetation indices"""

    # define spectral band names
    blue = 'blue'
    green = 'green'
    red = 'red'
    red_edge_1 = 'red_edge_1'
    red_edge_2 = 'red_edge_2'
    red_edge_3 = 'red_edge_3'
    nir = 'nir'
    swir_1 = 'swir_1'
    swir_2 = 'swir_2'


    @classmethod
    def NDVI(
            cls,
            **kwargs
        ) -> np.array:
        """
        Calculates the Normalized Difference Vegetation Index
        (NDVI) using the red and the near-infrared (NIR) channel.
    
        :param kwargs:
            reflectance in the 'red' and the 'nir' channel
        :return:
            NDVI values
        """

        nir = kwargs.get(cls.nir)
        red = kwargs.get(cls.red)
        return (nir - red) / (nir + red)


    @classmethod
    def EVI(
            cls,
            blue: np.array,
            red: np.array,
            nir: np.array
        ):
        """
        Calculates the Enhanced Vegetation Index (EVI) following the formula
        provided by Huete et al. (2002)
    
        :param blue:
            reflectance in the blue band (Sentinel-2 B02)
        :param red:
            reflectance in the red band (Sentinel-2 B04)
        :param nir:
            reflectance in the near infrared band (Sentinel-2 B08)
            spectrum
        :return:
            EVI values
        """
    
        return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)


    @classmethod
    def AVI(
            cls,
            red: np.array,
            nir: np.array
        ) -> np.array:
        """
        Calculates the Advanced Vegetation Index (AVI) using Sentinel-2 bands
        4 (red) and 8 (nir)
    
        :param red:
            reflectance in the red band (Sentinel-2 B04)
        :param nir:
            reflectance in the near infrared band (Sentinel-2 B08)
            spectrum
        :return:
            AVI values
        """
    
        expr = nir * (1 - red) * (nir - red)
        return np.power(expr, (1./3.))


    @classmethod
    def MSAVI(
            cls,
            red: np.array,
            nir: np.array
        ) -> np.array:
        """
        Calculates the Modified Soil-Adjusted Vegetation Index
        (MSAVI). MSAVI is sensitive to the green leaf area index
        (greenLAI).
    
        :param red:
            reflectance in the red band (Sentinel-2 B04)
        :param red_edge_3:
           reflectance in the NIR band (Sentinel-2 B08)
        :return:
            MSAVI values
        """
        return 0.5 * (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red)))


    @classmethod    
    def CI_green(
            cls,
            green: np.array,
            nir: np.array 
        ) -> np.array:
        """
        Calculates the green chlorophyll index (CI_green) using
        Sentinel-2 bands 3 (green) and 8 (nir) as suggested by Clevers
        et al. (2017, doi:10.3390/rs9050405). It is sensitive to
        canopy chlorophyll concentration (CCC).
    
        :param green:
            reflectance in the green band (Sentinel-2 B03)
        :param nir:
            reflectance in the NIR band (Sentinel-2 B08)
        """
    
        return (nir / green) - 1
    

    @classmethod
    def TCARI_OSAVI(
            cls,
            green: np.array,
            red: np.array,
            red_edge_1: np.array,
            red_edge_3: np.array
        ) -> np.array:
        """
        Calculates the ratio of the Transformed Chlorophyll Index (TCARI)
        and the Optimized Soil-Adjusted Vegetation Index (OSAVI). It is sensitive
        to changes in the leaf chlorophyll content (LCC). The Sentinel-2 band
        selection follows the paper by Clevers et al. (2017, doi:10.3390/rs9050405)
    
        :param green:
            reflectance in the green band (Sentinel-2 B03)
        :param red:
            reflectance in the green band (Sentinel-2 B04)
        :param red_edge_1:
            reflectance in the red edge 1 band (Sentinel-2 B05)
        :param nir:
           reflectance in the red edge 3 band (Sentinel-2 B07)
        :return:
            TCARI/OSAVI values
        """
    
        TCARI = 3*((red_edge_1 - red) - 0.2*(red_edge_1 - green) * (red_edge_1 / red))
        OSAVI = (1 + 0.16) * (red_edge_3 - red) / (red_edge_3 + red + 0.16)
        tcari_osavi = TCARI/OSAVI
        # clip values to range between 0 and 1 (division by zero might cause infinity)
        tcari_osavi[tcari_osavi < 0.] = 0.
        tcari_osavi[tcari_osavi > 1.] = 1.
        return tcari_osavi


    @classmethod
    def NDRE(
            cls,
            red_edge_1: np.array,
            red_edge_3: np.array
        ) -> np.array:
        """
        Calculates the Normalized Difference Red Edge (NDRE). It extends
        the capabilities of the NDVI for middle and late season crops.
    
        :param red_edge_1:
            reflectance in the red edge 1 band (Sentinel-2 B05)
        :param red_edge_3:
            reflectance in the red edge 3 band (Sentinel-2 B07)
        """
        
        return (red_edge_3 - red_edge_1) / (red_edge_3 + red_edge_1)
    

    @classmethod
    def MCARI(
            cls,
            green: np.array,
            red: np.array,
            red_edge_1: np.array
        ):
        """
        Calculates the Modified Chlorophyll Absorption Ratio Index (MCARI)
        using Sentinel-2 bands 3 (green), 4 (red), and 5 (red edge 1).
        It is sensitive to leaf chlorophyll concentration (LCC).
    
        :param green:
            reflectance in the green band (Sentinel-2 B03)
        :param red:
            reflectance in the red band (Sentinel-2 B04)
        :param nir:
            reflectance in the NIR band (Sentinel-2 B08)
        """
    
        return ((red_edge_1 - red) - 0.2 * (red_edge_1 - green)) * (red_edge_1 / red)


    @classmethod    
    def BSI(
            blue: np.array,
            red: np.array,
            nir: np.array,
            swir_1: np.array
        ):
        """
        Calculates the Bare Soil Index (BSI) using Sentinel-2 bands
        2 (blue), 4 (red), and 11 (SWIR 1)
    
        :param green:
            reflectance in the blue band (Sentinel-2 B02)
        :param red:
            reflectance in the red band (Sentinel-2 B04)
        :param nir:
            reflectance in the NIR band (Sentinel-2 B08)
        :param swir_1:
            reflectance in the SWIR band (Sentinel-2 11)
        """
    
        return ((swir_1 + red) - (nir + blue)) / ((swir_1 + red) + (nir + blue))


class Sentinel2VegetationIndices(VegetationIndices):
    """Sentinel-2 derived Vegetation Indices"""

    s2_band_mapping = {
        'B02': 'blue',
        'B03': 'green',
        'B04': 'red',
        'B05': 'red_edge_1',
        'B06': 'red_edge_2',
        'B07': 'red_edge_3',
        'B08': 'nir',
        'B11': 'swir_1'
    }

    # S2 data is stored as uint16
    s2_gain_factor = 0.0001

    @classmethod
    def read_from_bandstack(
            cls,
            fname_bandstack: Path,
            in_file_aoi: Optional[Path] = None,
            full_bounding_box_only: Optional[bool] = False,
            int16_to_float: Optional[bool] = True 
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
        :return:
            dictionary with numpy arrays of the spectral bands as well as
            two entries denoting the geo-referencation information and bounding
            box in the projection of the input satellite data
        """

        s2_band_data = dict.fromkeys(cls.s2_band_mapping.values())

        # check for vector file defining AOI
        masking = False
        if in_file_aoi is not None:

            # read AOI into a geodataframe
            gdf_aoi = gpd.read_file(in_file_aoi)
            # check if the spatial reference systems match
            sat_crs = rio.open(fname_bandstack).crs
            # reproject vector data if necessary
            if gdf_aoi.crs != sat_crs:
                gdf_aoi.to_crs(sat_crs, inplace=True)
            # consequently, masking is necessary
            masking = True

            # if the the entire bounding box shall be extracted
            # we need the hull encompassing all geometries in gdf_aoi
            if full_bounding_box_only:
                bbox = box(*gdf_aoi.total_bounds)
                gdf_aoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bbox))


        with rio.open(fname_bandstack, 'r') as src:
            # get geo-referencation information
            meta = src.meta
            # and bounds which are helpful for plotting
            bounds = src.bounds
            # read relevant bands and store them in dict
            band_names = src.descriptions
            for idx, band_name in enumerate(band_names):
                if band_name in list(cls.s2_band_mapping.keys()):
                    if not masking:
                        s2_band_data[cls.s2_band_mapping[band_name]] = src.read(idx+1)
                    else:
                        s2_band_data[cls.s2_band_mapping[band_name]], out_transform = rio.mask.mask(
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
                                'height': s2_band_data[cls.s2_band_mapping[band_name]].shape[0],
                                'width': s2_band_data[cls.s2_band_mapping[band_name]].shape[1], 
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
                        s2_band_data[cls.s2_band_mapping[band_name]] = \
                            s2_band_data[cls.s2_band_mapping[band_name]].astype(float) * \
                            cls.s2_gain_factor

        # meta and bounds are saved as additional items of the dict
        meta.update(
            {'count': len(s2_band_data)}
        )
        s2_band_data['meta'] = meta
        s2_band_data['bounds'] = bounds

        return s2_band_data
        

    

if __name__ == '__main__':
    
    testdata = Path('/mnt/ides/Lukas/04_Work/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    in_file_aoi = Path('/run/media/graflu/ETH-KP-SSD6/SAT/Uncertainty/scripts_paper_uncertainty/shp/ZH_Polygons_2019_EPSG32632_selected-crops.shp')

    Sentinel2VegetationIndices.read_from_bandstack(
        fname_bandstack=testdata,
        in_file_aoi=None
    )

