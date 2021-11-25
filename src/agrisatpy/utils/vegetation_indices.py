'''
Created on Nov 24, 2021

@author:    Lukas Graf

@purpose:   This module contains a set of commonly used Vegetation
            Indices (VIs). The formula are generic, i.e., they can be
            applied to different remote sensing platforms. 
'''

import numpy as np
from typing import Dict

from agrisatpy.utils.io import Sat_Data_Reader


class VegetationIndices(object):
    """generic vegetation indices"""

    # define spectral band names
    blue = 'blue'
    green = 'green'
    red = 'red'
    red_edge_1 = 'red_edge_1'
    red_edge_2 = 'red_edge_2'
    red_edge_3 = 'red_edge_3'
    nir_1 = 'nir_1'
    nir_2 = 'nir_2'
    swir_1 = 'swir_1'
    swir_2 = 'swir_2'


    def __init__(self, reader: Sat_Data_Reader):
        """
        :param reader:
            any class inheriting from Sat_Data_Reader
        """
        # we only take the readers data attribute
        self._band_data = reader.data

    def calc_vi(self, vi: str) -> np.array:
        """
        Calculates the selected vegetation index (VI) for
        spectral band data derived from agrisatpy.utils.io

        :param vi:
            name of the selected vegetation index (e.g., NDVI)
        :return:
            vegetation index values
        """
        
        return eval(f'self.{vi.upper()}(**self._band_data)')


    @classmethod
    def NDVI(
            cls,
            **kwargs
        ) -> np.array:
        """
        Calculates the Normalized Difference Vegetation Index
        (NDVI) using the red and the near-infrared (NIR) channel.
    
        :param kwargs:
            reflectance in the 'red' and 'nir_1' channel
        :return:
            NDVI values
        """

        nir = kwargs.get(cls.nir_1)
        red = kwargs.get(cls.red)
        return (nir - red) / (nir + red)


    @classmethod
    def EVI(
            cls,
            **kwargs
        ):
        """
        Calculates the Enhanced Vegetation Index (EVI) following the formula
        provided by Huete et al. (2002)
    
        :param kwargs:
            reflectance in the 'blue', 'red' and 'nir_1' channel
        :return:
            EVI values
        """

        blue = kwargs.get(cls.blue)
        nir = kwargs.get(cls.nir_1)
        red = kwargs.get(cls.red)
        return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)


    @classmethod
    def AVI(
            cls,
            **kwargs
        ) -> np.array:
        """
        Calculates the Advanced Vegetation Index (AVI)
    
        :param kwargs:
            reflectance in the 'red' and 'nir_1' channel
        :return:
            AVI values
        """

        nir = kwargs.get(cls.nir_1)
        red = kwargs.get(cls.red)
        expr = nir * (1 - red) * (nir - red)
        return np.power(expr, (1./3.))


    @classmethod
    def MSAVI(
            cls,
            **kwargs
        ) -> np.array:
        """
        Calculates the Modified Soil-Adjusted Vegetation Index
        (MSAVI). MSAVI is sensitive to the green leaf area index
        (greenLAI).
    
        :param kwargs:
            reflectance in the 'red' and 'nir_1' channel
        :return:
            MSAVI values
        """

        nir = kwargs.get(cls.nir_1)
        red = kwargs.get(cls.red)
        return 0.5 * (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red)))


    @classmethod    
    def CI_green(
            cls,
            **kwargs
        ) -> np.array:
        """
        Calculates the green chlorophyll index (CI_green).
        It is sensitive to canopy chlorophyll concentration (CCC).
    
        :param kwargs:
            reflectance in the 'green' and 'nir_1' channel
        :return:
            CI-green values
        """

        nir = kwargs.get(cls.nir_1)
        green = kwargs.get(cls.green)
        return (nir / green) - 1
    

    @classmethod
    def TCARI_OSAVI(
            cls,
            **kwargs
        ) -> np.array:
        """
        Calculates the ratio of the Transformed Chlorophyll Index (TCARI)
        and the Optimized Soil-Adjusted Vegetation Index (OSAVI). It is sensitive
        to changes in the leaf chlorophyll content (LCC).
    
        :param kwargs:
            reflectance in the 'green', 'red', 'red_edge_1', 'red_edge_3' channel
        :return:
            TCARI/OSAVI values
        """

        green = kwargs.get(cls.green)
        red = kwargs.get(cls.red)
        red_edge_1 = kwargs.get(cls.red_edge_1)
        red_edge_3 = kwargs.get(cls.red_edge_3)
    
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
            **kwargs
        ) -> np.array:
        """
        Calculates the Normalized Difference Red Edge (NDRE). It extends
        the capabilities of the NDVI for middle and late season crops.
    
        :param kwargs:
            reflectance in the 'red_edge_1' and 'red_edge_3' channel
        :return:
            NDRE values
        """

        red_edge_1 = kwargs.get(cls.red_edge_1)
        red_edge_3 = kwargs.get(cls.red_edge_3)
        return (red_edge_3 - red_edge_1) / (red_edge_3 + red_edge_1)


    @classmethod
    def MCARI(
            cls,
            **kwargs
        ):
        """
        Calculates the Modified Chlorophyll Absorption Ratio Index (MCARI).
        It is sensitive to leaf chlorophyll concentration (LCC).
    
        :param **kwargs:
            refletcnace in the 'green', 'red', and 'red_edge_1' channel
        """

        green = kwargs.get(cls.green)
        red = kwargs.get(cls.red)
        red_edge_1 = kwargs.get(cls.red_edge_1)
        return ((red_edge_1 - red) - 0.2 * (red_edge_1 - green)) * (red_edge_1 / red)


    @classmethod    
    def BSI(
            cls,
            **kwargs
        ):
        """
        Calculates the Bare Soil Index (BSI).
    
        :param kwargs:
            reflectance in the 'blue', 'red', 'nir_1' and 'swir_1' channel
        """

        blue = kwargs.get(cls.blue)
        red = kwargs.get(cls.red)
        nir = kwargs.get(cls.nir_1)
        swir_1 = kwargs.get(cls.swir_1)
        return ((swir_1 + red) - (nir + blue)) / ((swir_1 + red) + (nir + blue))


if __name__ == '__main__':

    from pathlib import Path
    from agrisatpy.utils.io.sentinel2 import S2_Band_Reader
    
    testdata = Path('/mnt/ides/Lukas/04_Work/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    in_file_aoi = Path('/mnt/ides/Lukas/04_Work/ESCH_2021/ZH_Polygons_2020_ESCH_EPSG32632.shp')

    reader = S2_Band_Reader()
    reader.read_from_bandstack(
        fname_bandstack=testdata,
        in_file_aoi=in_file_aoi
    )

    vi_s2 = VegetationIndices(reader=reader)
    reader.data['ndvi'] = vi_s2.calc_vi('NDVI')
    reader.data['evi'] = vi_s2.calc_vi('EVI')

