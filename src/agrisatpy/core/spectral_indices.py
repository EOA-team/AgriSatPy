'''
This module contains a set of commonly used Spectral Indices (VIs).
The formula are generic by using color names. Thus, they can be applied
to different remote sensing platforms and are not bound to a predefined band
selection.
'''

import numpy as np

from typing import List, Union


class SpectralIndices(object):
    """generic spectral indices"""

    # define color names
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

    @classmethod
    def get_si_list(cls) -> List[str]:
        """
        Returns a list of implemented Spectral Indices (SIs)

        :returns:
            list of SIs currently implemented
        """

        return [x for x in dir(cls) if not x.startswith('__') and not x.endswith('__') \
                and not x.islower()]

    @classmethod
    def calc_si(
            cls,
            si: str,
            collection: dict
        ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        Calculates the selected spectral index (SI) for
        spectral band data derived from `~agrisatpy.core.RasterCollection`.
        The resulting vi is returned as ``numpy.ndarray``.

        :param si:
            name of the selected vegetation index (e.g., NDVI). Raises
            an error if the vegetation index is not implemented/ found.
        :returns:
            2d ``numpy.ndarray`` or ``np.ma.MaskedArray`` with VI values
            (depends on array-type of the input)
        """
        try:
            si_fun = eval(f'cls.{si.upper()}')
            si_data = si_fun.__call__(collection)
        except Exception as e:
            raise NotImplementedError(e)
        return si_data

    @classmethod
    def NDVI(
            cls,
            collection
        ) -> np.array:
        """
        Calculates the Normalized Difference Vegetation Index
        (NDVI) using the red and the near-infrared (NIR) channel.
    
        :param collection:
            reflectance in the 'red' and 'nir_1' channel
        :returns:
            NDVI values
        """

        nir = collection.get(cls.nir_1).values
        red = collection.get(cls.red).values
        return (nir - red) / (nir + red)

    @classmethod
    def EVI(
            cls,
            collection
        ):
        """
        Calculates the Enhanced Vegetation Index (EVI) following the formula
        provided by Huete et al. (2002)
    
        :param collection:
            reflectance in the 'blue', 'red' and 'nir_1' channel
        :returns:
            EVI values
        """
        blue = collection.get(cls.blue).values
        nir = collection.get(cls.nir_1).values
        red = collection.get(cls.red).values
        return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)

    @classmethod
    def MSAVI(
            cls,
            collection
        ) -> np.array:
        """
        Calculates the Modified Soil-Adjusted Vegetation Index
        (MSAVI). MSAVI is sensitive to the green leaf area index
        (greenLAI).
    
        :param collection:
            reflectance in the 'red' and 'nir_1' channel
        :returns:
            MSAVI values
        """
        nir = collection.get(cls.nir_1).values
        red = collection.get(cls.red).values
        return 0.5 * (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red)))

    @classmethod    
    def CI_green(
            cls,
            collection
        ) -> np.array:
        """
        Calculates the green chlorophyll index (CI_green).
        It is sensitive to canopy chlorophyll concentration (CCC).
    
        :param collection:
            reflectance in the 'green' and 'nir_1' channel
        :returns:
            CI-green values
        """

        nir = collection.get(cls.nir_1).values
        green = collection.get(cls.green).values
        return (nir / green) - 1

    @classmethod
    def TCARI_OSAVI(
            cls,
            collection
        ) -> np.array:
        """
        Calculates the ratio of the Transformed Chlorophyll Index (TCARI)
        and the Optimized Soil-Adjusted Vegetation Index (OSAVI). It is sensitive
        to changes in the leaf chlorophyll content (LCC).
    
        :param collection:
            reflectance in the 'green', 'red', 'red_edge_1', 'red_edge_3' channel
        :returns:
            TCARI/OSAVI values
        """

        green = collection.get(cls.green).values
        red = collection.get(cls.red).values
        red_edge_1 = collection.get(cls.red_edge_1).values
        red_edge_3 = collection.get(cls.red_edge_3).values
    
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
            collection
        ) -> np.array:
        """
        Calculates the Normalized Difference Red Edge (NDRE). It extends
        the capabilities of the NDVI for middle and late season crops.
    
        :param collection:
            reflectance in the 'red_edge_1' and 'red_edge_3' channel
        :returns:
            NDRE values
        """
        red_edge_1 = collection.get(cls.red_edge_1)
        red_edge_3 = collection.get(cls.red_edge_3)
        return (red_edge_3 - red_edge_1) / (red_edge_3 + red_edge_1)

    @classmethod
    def MCARI(
            cls,
            collection
        ):
        """
        Calculates the Modified Chlorophyll Absorption Ratio Index (MCARI).
        It is sensitive to leaf chlorophyll concentration (LCC).
    
        :param collection:
            refletcnace in the 'green', 'red', and 'red_edge_1' channel
        :returns:
            MCARI values
        """
        green = collection.get(cls.green)
        red = collection.get(cls.red)
        red_edge_1 = collection.get(cls.red_edge_1)
        return ((red_edge_1 - red) - 0.2 * (red_edge_1 - green)) * (red_edge_1 / red)

    @classmethod    
    def BSI(
            cls,
            collection
        ):
        """
        Calculates the Bare Soil Index (BSI).
    
        :param collection:
            reflectance in the 'blue', 'red', 'nir_1' and 'swir_1' channel
        :returns:
            BSI values
        """
        blue = collection.get(cls.blue)
        red = collection.get(cls.red)
        nir = collection.get(cls.nir_1)
        swir_1 = collection.get(cls.swir_1)
        return ((swir_1 + red) - (nir + blue)) / ((swir_1 + red) + (nir + blue))
