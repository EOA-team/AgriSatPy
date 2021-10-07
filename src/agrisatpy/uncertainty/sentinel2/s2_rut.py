'''
Created on Oct 6, 2021

@author: Lukas Graf
'''

import math
import numpy as np
from typing import Optional
from typing import List


# define constants from S2-RUT
# https://github.com/senbox-org/snap-rut/blob/master/src/main/python/s2_l1_rad_conf.py
# (Oct-06-2021)
Lref = [129.11, 128, 128, 108, 74.6, 68.23, 66.70, 103, 52.39, 8.77, 6, 4, 1.70]

u_stray_rand_all = {'Sentinel-2A': [0.1, 0.1, 0.08, 0.12, 0.44, 0.16, 0.2, 0.2, 0.04, 0.8, 0, 0, 0],
                    'Sentinel-2B': [0.1, 0.1, 0.08, 0.12, 0.44, 0.16, 0.2, 0.2, 0.04, 0.8, 0, 0, 0]}

# units in W.m-2.sr-1.μm-1
u_xtalk_all = {'Sentinel-2A': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                    'Sentinel-2B': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]}

u_DS_all = {'Sentinel-2A': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.24, 0.12, 0.16],
                    'Sentinel-2B': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.24, 0.12, 0.16]}

# values from ICCDB (S2A at S2_OPER_MSI_DIFF___20150519T000000_0009.xml and
# S2B at S2_OPER_MSI_DIFF___20160415T000000_0001.xml)
u_diff_absarray = {'Sentinel-2A': [1.09, 1.08, 0.84, 0.73, 0.68, 0.97, 0.83, 0.81, 0.88, 0.97, 1.39, 1.39, 1.58],
                    'Sentinel-2B': [1.16, 1.00, 0.79, 0.70, 0.85, 0.77, 0.80, 0.80, 0.85, 0.66, 1.70, 1.46, 2.13]}

u_diff_temp_rate = {'Sentinel-2A': [0.15, 0.09, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'Sentinel-2B': [0.15, 0.09, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}


class S2RutAlgo:
    """
    Algorithm for the Sentinel-2 Radiometric Uncertainty Tool (RUT) developed
    by Gorrono et al. (2017).

    :param a:
        physical gain factor for the selected spectral band from datastrip
        metadata
    :param e_sun:
        solar irradiance in the selected spectral band at the corresponding
        image acquisition date
    :param u_sun:
        conversion factor to account for variations in the sun-earth distance
    :param tecta:
        pixel solar zenith angle (deg)
    :param alpha:
        datastrip noise model parameter alpha
    :param beta:
        datastrip noise model parameter beta
    :param u_diff_temp:
        temporal degradation uncertainty contributor per band and acqiusition
        date
    :param k:
        uncertainty coverage factor (number of standard uncertainties). Default
        is 2 based on recommendations in GUM.
    :param quant:
        quantification value. Defaults to 10000.
    :param u_diff_cos:
        0.13° diffuser planarity/micro as in (AIRBUS 2015). Assumed same for S2A/S2B.
    :param u_diff_k:
        [%] as a conservative residual (AIRBUS 2015). Assumed same for S2A/S2B.
    :param u_ADC:
        analog-digital quantisation uncertainty contribution [DN] (rectangular distribution)
    :param u_gamma:
        gamma knowledge contributor. Default is 0.4.
    :param unc_select:
        list of booleans with user selected uncertainty sources. Order:
        ["INSTRUMENT_NOISE", "OOF_STRAYLIGHT-SYSTEMATIC", "OOF_STRAYLIGHT-RANDOM", "CROSSTALK",
        "ADC_QUANTISATION", "DS_STABILITY", "GAMMA_KNOWLEDGE", "DIFFUSER-ABSOLUTE_KNOWLEDGE",
        "DIFFUSER-TEMPORAL_KNOWLEDGE", "DIFFUSER-COSINE_EFFECT", "DIFFUSER-STRAYLIGHT_RESIDUAL",
        "L1C_IMAGE_QUANTISATION"]
    """

    def __init__(
            self,
            a: float,
            e_sun: float,
            u_sun: float,
            tecta: float,
            alpha: float,
            beta: float,
            u_diff_temp: float,
            k: Optional[int]=2,
            quant: Optional[float]=10000.0,
            u_diff_cos: Optional[float]=0.4,
            u_diff_k: Optional[float]=0.3,
            u_ADC: Optional[float]=0.5,
            u_gamma: Optional[float]=0.4,
            unc_select: Optional[List[bool]]=[True,
                                              True,
                                              True,
                                              True,
                                              True,
                                              True,
                                              True,
                                              True,
                                              True,
                                              True,
                                              True,
                                              True]
        ):
        # uncertainty values for DS and Abs.cal
        self.a = a
        self.e_sun = e_sun
        self.u_sun = u_sun
        self.tecta = tecta
        self.alpha = alpha
        self.beta = beta
        self.u_diff_temp = u_diff_temp

        self.quant = quant
        self.k = k
        self.u_diff_cos = u_diff_cos
        self.u_diff_k = u_diff_k
        self.u_ADC = u_ADC
        self.u_gamma = u_gamma
        self.unc_select = unc_select


    def unc_calculation(
            self,
            band_data: np.array,
            band_id: int,
            spacecraft: str
        ) -> List[float]:
        """
        This function represents the core of the RUTv1.
        It takes as an input the pixel data of a specific band and tile in
        a S2-L1C product and produces an image with the same dimensions that
        contains the radiometric uncertainty of each pixel reflectance factor.

        The steps and its numbering is equivalent to the RUT-DPM. This document
        can be found in the tool github. Also there a more detailed explanation
        of the theoretical background can be found.

        :param band_data:
            list with the quantized L1C reflectance pixels of a band (flattened; 1-d)
        :param band_id:
            zero-based index of the band
        :param spacecraft:
            satellite for which uncertainty is calculated. Valid values:
            "Sentinel-2A" and "Sentinel-2B"
        :return:
            list of u_int8 with uncertainty associated to each pixel.
        """

        #######################################################################
        # 1.    Undo reflectance conversion
        #######################################################################

        # Replace the reflectance factors by CN values
        # cn = (self.a * self.e_sun * self.u_sun * math.cos(math.radians(self.tecta)) / math.pi) * band_data
        cn = (self.a * self.e_sun * self.u_sun * np.cos(np.radians(self.tecta)) / math.pi) * band_data

        #######################################################################
        # 2.    Orthorectification process
        #######################################################################

        # TBD. Here both terms will be used with no distinction.

        #######################################################################
        # 3.    L1B uncertainty contributors: raw and dark signal
        #######################################################################

        if self.unc_select[0]:
            u_noise = 100 * np.sqrt(self.alpha ** 2 + self.beta * cn) / cn
        else:
            u_noise = 0

        # [W.m-2.sr-1.μm-1] 0.3%*Lref all bands (AIRBUS 2015) and (AIRBUS 2014)
        if self.unc_select[1]:
            u_stray_sys = 0.3 * Lref[band_id] / 100
        else:
            u_stray_sys = 0

        if self.unc_select[2]:
            u_stray_rand = u_stray_rand_all[spacecraft][band_id]  # [%](AIRBUS 2015) and (AIRBUS 2012)
        else:
            u_stray_rand = 0

        if self.unc_select[3]:
            u_xtalk = u_xtalk_all[spacecraft][band_id]  # [W.m-2.sr-1.μm-1](AIRBUS 2015)
        else:
            u_xtalk = 0

        if not self.unc_select[4]:
            self.u_ADC = 0  # predefined but updated to 0 if deselected by user

        if self.unc_select[5]:
            u_DS = u_DS_all[spacecraft][band_id]
        else:
            u_DS = 0

        #######################################################################
        # 4.    L1B uncertainty contributors: gamma correction
        #######################################################################

        if self.unc_select[6]:
            self.u_gamma = 0.4  # [%] (AIRBUS 2015)
        else:
            self.u_gamma = 0

        #######################################################################
        # 5.    L1C uncertainty contributors: absolute calibration coefficient
        #######################################################################

        if self.unc_select[7]:
            u_diff_abs = u_diff_absarray[spacecraft][band_id]
        else:
            u_diff_abs = 0

        if not self.unc_select[8]:
            self.u_diff_temp = 0  # calculated in s2_rut.py. Updated to 0 if deselected by user

        if not self.unc_select[9]:
            self.u_diff_cos = 0  # predefined but updated to 0 if deselected by user

        if not self.unc_select[10]:
            self.u_diff_k = 0  # predefined but updated to 0 if deselected by user

        #######################################################################
        # 6.    L1C uncertainty contributors: reflectance conversion
        #######################################################################

        if self.unc_select[11]:
            u_ref_quant = 100 * (0.5 / math.sqrt(3)) / (self.quant * band_data)  # [%]scaling 0-1 in steps number=quant
        else:
            u_ref_quant = 0

        #######################################################################        
        # 7.    Combine uncertainty contributors
        #######################################################################        
        # NOTE: no gamma propagation for RUTv1!!!
        # values given as percentages. Multiplied by 10 and saved to 1 byte(uint8)
        # Clips values to 0-250 --> uncertainty >=25%  assigns a value 250.
        # Uncertainty <=0 represents a processing error (uncertainty is positive)
        u_adc = (100 * self.u_ADC / math.sqrt(3)) / cn
        u_ds = (100 * u_DS) / cn
        u_stray = np.sqrt(u_stray_rand ** 2 + ((100 * self.a * u_xtalk) / cn) ** 2)
        u_diff = math.sqrt(u_diff_abs ** 2 + self.u_diff_cos ** 2 + self.u_diff_k ** 2)
        u_1sigma = np.sqrt(u_ref_quant ** 2 + self.u_gamma ** 2 + u_stray ** 2 + u_diff ** 2 +
                           u_noise ** 2 + u_adc ** 2 + u_ds ** 2)
        u_expand = np.round(10 * (self.u_diff_temp + ((100 * self.a * u_stray_sys) / cn) + self.k * u_1sigma))
        u_ref = np.uint8(np.clip(u_expand, 0, 250))

        return u_ref
