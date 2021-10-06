'''
Created on Oct 6, 2021

@author: graflu
'''

import datetime

from typing import Dict
from agrisatpy.config.sentinel2 import Sentinel2


u_diff_temp_rate = {
    'Sentinel-2A': [0.15, 0.09, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Sentinel-2B': [0.15, 0.09, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
time_init = {
    'Sentinel-2A': datetime.datetime(2015, 6, 23, 10, 0),
    'Sentinel-2B': datetime.datetime(2017, 3, 7, 10, 0)
}
S2 = Sentinel2()


def calc_temporal_degradation(
        sensing_time: str,
        spacecraft: str
    ) -> Dict[str, float]:
    """
    Calculates the L1C Sentinel-2 uncertainty originating from the temporal
    degradation of the MSI instrument based on MERIS annual degradation rates.
    Original function taken from the Sentinel-2 Uncertainty Toolbox written
    by Gorrono et al. (2017), doi:10.3390/rs9020178

    :param sensing_time:
        sensing timestamp extracted from the scene metadata. The format
        shall be '%Y-%m-%d %H:%M:%S.%fZ' (e.g., 2019-10-22 10:38:05.579265)
    :param spacecraft:
        name of the spacecraft. Must be Sentinel-2A or Sentinel-2B.
    """
    # START or STOP time has no effect. We provide a degradation based on MERIS year rates
    time_start = datetime.datetime.strptime(sensing_time, '%Y-%m-%d %H:%M:%S.%f')
    res_dict = dict.fromkeys(S2.BAND_INDICES.keys())
    for spectral_band in res_dict.keys():
        res_dict[spectral_band] = (time_start - time_init[spacecraft]).days / 365.25 * \
           u_diff_temp_rate[spacecraft][S2.BAND_INDICES[spectral_band]]
    return res_dict


if __name__ == '__main__':
    
    sensing_time = '2019-10-22 10:38:05.579265'
    spacecraft = 'Sentinel-2A'
    u_temp_degradation = calc_temporal_degradation(
        sensing_time=sensing_time,
        spacecraft=spacecraft
    )
    