'''
Created on Sep 21, 2021

@author: graflu
'''

import numpy as np
from datetime import date
from numbers import Number


def refl_to_rad(
        self,
        refl: Number,
        sensing_date: date,
        sza: float,
        esun: float,
        metadata: dict,
    ) -> float:
    """
    Converts top of atmosphere reflectance to at-sensor radiance
    using solar parameters (irradiance plus path). Works band-wise,
    i.e., this function must be applied to each spectral band,
    separately.

    :param refl:
        top-of-atmopshere reflectance in a selected spectral band
        of a single pixel
    :param sensing_date:
        sensing date of the reflectance value
    :param sza:
        sun zenith angle of the pixel for which to caluclate the
        radiance
    :param esun:
        solar irradiance in the selected spectral band
    :param bandname:
        name of the selected spectral band in order to get the correct
        solar irradiance from the scene metadata
    :param metadata:
        metadata values extracted from the AgriSatPy metadata base for
        the selected scene
    :return rad:
        at-sensor-radiance (W m-2 sr-1 um-1)
    """
    # solar exoatmospheric spectral irradiance
    # esun = metadata[f'solar_irradiance_{bandname}']
    solar_angle_correction = np.cos(np.radians(sza))
    # Earth-Sun distance (from day of year)
    doy = sensing_date.timetuple().tm_yday
    # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
    d = 1 - 0.01672 * np.cos(0.9856 * (doy-4))
    # conversion factor
    multiplier = esun * solar_angle_correction/(np.pi*d**2)
    # at-sensor radiance
    rad = refl * multiplier
    return rad
