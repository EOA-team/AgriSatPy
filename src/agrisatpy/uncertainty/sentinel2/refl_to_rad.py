'''
Created on Sep 21, 2021

@author: graflu
'''

import numpy as np
import pandas as pd
from typing import List
from datetime import date
from datetime import datetime
from numbers import Number


def refl_to_rad(
        refl: Number,
        sensing_date: date,
        sza: float,
        esun: float
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
        sun zenith angle of the pixel for which to calculate the
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
    # 0.01672: eccentritcity of the ellipse of the Earth around the sun
    # 0.9856 = 360/365.256363: one full rotation in deg divided by the length of a solar year
    # 4: day of year when Earth reaches perihelion
    d = 1 - 0.01672 * np.cos(0.9856 * (doy-4))
    # conversion factor
    multiplier = esun * solar_angle_correction/(np.pi*d**2)
    # at-sensor radiance
    rad = refl * multiplier
    return rad


def spectra_to_radiance(
        spectra_df: pd.DataFrame,
        colnames_bands: List[str],
        colnames_irrad: List[str],
        colname_sza: str,
        colname_date: str
    ) -> pd.DataFrame:
    """
    Applies the reflectance to at-sensor-radiance conversion on a
    dataframe of pixel (top-of-atmosphere) spectra. In addition,
    the pixel sun zenith angle must be provided as a column
    named sza in the dataframe. Moreover, the solar irradiance in each
    spectral band must be given.

    :param spectra_df:
        dataframe with spectral bands, the pixel solar zenith angle
        and the solar irradiance per band. Each row denotes a pixel
        at a specific image acquisition date.
    :param colnames_bands:
        list of column names denoting the spectral bands. E.g.,
        ['b02_toa', 'b03_toa']
    :param colnames_irrad:
        list of column names denoting the solar irradiance per band
        and acquisition date. E.g,['solar_irradiance_b02', 'solar_irradiance_b03'].
        Should be in the same order as colnames_bands!
    :param colname_sza:
        name of the column with the solar zenith angle
    :param colname_date:
        name of the column with the (sensing) date
    """

    # band names for saving the radiance outputs to empty df columns
    bandnames_rad = [f'{x.replace("_toa", "")}_radiance' for x in colnames_bands]
    for bandname_rad in bandnames_rad:
        spectra_df[bandname_rad] = np.nan

    # convert date column to datetime date
    spectra_df[colname_date] = spectra_df[colname_date].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d').date()
    )

    # loop over spectral bands and calculate the radiance
    for idx, colname_band in enumerate(colnames_bands):
        spectra_df[bandnames_rad[idx]] = spectra_df.apply(
            lambda x,
            cb=colname_band,
            cd=colname_date,
            cs=colname_sza,
            ci=colnames_irrad,
            idx=idx: refl_to_rad(
                refl=x[cb],
                sensing_date=x[cd],
                sza=x[cs],
                esun=x[ci[idx]]
            ),
            axis=1
        )

    return spectra_df


if __name__ == '__main__':

    from pathlib import Path
    
    data_dir = Path('/mnt/ides/Lukas/04_Work/Uncertainty/Refl_to_Rad')
    
    toa_refl = pd.read_csv(data_dir.joinpath('toa_reflectance2019.csv'))
    sol_irrad = pd.read_csv(data_dir.joinpath('solar_irradiance2019.csv'))
    spectra_df = pd.merge(toa_refl, sol_irrad, on='product_uri')

    # define dataframe column names
    colnames_bands = [
        'b02_toa',
        'b03_toa',
        'b04_toa',
        'b05_toa',
        'b06_toa',
        'b07_toa',
        'b08_toa',
        'b8a_toa',
        'b11_toa',
        'b12_toa'
    ]
    colnames_irrad = [
        'solar_irradiance_b02',
        'solar_irradiance_b03',
        'solar_irradiance_b04',
        'solar_irradiance_b05',
        'solar_irradiance_b06',
        'solar_irradiance_b07',
        'solar_irradiance_b08',
        'solar_irradiance_b8a',
        'solar_irradiance_b11',
        'solar_irradiance_b12'
    ]
    colname_sza = 'sza'
    colname_date = 'date'

    res = spectra_to_radiance(
        spectra_df=spectra_df,
        colnames_bands=colnames_bands,
        colnames_irrad=colnames_irrad,
        colname_sza=colname_sza,
        colname_date=colname_date
    )

    res.to_csv(data_dir.joinpath('toa_radiance2019.csv'), index=False)
