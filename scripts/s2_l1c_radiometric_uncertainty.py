"""
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from agrisatpy.config import get_settings
from agrisatpy.config.sentinel2 import Sentinel2
from agrisatpy.uncertainty import calc_temporal_degradation, S2RutAlgo


S2 = Sentinel2()
S2_Bands = S2.BAND_INDICES
Settings = get_settings()
DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)

# coverage factor
k = 2

# CSV with TOA reflectances to analyze
fname_toa_refl = '/mnt/ides/Lukas/04_Work/Uncertainty/Radiometric_Uncertainty/toa_reflectance2019.csv'
toa_refl = pd.read_csv(fname_toa_refl)

# get temporal date range
min_date = toa_refl.date.min()
max_date = toa_refl.date.max()

# query metadb to get scene metadata required for uncertainty calculation
query = f"""
SELECT 
    scene_id,
    product_uri,
    spacecraft_name,
    sensing_time,
    solar_irradiance_b02,
    solar_irradiance_b03,
    solar_irradiance_b04,
    solar_irradiance_b05,
    solar_irradiance_b06,
    solar_irradiance_b07,
    solar_irradiance_b08,
    solar_irradiance_b8a,
    solar_irradiance_b11,
    solar_irradiance_b12,
    reflectance_conversion,
    alpha_b02,
    alpha_b03,
    alpha_b04,
    alpha_b05,
    alpha_b06,
    alpha_b07,
    alpha_b08,
    alpha_b8a,
    alpha_b11,
    alpha_b12,
    beta_b02,
    beta_b03,
    beta_b04,
    beta_b05,
    beta_b06,
    beta_b07,
    beta_b08,
    beta_b8a,
    beta_b11,
    beta_b12,
    physical_gain_b02,
    physical_gain_b03,
    physical_gain_b04,
    physical_gain_b05,
    physical_gain_b06,
    physical_gain_b07,
    physical_gain_b08,
    physical_gain_b8a,
    physical_gain_b11,
    physical_gain_b12
FROM
    sentinel2_raw_metadata
LEFT JOIN sentinel2_raw_metadata_ql
ON sentinel2_raw_metadata.datatakeidentifier = sentinel2_raw_metadata_ql.datatakeidentifier    
WHERE
    processing_level = 'Level-1C'
AND
    sensing_date
BETWEEN
    '{min_date}'
AND
    '{max_date}';
"""
meta_df = pd.read_sql(sql=query, con=engine)

# join dataframes
df = pd.merge(toa_refl, meta_df, on='scene_id')

# apply the uncertainty calculation
S2_Bands.pop('B01')
S2_Bands.pop('B09')
S2_Bands.pop('B10')
S2_Bands.pop('B11')

# loop over rows in dataframe
for _, record in df.iterrows():
    
    # calculate temporal degradation
    u_diff_temp = calc_temporal_degradation(
        sensing_time=str(record.sensing_time),
        spacecraft=record.spacecraft_name
    )

    # radiometric uncertainty of the spectral bands
    for band_item in list(S2_Bands.items()):
        
        band_name = band_item[0].lower()
        band_idx = band_item[1]

        s2rutalgo = S2RutAlgo(
            a=record[f'physical_gain_{band_name}'],
            e_sun=record[f'solar_irradiance_{band_name}'],
            u_sun=record.reflectance_conversion,
            tecta=record.sza,
            alpha=record[f'alpha_{band_name}'],
            beta=record[f'beta_{band_name}'],
            k=k,
            u_diff_temp=u_diff_temp[band_name.upper()]
        )
        record[f'{band_name}_unc'] = s2rutalgo.unc_calculation(
            band_data=record[f'{band_name}_toa'],
            band_id=band_idx,
            spacecraft=record.spacecraft_name
        )

    # save outputs to DB -> TODO
    print('a')
