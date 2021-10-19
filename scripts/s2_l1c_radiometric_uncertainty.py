# """
# """
#
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from agrisatpy.config import get_settings
from agrisatpy.config.sentinel2 import Sentinel2
from agrisatpy.uncertainty import calc_temporal_degradation, S2RutAlgo

# TODO: recalculate because B11 was forgotten!!
S2 = Sentinel2()
S2_Bands = S2.BAND_INDICES

# remove those spectral bands not required
S2_Bands.pop('B01')
S2_Bands.pop('B09')
S2_Bands.pop('B10')
#
# Settings = get_settings()
# DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
# engine = create_engine(DB_URL, echo=Settings.ECHO_DB)
#
# # coverage factor
# k = 2
#
# # CSV with TOA reflectances to analyze
# fname_toa_refl = '/mnt/ides/Lukas/04_Work/Uncertainty/Radiometric_Uncertainty/toa_reflectance2019.csv'
# toa_refl = pd.read_csv(fname_toa_refl)
#
# # get temporal date range
# min_date = toa_refl.date.min()
# max_date = toa_refl.date.max()
#
# # query metadb to get scene metadata required for uncertainty calculation
# query = f"""
# SELECT 
#     scene_id,
#     product_uri,
#     spacecraft_name,
#     sensing_time,
#     solar_irradiance_b02,
#     solar_irradiance_b03,
#     solar_irradiance_b04,
#     solar_irradiance_b05,
#     solar_irradiance_b06,
#     solar_irradiance_b07,
#     solar_irradiance_b08,
#     solar_irradiance_b8a,
#     solar_irradiance_b11,
#     solar_irradiance_b12,
#     reflectance_conversion,
#     alpha_b02,
#     alpha_b03,
#     alpha_b04,
#     alpha_b05,
#     alpha_b06,
#     alpha_b07,
#     alpha_b08,
#     alpha_b8a,
#     alpha_b11,
#     alpha_b12,
#     beta_b02,
#     beta_b03,
#     beta_b04,
#     beta_b05,
#     beta_b06,
#     beta_b07,
#     beta_b08,
#     beta_b8a,
#     beta_b11,
#     beta_b12,
#     physical_gain_b02,
#     physical_gain_b03,
#     physical_gain_b04,
#     physical_gain_b05,
#     physical_gain_b06,
#     physical_gain_b07,
#     physical_gain_b08,
#     physical_gain_b8a,
#     physical_gain_b11,
#     physical_gain_b12
# FROM
#     sentinel2_raw_metadata
# LEFT JOIN sentinel2_raw_metadata_ql
# ON sentinel2_raw_metadata.datatakeidentifier = sentinel2_raw_metadata_ql.datatakeidentifier    
# WHERE
#     processing_level = 'Level-1C'
# AND
#     sensing_date
# BETWEEN
#     '{min_date}'
# AND
#     '{max_date}';
# """
# meta_df = pd.read_sql(sql=query, con=engine)
#
# # join dataframes
# df = pd.merge(toa_refl, meta_df, on='scene_id')
#
# # apply the uncertainty calculation
# # loop over rows in dataframe
# for idx, record in df.iterrows():
#
#     # calculate temporal degradation
#     u_diff_temp = calc_temporal_degradation(
#         sensing_time=str(record.sensing_time),
#         spacecraft=record.spacecraft_name
#     )
#
#     # radiometric uncertainty of the spectral bands
#     for band_item in list(S2_Bands.items()):
#
#         band_name = band_item[0].lower()
#         band_idx = band_item[1]
#
#         s2rutalgo = S2RutAlgo(
#             a=record[f'physical_gain_{band_name}'],
#             e_sun=record[f'solar_irradiance_{band_name}'],
#             u_sun=record.reflectance_conversion,
#             tecta=record.sza,
#             alpha=record[f'alpha_{band_name}'],
#             beta=record[f'beta_{band_name}'],
#             k=k,
#             u_diff_temp=u_diff_temp[band_name.upper()]
#         )
#         df.loc[idx, f'{band_name}_unc'] = s2rutalgo.unc_calculation(
#             band_data=record[f'{band_name}_toa'],
#             band_id=band_idx,
#             spacecraft=record.spacecraft_name
#         )
#
# # save outputs to  print('a')
# df.to_csv('/mnt/ides/Lukas/04_Work/Uncertainty/Radiometric_Uncertainty/radiometric_uncertainty.csv')
# # TODO write to database
unc_cols = [f'{x.lower()}_unc' for x in list(S2_Bands.keys())]
#
from phenomen.pheno_db import update_pixel_observations
for _, record in df.iterrows():
    fid = record.fid,
    pixid = record.pixid,
    date = record.date
    value_dict = record[unc_cols].to_dict()
    update_pixel_observations(
        parcel_id=fid,
        pixel_id=pixid,
        date=date,
        value_dict=value_dict
    )


# # analyze data
df = pd.read_csv('/mnt/ides/Lukas/04_Work/Uncertainty/Radiometric_Uncertainty/radiometric_uncertainty.csv')
df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
min_date = df.date.min()
max_date = df.date.max()

# group data by date to get time series of uncertainty values
unc_cols.append('date')
grouped_unc = df[unc_cols].groupby(by='date').mean()

# plotting
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('bmh')

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111)

# get qualitative color map
cmap = ['rosybrown', 'indianred', 'chocolate', 'limegreen', 'darkgreen',
        'cyan', 'steelblue', 'indigo', 'slategrey', 'goldenrod']
num_colors = len(unc_cols)
for idx, unc_col in enumerate(unc_cols):
    if unc_col == 'date': continue
    ax.plot(
        grouped_unc.index,
        grouped_unc[unc_col]*0.1, # values are scaled by 10
        label=unc_col.split('_')[0].upper(),
        marker='x',
        color=cmap[idx]
    )

ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('Time', fontsize=18)
ax.set_ylabel('Expanded Radiometric Uncertainty [%]\n(k=2)', fontsize=18)
ax.set_ylim(1, 5.5)
ax.set_xlim(min_date, max_date)
xlims = ax.get_xlim()
ax.hlines(3, xlims[0], xlims[1], linestyle='dashed', label='Targeted Uncertainty', color='green')
ax.hlines(5, xlims[0], xlims[1], linestyle='dashed', label='Max Accepted Uncertainty', color='red')
plt.xticks(rotation=45)

ax.legend(bbox_to_anchor=(1.01, 1.), fontsize=14)
fig.savefig(
    '/mnt/ides/Lukas/04_Work/Uncertainty/Radiometric_Uncertainty/radiomtric_unc.png',
    bbox_inches='tight'
)
plt.show()
