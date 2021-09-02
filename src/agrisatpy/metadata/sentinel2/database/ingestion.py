'''
Created on Aug 30, 2021

@author: Lukas Graf (D-USYS; ETHZ)
'''

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agrisatpy.metadata.sentinel2.database.db_model import S2_Raw_Metadata
from agrisatpy.config import get_settings


Settings = get_settings()
engine = create_engine(Settings.DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()


def meta_df_to_database(meta_df: pd.DataFrame,
                        ) -> None:
    """
    Once the metadata from one or more scenes have been extracted
    the data can be ingested into the metadata base (strongly
    recommended).
    This function takes a metadata frame extracted from "raw"
    Sentinel-2 data and inserts the data via pandas intrinsic
    sql-methods into the database.

    :param meta_df:
        data frame with metadata of one or more scenes to insert
    """

    meta_df.columns = meta_df.columns.str.lower()
    for _, record in meta_df.iterrows():
        metadata = record.to_dict()
        session.add(S2_Raw_Metadata(**metadata))
        session.flush()
    session.commit()


def metadata_dict_to_database(metadata: dict
                              ) -> None:
    """
    Inserts extracted metadata into the meta database

    :param metadata:
        dictionary with the extracted metadata
    """

    # convert keys to lower case
    metadata =  {k.lower(): v for k, v in metadata.items()}
    session.add(S2_Raw_Metadata(**metadata))
    session.commit()
    


if __name__ == '__main__':

    from agrisatpy.metadata.sentinel2 import loop_s2_archive
    from pathlib import Path

    sat_dir = Path('/home/graflu/public/Evaluation/Projects/KP0022_DeepField/Sentinel-2/S2_L2A_data/CH/2019')
    
    metadata = loop_s2_archive(in_dir=sat_dir)
    metadata['storage_device_ip'] = '//hest.nas.ethz.ch/green_groups_kp_public'
    metadata['storage_share'] = metadata['storage_share'].apply(lambda x: x.replace('/home/graflu/public/',''))
    meta_df_to_database(meta_df=metadata)
    