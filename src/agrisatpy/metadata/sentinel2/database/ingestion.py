'''
Created on Aug 30, 2021

@author: Lukas Graf (D-USYS; ETHZ)
'''

import pandas as pd
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agrisatpy.metadata.sentinel2.database.db_model import S2_Raw_Metadata
from agrisatpy.metadata.sentinel2.database.db_model import S2_Processed_Metadata
from agrisatpy.config import get_settings


Settings = get_settings()

DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()


def meta_df_to_database(meta_df: pd.DataFrame,
                        raw_metadata: Optional[bool]=True
                        ) -> None:
    """
    Once the metadata from one or more scenes have been extracted
    the data can be ingested into the metadata base (strongly
    recommended).

    This function takes a metadata frame extracted from "raw" or
    "processed" (i.e., after spatial resampling, band stacking and merging)
    Sentinel-2 data and inserts the data via pandas intrinsic
    sql-methods into the database.

    :param meta_df:
        data frame with metadata of one or more scenes to insert
    """

    meta_df.columns = meta_df.columns.str.lower()
    for _, record in meta_df.iterrows():
        metadata = record.to_dict()
        if raw_metadata:
            session.add(S2_Raw_Metadata(**metadata))
        else:
            session.add(S2_Processed_Metadata(**metadata))
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
