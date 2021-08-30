'''
Created on Aug 30, 2021

@author: Lukas Graf (D-USYS; ETHZ)
'''

import pandas as pd
from sqlalchemy import create_engine
from agrisatpy.config import get_settings


Settings = get_settings()
engine = create_engine(Settings.DB_URL, echo=Settings.ECHO_DB)


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
    pass
    # convert to geodataframe
    
    # convert dataframe column names to lower case
    
    # connect to database and insert the data