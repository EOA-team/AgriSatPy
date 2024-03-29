'''
Functions to insert Sentinel-2 specific metadata into the metadata DB
'''

import pandas as pd

from typing import Optional
from typing import List
from sqlalchemy import create_engine
from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker

from agrisatpy.metadata.database.db_model import S2_Raw_Metadata
from agrisatpy.metadata.database.db_model import S2_Processed_Metadata
from agrisatpy.config import get_settings


Settings = get_settings()
logger = Settings.logger

DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()


def meta_df_to_database(
        meta_df: pd.DataFrame,
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
    :param raw_metadata:
        If set to False, assumes the metadata is about processed
        products
    """

    meta_df.columns = meta_df.columns.str.lower()
    for _, record in meta_df.iterrows():
        metadata = record.to_dict()
        try:
            if raw_metadata:
                session.add(S2_Raw_Metadata(**metadata))
            else:
                session.add(S2_Processed_Metadata(**metadata))
            session.flush()
        except Exception as e:
            logger.error(f'Database INSERT failed: {e}')
            session.rollback()
    session.commit()


def metadata_dict_to_database(
        metadata: dict
    ) -> None:
    """
    Inserts extracted metadata into the meta database

    :param metadata:
        dictionary with the extracted metadata
    """

    # convert keys to lower case
    metadata =  {k.lower(): v for k, v in metadata.items()}
    try:
        session.add(S2_Raw_Metadata(**metadata))
        session.flush()
    except Exception as e:
        logger.error(f'Database INSERT failed: {e}')
        session.rollback()
    session.commit()


def update_raw_metadata(
        meta_df: pd.DataFrame,
        columns_to_update: List[str]
    ) -> None:
    """
    Function to update one or more atomic columns
    in the metadata base. The table primary keys 'scene_id'
    and 'product_uri' must be given in the passed dataframe.

    :param meta_df:
        dataframe with metadata entries to update. Must
        contain the two primary key columns 'scene_id' and
        'product_uri'
    :param columns_to_update:
        List of columns to update. These must be necessarily
        atomic attributes of the raw_metadata table.
    """

    meta_df.columns = meta_df.columns.str.lower()

    try:
        for _, record in meta_df.iterrows():
            # save values to update in dict
            value_dict = record[columns_to_update].to_dict()
            for key, val in value_dict.items():
                meta_db_rec = session.query(S2_Raw_Metadata) \
                    .filter(
                        and_(
                            S2_Raw_Metadata.scene_id == record.scene_id,
                            S2_Raw_Metadata.product_uri == record.product_uri
                        )
                    ).first()
                meta_db_rec.__getattribute__(key)
                setattr(meta_db_rec, key, val)
                session.flush()
            session.commit()
    except Exception as e:
        logger.error(f'Database UPDATE failed: {e}')
        session.rollback()
