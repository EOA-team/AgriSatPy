'''
Function that could be used in (parallelized) way for preprocessing Sentinel-2 data
as part of the ``AgriSatPy`` operational processing pipeline.
             
- resampling from 20 to 10m spatial resolution
- generation of a RGB preview image per scene
- merging of split scenes (due to data take issues)
- resampling of SCL data (L2A processing level, only)
- generation of metadata file containing file-paths to the processed data

Relies on metadata from AgriSatPy's metadata DB. Currently metadata CSV format is also
supported but this functionality will most likely be removed in the near future.

Usage example (assuming that the database has entries for the selected tile and time period):

```{python}

from agrisatpy.operational.resampling.sentinel2 import exec_S2_pipeline



```
'''

import os
import time
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from joblib import Parallel, delayed
from sqlalchemy import create_engine
from sqlalchemy import and_
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker
from typing import Optional
from typing import Tuple
from typing import Dict
from datetime import datetime

from agrisatpy.spatial_resampling.sentinel2 import resample_and_stack_s2
from agrisatpy.spatial_resampling.sentinel2 import merge_split_scenes
from agrisatpy.operational.utils import identify_split_scenes
from agrisatpy.utils import reconstruct_path
from agrisatpy.config import get_settings
from agrisatpy.metadata.sentinel2.database import S2_Raw_Metadata
from agrisatpy.utils.constants import ProcessingLevels
from agrisatpy.utils.constants.sentinel2 import ProcessingLevelsDB


Settings = get_settings()
logger = Settings.logger


# TODO: move to metadata
def _get_db_url() -> str:
    """
    returns the database connection string
    """

    return f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'

# TODO: move to metadata
def connect_db():
    """
    connects to AgriSatPy's metadata base and returns a session
    object
    """
    DB_URL = _get_db_url()
    engine = create_engine(DB_URL, echo=Settings.ECHO_DB)
    return sessionmaker(bind=engine)()


def do_parallel(
        in_df: pd.DataFrame,
        loopcounter: int, 
        out_dir: Path, 
        **kwargs
    ) -> Dict[str, Path]:
    """
    Wrapper function for (potential) parallel execution of S2 resampling & stacking, 
    and SCL resampling (L2A, only).

    Returns a dict containing the file-paths to the generated datasets.

    :param in_df:
        dataframe containing metadata of S2 scenes (must follow AgripySat convention)
    :param loopcounter:
        Index to get actual S2 scene in loop.
    :param out_dir:
        path where the bandstacks (and the SCL files) get written to.
    :param **kwargs:
        Optional key-word arguments to pass to ``resample_and_stack_s2``
    :return innerdict:
        dictionary with file-paths to generated datasets returned from
        ``resample_and_stack_s2``
    """

    innerdict = {}

    try:
        # reconstruct storage location based on DB records
        in_dir = reconstruct_path(record=in_df.iloc[loopcounter])
        innerdict = resample_and_stack_s2(
            in_dir=in_dir, 
            out_dir=out_dir,
            **kwargs
        )

    except Exception as e:
        logger.error(e)
        return innerdict

    # write to successful_scenes.txt to indicate that the processing of
    # the scene was accomplished without any errors
    scenes_log_file = out_dir.joinpath('log').joinpath(Settings.PROCESSING_CHECK_FILE_NO_BF)
    creation_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    with open(scenes_log_file, 'a+') as src:
        line = ''
        for key in innerdict.keys():
            line += str(innerdict[key]) + ','
        line += creation_time
        src.write(line + '\n')

    return innerdict


def exec_pipeline(
        target_s2_archive: Path,
        date_start: date,
        date_end: date,
        tile: str,
        processing_level: Optional[ProcessingLevels] = ProcessingLevels.L2A,
        n_threads: Optional[int] = 1,
        use_database: Optional[bool] = True,
        metadata_csv: Optional[Path] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execution of the Sentinel-2 pre-processing pipeline, including:
    
    -> resampling from 20 to 10m spatial resolution
    -> generation of a RGB preview image per scene
    -> merging of split scenes (due to data take issues)
    -> resampling of SCL data (L2A processing level, only)

    IMPORTANT: The function works on single tiles and L1C or L2A processing level.

    HINT: The main part of the pipeline except the blackfill merging can be executed
    using multiple threads if ``n_threads`` is set to a value larger 1. This feature
    is experimental.

    :param target_s2_archive:
        target archive where the processed data should be stored. The
        root of the archive where the L2A/ L1C sub-folders are located is required.
        IMPORTANT: The archive structure must follow the AgripySat conventions. It is
        recommended to use the provided archive creation functions!
    :param date_start:
        start date of the period to process. NOTE: The temporal
        selection is bound by the available (i.e., downloaded) Sentinel-2 data!
    :param date_end:
        end_date of the period to process. NOTE: The temporal
        selection is bound by the available (i.e., downloaded) Sentinel-2 data!
    :param tile:
        Sentinel-2 tile to process (e.g. 'T32TLT')
    :param processing_level:
        Sentinel-2 processing level (either L1C or L2A, L2A by default).
    :param n_threads:
        number of threads to use for execution of the preprocessing depending on
        your computer hardware. Per default set to 1 (no parallelization).
    :param use_database:
        if True (default and strongly recommended) uses AgriSatPy's metadata base
        to find datasets. If False, the filepath to the metadata_csv must be provided.
    :param metadata_csv:
        filepath to the metadata csv. Must be provided if ``use_database == False``
    :param kwargs:
        kwargs to pass to resample_and_stack_S2 and scl_10m_resampling (L2A, only)
    :return:
        tuple containing metadata of the processed datasets [0] and eventually
        failed datasets in the second tuple item
    """

    # make sub-directory for logging successfully processed scenes in out_dir
    log_dir = target_s2_archive.joinpath('log')
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # check the metadata from the database (recommended)
    if use_database:

        # translate processing level
        processing_level_db = ProcessingLevelsDB[processing_level.value]

        # TODO: move this function to metadata sub-package
        session = connect_db()
        query_statement = session.query(
                S2_Raw_Metadata.product_uri,
                S2_Raw_Metadata.scene_id,
                S2_Raw_Metadata.storage_share,
                S2_Raw_Metadata.storage_device_ip_alias,
                S2_Raw_Metadata.storage_device_ip,
                S2_Raw_Metadata.sensing_date
            ).filter(
                S2_Raw_Metadata.tile_id == tile
            ).filter(
                and_(
                    S2_Raw_Metadata.sensing_date <= date_end,
                    S2_Raw_Metadata.sensing_date >= date_start
                )
            ).filter(
                S2_Raw_Metadata.processing_level == processing_level_db
            ).order_by(
                S2_Raw_Metadata.sensing_date.desc()
            ).statement
        metadata = pd.read_sql(
            query_statement,
            session.bind
        )
    # or use the csv (not recommended but possible)
    else:
        try:
            raw_metadata = pd.read_csv(metadata_csv)
        except Exception as e:
            raise Exception from e

        raw_metadata.columns = raw_metadata.columns.str.lower()
        metadata_filtered = raw_metadata[
            (raw_metadata.tile_id == tile) & (raw_metadata.processing_level == processing_level)
        ]
        metadata = metadata_filtered[
            pd.to_datetime(raw_metadata.sensing_date).between(
                np.datetime64(date_start), np.datetime64(date_end), inclusive=True
            )
        ]

    # get "duplicates", i.e, scenes that have the same sensing date because of datastrip
    # beginning/end issue
    num_scenes = metadata.shape[0]
    meta_blackfill = identify_split_scenes(metadata_df=metadata)

    # exclude these duplicated scenes from the main (parallelized) workflow!
    metadata = metadata[~metadata.product_uri.isin(meta_blackfill['product_uri'])]
    if meta_blackfill.empty:
        logger.info(
            f'Found {num_scenes} scenes out of which 0 must be merged'
        )
    else:
        logger.info(
            f'Found {num_scenes} scenes out of which {meta_blackfill.shape[0]} must be merged'
        )

    t = time.time()
    result = Parallel(n_jobs = n_threads)(
        delayed(do_parallel)(
            in_df=metadata, 
            loopcounter=idx, 
            out_dir=target_s2_archive,
            **kwargs
            ) for idx in range(metadata.shape[0]))
    logger.info(f'Execution time: {time.time()-t} seconds.')

    # concatenate the metadata of the stacked image files into a pandas dataframe
    bandstack_meta = pd.DataFrame(result)

    # merge black-fill scenes (data take issue) if any
    if not meta_blackfill.empty:
        logger.info('Starting merging of blackfill scenes')

        # after regular scene processsing, process the blackfill scenes single-threaded
        for date in meta_blackfill.sensing_date.unique():
            scenes = meta_blackfill[meta_blackfill.sensing_date == date]
            product_id_1 = scenes.product_uri.iloc[0]
            product_id_2 = scenes.product_uri.iloc[1]
            # reconstruct storage location
            raw_data_archive = reconstruct_path(record=scenes.iloc[0]).parent
            scene_1 = raw_data_archive.joinpath(product_id_1)
            scene_2 = raw_data_archive.joinpath(product_id_2)
            logger.info(
                f'Working on {scene_1} and {scene_2}'
            )

            # the usual naming problem with the .SAFE directories
            if not scene_1.exists():
                # product_id_1 = Path(str(product_id_1.replace('.SAFE', '')))
                scene_1 = Path(str(scene_1).replace('.SAFE',''))
            if not scene_2.exists():
                # product_id_2 = Path(str(product_id_2.replace('.SAFE', '')))
                scene_2 = Path(str(scene_2).replace('.SAFE',''))

            try:
                res = merge_split_scenes(
                    scene_1=scene_1,
                    scene_2=scene_2,
                    out_dir=target_s2_archive, 
                    **kwargs
                )
                res.update(
                    {'scene_was_merged': True,
                     'spatial_resolution': kwargs.get('target_resolution', 10)
                     }
                )
                # check if bandstack_meta is already populated, otherwise create it
                if bandstack_meta.empty:
                    bandstack_meta = pd.DataFrame.from_records([res])
                else:
                    bandstack_meta = bandstack_meta.append(res, ignore_index=True)

                # write to log file
                scenes_log_file = target_s2_archive.joinpath('log').joinpath(
                    Settings.PROCESSING_CHECK_FILE_BF
                )
                creation_time = datetime.now().strftime('%Y%m%d-%H%M%S')
                with open(scenes_log_file, 'a+') as src:
                    line = f'{str(scene_1)}, {str(scene_2)}, {creation_time}'
                    src.write(line + '\n')

            except Exception as e:
                logger.error(f'Failed to merge {scene_1} and {scene_2}: {e}')

        # also the storage location shall be inserted into the database later
        # bandstack_meta['storage_share'] = target_s2_archive

        logger.info('Finished merging of blackfill scenes')

    # check for any empty bandstack paths (indicates that something went wrong
    # during the processing
    emtpy_bandstacks = bandstack_meta.loc[
        (bandstack_meta.bandstack == None) | (bandstack_meta.bandstack == '')
    ]

    # in this remove the errored records from bandstack_meta and reported the
    # failed records
    if not emtpy_bandstacks.empty:
        logger.error(f'Resampling failed for {emtpy_bandstacks.shape[0]} datasets')
        errored_records = pd.merge(
            bandstack_meta, emtpy_bandstacks, how='inner', on=['scene_id']
        )['scene_id']
        bandstack_meta = bandstack_meta[~bandstack_meta['scene_id'].isin(errored_records)]

    # TODO: automatically insert records into DB if selected by the user

    # save metadata of all stacked files and return
    return bandstack_meta, emtpy_bandstacks
