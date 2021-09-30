'''
Created on Jul 9, 2021

@author:     Lukas Graf, Gregor Perich (D-USYS, ETHZ)

@purpose:    function that could be used in parallelized way for preprocessing Sentinel-2 data
             -> resampling from 20 to 10m spatial resolution
             -> generation of a RGB preview image per scene
             -> merging of split scenes (due to data take issues)
             -> resampling of SCL data (L2A processing level, only)
             -> generation of metadata file containing file-paths to the processed data
'''

import os
import time
import pandas as pd
import numpy as np
import shutil
from datetime import date
from pathlib import Path
from joblib import Parallel, delayed
from sqlalchemy import create_engine
from sqlalchemy import and_
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker
from rasterio.enums import Resampling
from typing import Optional

from agrisatpy.spatial_resampling import resample_and_stack_S2
from agrisatpy.spatial_resampling import scl_10m_resampling
from agrisatpy.spatial_resampling import identify_split_scenes
from agrisatpy.spatial_resampling import merge_split_scenes
from agrisatpy.utils import reconstruct_path
from agrisatpy.config import get_settings
from agrisatpy.metadata.sentinel2.database import S2_Raw_Metadata

Settings = get_settings()
DB_URL = f'postgresql://{Settings.DB_USER}:{Settings.DB_PW}@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}'
engine = create_engine(DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()
logger = Settings.logger


class ArchiveNotFoundError(Exception):
    pass

class MetadataNotFoundError(Exception):
    pass


def do_parallel(in_df: pd.DataFrame,
                loopcounter: int, 
                out_dir: Path, 
                **kwargs
                ) -> dict:
    """
    Wrapper function for (potential) parallel execution of S2 resampling & stacking, 
    SCL resampling and optionally per-polygon-ID pixel value extraction (per S2 scene)
    Has to be looped over a metadata folder.

    Returns a dict containing the metadata for each resampled & stacked scene.

    :param in_df:
        dataframe containing metadata of S2 scenes (must follow AgripySat convention)
    :param loopcounter:
        Index to get actual S2 scene in loop.
    :param out_dir:
        path where the bandstacks (and the SCL files) get written to.
    :param **kwargs:
        is_L2A = kwargs.get('is_L2A', bool)
        in_file_polys = kwargs.get("in_file_polys", str)
        out_dir_csv = kwargs.get("out_dir_csv", str)
        buffer = kwargs.get("buffer", float)
        id_column = kwargs.get("id_column", str).
    :return innerdict:
        metadata dict of the processed dataset
    """
    
    # define (to catch win10 related errors)
    path_bandstack = '' 
    innerdict = {}

    try:

        # reconstruct storage location based on DB records
        in_dir = reconstruct_path(record=in_df.iloc[loopcounter])
    
        path_bandstack = resample_and_stack_S2(
            in_dir=in_dir, 
            out_dir=out_dir,
            **kwargs
        )
        
        # continue if the bandstack was only blackfilled
        if path_bandstack == '':
            return {}
        
    except Exception as e:
        logger.error(e)
        
    try:
        # resample each SCL file (L2A processing level only)
        is_L2A = kwargs.get('is_L2A', True)
        path_sclfile = ''
        if is_L2A:
            # remove the 'is_L2A' kwarg since it is not understood
            # by the scl_10m_resampling function
            kwargs.pop('is_L2A')
            path_sclfile = scl_10m_resampling(
                in_dir=in_dir, 
                out_dir=out_dir,
                **kwargs
            )

        # resampling method
        if kwargs.get('pixel_division'):
            resampling_method = 'pixel division'
        else:
            interpol_method = kwargs.get('interpolation', -999)
            if  interpol_method > 0:
                resampling_method = Resampling(interpol_method).name
            else:
                resampling_method = 'cubic' # default

        # write to metadata dictionary
        innerdict = {
            "bandstack": os.path.basename(path_bandstack), 
            "scl": os.path.join(
                Settings.SUBDIR_SCL_FILES,
                os.path.basename(path_sclfile)
            ),
            "preview": os.path.join(
                Settings.SUBDIR_RGB_PREVIEWS,
                os.path.splitext(
                    os.path.basename(path_bandstack))[0] + '.png'
            ),
            'product_uri': in_df.product_uri.iloc[loopcounter],
            'scene_id': in_df.scene_id.iloc[loopcounter],
            'spatial_resolution': 10.,
            'resampling_method': resampling_method,
            'scene_was_merged': False
        }

    except Exception as e:
        logger.error(e)

    # write to successful_scenes.txt to indicate that the processing of
    # the scene was accomplished without any errors
    scenes_log_file = out_dir.joinpath('log').joinpath(Settings.PROCESSING_CHECK_FILE_NO_BF)
    with open(scenes_log_file, 'a+') as src:
        line = f"{in_dir.joinpath(innerdict['product_uri'])}, {innerdict['bandstack']}, {innerdict['scl']}, {innerdict['preview']}"
        src.write(line + os.linesep)

    return innerdict


def exec_parallel(target_s2_archive: Path,
                  date_start: date,
                  date_end: date,
                  n_threads: int,
                  tile: str,
                  use_database: Optional[bool]=True,
                  **kwargs
                  ) -> pd.DataFrame:
    """
    parallel execution of the Sentinel-2 pre-processing pipeline, including:
    
    -> resampling from 20 to 10m spatial resolution
    -> generation of a RGB preview image per scene
    -> merging of split scenes (due to data take issues)*
    -> resampling of SCL data (L2A processing level, only)

    Merging is not done in the main parallelized function (see above) but executed
    afterwards (currently single threaded).

    :param target_s2_archive:
        target archive where the preprocessed data should be stored. The
        root of the archive where the L2A/ L1C sub-folders are located is required.
        IMPORTANT: The archive structure must follow the AgripySat conventions. It is
        recommended to use the provided archive creation functions!
    :param date_start:
        start date of the period to process. NOTE: The temporal
        selection is bound by the available (i.e., downloaded) Sentinel-2 data!
    :param date_end:
        end_date of the period to process. NOTE: The temporal
        selection is bound by the available (i.e., downloaded) Sentinel-2 data!
    :param n_threads:
        number of threads to use for execution of the preprocessing depending on
        your computer hardware.
    :param kwargs:
        kwargs to pass to resample_and_stack_S2 and scl_10m_resampling (L2A, only)
    :return bandstack_meta:
        metadata of the processed datasets
    """

    # check processing level; default is L2A
    is_l2a = kwargs.get('is_L2A', True)
    if is_l2a:
        processing_level = 'Level-2A'
    else:
        processing_level = 'Level-1C'

    # make subdirectory for logging successfully processed scenes in out_dir
    log_dir = target_s2_archive.joinpath('log')
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # check the metadata from the database
    if use_database:
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
                S2_Raw_Metadata.processing_level == processing_level
            ).order_by(
                S2_Raw_Metadata.sensing_date.desc()
            ).statement
        metadata = pd.read_sql(
            query_statement,
            session.bind
        )
    # or use the csv
    else:
        raw_data_archive = Path(kwargs.get('raw_data_archive'))
        raw_metadata = pd.read_csv(raw_data_archive)
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
    logger.info(f'Time difference of {time.time()-t} seconds.')

    # concatenate the metadata of the stacked image files into a pandas dataframe
    bandstack_meta = pd.DataFrame(result)

    # merge blackfill scenes (data take issue) if any
    # TODO: this section is for sure buggy and there is a problem in the datamodel...
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
                f'Starting merging {scene_1} and {scene_2}'
            )
            # the usual naming problem witht the .SAFE directories
            if not scene_1.exists():
                product_id_1 = Path(str(product_id_1.replace('.SAFE', '')))
            if not scene_2.exists():
                product_id_2 = Path(str(product_id_2.replace('.SAFE', '')))
            try:
                res = merge_split_scenes(
                    scene_1=scene_1,
                    scene_2=scene_2,
                    out_dir=target_s2_archive, 
                    **kwargs
                )
                res.update(
                    {'scene_was_merged': True,
                     'spatial_resolution': 10.,
                     'resampling_method': bandstack_meta.resampling_method.iloc[0],
                     'product_uri': product_id_1,  # take the first scene
                     'scene_id': scenes.scene_id.iloc[0]
                     }
                )
                bandstack_meta = bandstack_meta.append(res, ignore_index=True)

                scenes_log_file = target_s2_archive.joinpath('log').joinpath(
                    Settings.PROCESSING_CHECK_FILE_BF
                )
                with open(scenes_log_file, 'a+') as src:
                    line = f"{scene_1}, {scene_2}"
                    src.write(line + os.linesep)
            except Exception as e:
                logger.error(f'Failed to merge {scene_1} and {scene_2}: {e}')

        # also the storage location shall be inserted into the database later
        bandstack_meta['storage_location'] = target_s2_archive

        # move to target archive
        try:
            shutil.move(
                res['bandstack'],
                os.path.join(
                    target_s2_archive,
                    os.path.basename(
                        res['bandstack']
                    )
                )
            )
        except Exception as e:
            logger.error(f'Could not move {res["bandstack"]}: {e}')

        try:
            shutil.move(
                res['scl'],
                os.path.join(
                    os.path.join(
                        target_s2_archive,
                        Settings.SUBDIR_SCL_FILES
                    ),
                    os.path.basename(
                        res['scl']
                    )
                )
            )
        except Exception as e:
            logger.error(f'Could not move {res["scl"]}: {e}')

        try:
            shutil.move(
                res['preview'],
                os.path.join(
                    os.path.join(
                        target_s2_archive,
                        Settings.SUBDIR_RGB_PREVIEWS
                    ),
                    os.path.basename(
                        res['preview']
                    )
                )
            )
        except Exception as e:
            logger.error(f'Could not move {res["preview"]}: {e}')
    
        # remove working directory in the end
        try:
            shutil.rmtree(
                os.path.join(
                    target_s2_archive,
                    'temp_blackfill'
                )
            )
        except Exception as e:
            logger.error(f'Could not delete temp_blackfill: {e}')

        logger.info('Finished merging of blackfill scenes')
    
    # write metadata of all stacked files to CSV
    return bandstack_meta
