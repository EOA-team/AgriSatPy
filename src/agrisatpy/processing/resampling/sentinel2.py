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
from agrisatpy import processing

Settings = get_settings()
engine = create_engine(Settings.DB_URL, echo=Settings.ECHO_DB)
session = sessionmaker(bind=engine)()



class ArchiveNotFoundError(Exception):
    pass

class MetadataNotFoundError(Exception):
    pass


<<<<<<< HEAD
def do_parallel(in_df: pd.DataFrame,
                loopcounter: int, 
                out_dir: Path, 
                **kwargs
                ) -> dict:
=======
def do_parallel(
        in_df: pd.DataFrame,
        loopcounter: int, 
        out_dir: str, 
        **kwargs
        ) -> dict:
>>>>>>> master
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
<<<<<<< HEAD
    :return innerdict:
        metadata dict of the processed dataset
    """
    try:

        # reconstruct storage location based on DB records
        in_dir = reconstruct_path(record=in_df.iloc[loopcounter])
    
        path_bandstack = resample_and_stack_S2(
            in_dir=in_dir, 
            out_dir=out_dir,
            **kwargs
        )
=======
    """
    try:

        # handle data obtained from Mundi
        is_mundi = kwargs.get('is_mundi', False)
        if not is_mundi:
            in_dir = in_df["PRODUCT_URI"].iloc[loopcounter]
        else:
            in_dir = in_df["PRODUCT_URI"].iloc[loopcounter].replace('.SAFE', '')
    
        path_bandstack = resample_and_stack_S2(in_dir=in_dir, 
                                               out_dir=out_dir,
                                               **kwargs)
>>>>>>> master
        # continue if the bandstack was blackfilled, only
        if path_bandstack == '':
            return {}

        # resample each SCL file (L2A processing level only)
        is_L2A = kwargs.get('is_L2A', True)
        path_sclfile = ''
        if is_L2A:
<<<<<<< HEAD
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
=======
            # remove the 'is_L2A' kwarg if present since it is not understood
            # by the scl_10m_resampling function
            if 'is_L2A' in kwargs.keys():
                kwargs.pop('is_L2A')
            path_sclfile = scl_10m_resampling(in_dir=in_dir, 
                                              out_dir=out_dir,
                                              **kwargs)

        # write to metadata dictionary
        innerdict = {
            "SENSING_DATE": in_df["SENSING_DATE"].iloc[loopcounter].date(), 
            "TILE": in_df["TILE"].iloc[loopcounter], 
            "FPATH_BANDSTACK": os.path.basename(path_bandstack), 
            "FPATH_SCL": os.path.join(
                Settings.SUBDIR_SCL_FILES,
                os.path.basename(path_sclfile)
            ),
            "FPATH_RGB_PREVIEW": os.path.join(
>>>>>>> master
                Settings.SUBDIR_RGB_PREVIEWS,
                os.path.splitext(
                    os.path.basename(path_bandstack))[0] + '.png'
            ),
<<<<<<< HEAD
            'product_uri': in_df.product_uri.iloc[loopcounter],
            'scene_id': in_df.scene_id.iloc[loopcounter],
            'spatial_resolution': 10.,
            'resampling_method': resampling_method,
            'scene_was_merged': False
        }


=======
            'PRODUCT_URI': in_df.PRODUCT_URI.iloc[loopcounter],
            'SCENE_ID': in_df.SCENE_ID.iloc[loopcounter]
        }

>>>>>>> master
    except Exception as e:
        logger.error(e)
        return {}

    return innerdict


<<<<<<< HEAD
def exec_parallel(target_s2_archive: Path,
                  date_start: date,
                  date_end: date,
                  n_threads: int,
                  tile: str,
                  use_database: Optional[bool]=True,
                  **kwargs
                  ) -> pd.DataFrame:
=======
def exec_parallel(raw_data_archive: str,
                  target_s2_archive: str,
                  date_start: str,
                  date_end: str,
                  n_threads: int,
                  tile: str,
                  **kwargs
                  ) -> None:
>>>>>>> master
    """
    parallel execution of the Sentinel-2 pre-processing pipeline, including:
    
    -> resampling from 20 to 10m spatial resolution
    -> generation of a RGB preview image per scene
    -> merging of split scenes (due to data take issues)*
    -> resampling of SCL data (L2A processing level, only)

    Merging is not done in the main parallelized function (see above) but executed
    afterwards (currently single threaded).

<<<<<<< HEAD
=======
    :param raw_data_archiv:
        directory containing collection of downloaded Sentinel-2 scenes in L1C or
        L2A processing level in 'raw' format as obtained from ESA/Copernicus or other
        data providers.
        IMPORTANT: This function will only work if the raw_data_archiv already contains
        a f'metadata_{year}.csv' file obtained by running loop_s2_archive from the
        metadata package of AgriSatPy.
>>>>>>> master
    :param target_s2_archive:
        target archive where the preprocessed data should be stored. The
        root of the archive where the L2A/ L1C sub-folders are located is required.
        IMPORTANT: The archive structure must follow the AgripySat conventions. It is
        recommended to use the provided archive creation functions!
    :param date_start:
<<<<<<< HEAD
        start date of the period to process. NOTE: The temporal
        selection is bound by the available (i.e., downloaded) Sentinel-2 data!
    :param date_end:
        end_date of the period to process. NOTE: The temporal
=======
        start date of the period to process (fmt: YYYY-MM-DD). NOTE: The temporal
        selection is bound by the available (i.e., downloaded) Sentinel-2 data!
    :param date_end:
        end_date of the period to process (fmt: YYYY-MM-DD). NOTE: The temporal
>>>>>>> master
        selection is bound by the available (i.e., downloaded) Sentinel-2 data!
    :param n_threads:
        number of threads to use for execution of the preprocessing depending on
        your computer hardware.
<<<<<<< HEAD
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

    # check the metadata from the database
    if use_database:
        metadata = pd.read_sql(
            session.query(S2_Raw_Metadata).filter(
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
            ).statement,
            session.bind
        )
    # or use the csv
    else:
        raw_data_archive = Path(kwargs.get('raw_data_archive'))
        raw_metadata = pd.read_csv(raw_data_archive.joinpath('metadata.csv'))
        raw_metadata.columns = raw_metadata.columns.str.lower()
        metadata_filtered = raw_metadata[
            (raw_metadata.tile_id == tile) & (raw_metadata.processing_level == processing_level)
        ]
        metadata = metadata_filtered[
            pd.to_datetime(raw_metadata.sensing_date).between(
                np.datetime64(date_start), np.datetime64(date_end), inclusive=True
            )
        ]
=======
    :param is_mundi:
        Data obtained from Mundi has no .SAFE scene naming convention which requires
        a slightly different search logic. Default is False.
    :param kwargs:
        kwargs to pass to resample_and_stack_S2 and scl_10m_resampling (L2A, only)
    """
    # check the metadata in the raw data archive
    if not os.path.isdir(raw_data_archive):
        raise ArchiveNotFoundError(f'No such file or directory: {raw_data_archive}')

    # cd into raw_data_archive
    os.chdir(raw_data_archive)
    try:
        metadata_file = glob.glob(os.path.join(raw_data_archive, 'metadata*.csv'))[0]
    except Exception as e:
        raise MetadataNotFoundError(
            f'Could not find metadata*.csv file in {raw_data_archive}: {e}'
    )

    metadata_full = pd.read_csv(metadata_file)

    # filter by S2 tile
    metadata = metadata_full[metadata_full.TILE == tile].copy()

    # filter metadata by provided date range and tile
    metadata.SENSING_DATE = pd.to_datetime(metadata.SENSING_DATE)
    date_start = datetime.strptime(date_start, '%Y-%m-%d')
    date_end = datetime.strptime(date_end, '%Y-%m-%d')
    metadata = metadata[pd.to_datetime(metadata.SENSING_DATE).between(date_start,
                                                                      date_end,
                                                                      inclusive=True)]
    metadata = metadata.sort_values(by='SENSING_DATE')
>>>>>>> master

    # get "duplicates", i.e, scenes that have the same sensing date because of datastrip
    # beginning/end issue
    meta_blackfill = identify_split_scenes(metadata_df=metadata)

    # exclude these duplicated scenes from the main (parallelized) workflow!
<<<<<<< HEAD
    metadata = metadata[~metadata.product_uri.isin(meta_blackfill['product_uri'])]
=======
    metadata = metadata[~metadata.PRODUCT_URI.isin(meta_blackfill["PRODUCT_URI"])]
>>>>>>> master
    if meta_blackfill.empty:
        logger.info(
            f'Found {metadata.shape[0]} scenes out of which 0 must be merged'
        )
    else:
        logger.info(
            f'Found {metadata.shape[0]} scenes out of which {meta_blackfill.shape[0]} must be merged'
        )

    t = time.time()
    result = Parallel(n_jobs = n_threads)(
        delayed(do_parallel)(
<<<<<<< HEAD
            in_df=metadata, 
=======
            in_df= metadata, 
>>>>>>> master
            loopcounter=idx, 
            out_dir=target_s2_archive,
            **kwargs
            ) for idx in range(metadata.shape[0]))
    logger.info(f'Time difference of {time.time()-t} seconds.')

    # concatenate the metadata of the stacked image files into a pandas dataframe
    bandstack_meta = pd.DataFrame(result)

    # merge blackfill scenes (data take issue) if any
<<<<<<< HEAD
    # TODO: this section is for sure buggy and there is a problem in the datamodel...
    if not meta_blackfill.empty:
        # after regular scene processsing, process the blackfill scenes single-threaded
        for date in meta_blackfill.sensing_date.unique():
            scenes = meta_blackfill[meta_blackfill.sensing_date == date]
            product_id_1 = scenes.product_uri.iloc[0]
            product_id_2 = scenes.product_uri.iloc[1]
            # reconstruct storage location
            raw_data_archive = reconstruct_path(record=scenes.iloc[0]).parent
            scene_1 = raw_data_archive.joinpath(product_id_1)
            scene_2 = raw_data_archive.joinpath(product_id_2)
            # the usual naming problem witht the .SAFE directories
            if not scene_1.exists():
                product_id_1 = Path(str(product_id_1.replace('.SAFE', '')))
            if not scene_2.exists():
                product_id_2 = Path(str(product_id_2.replace('.SAFE', '')))
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

        # also the storage location shall be inserted into the database later
        bandstack_meta['storage_location'] = target_s2_archive

=======
    if not meta_blackfill.empty:
        is_mundi = kwargs.get('is_mundi', False)
        # after regular scene processsing, process the blackfill scenes single-threaded
        for date in meta_blackfill.SENSING_DATE.unique():
            scenes = meta_blackfill[meta_blackfill.SENSING_DATE == date]
            scene_id_1 = scenes.PRODUCT_URI.iloc[0]
            scene_id_2 = scenes.PRODUCT_URI.iloc[1]
            if is_mundi:
                scene_id_1 = scene_id_1.replace('.SAFE', '')
                scene_id_2 = scene_id_2.replace('.SAFE', '')
            scene_1 = os.path.join(raw_data_archive, scene_id_1)
            scene_2 = os.path.join(raw_data_archive, scene_id_2)
            res = merge_split_scenes(scene_1=scene_1,
                                        scene_2=scene_2,
                                        out_dir=target_s2_archive, 
                                        **kwargs)

        # append blackfill scenes to bandstack_meta
        df_row = {
            'SENSING_DATE': pd.to_datetime(date).date(),
            'TILE': meta_blackfill.TILE.iloc[0],
            'FPATH_BANDSTACK': os.path.basename(
                res['bandstack']
            ),
            'FPATH_SCL': os.path.join(
                Settings.SUBDIR_SCL_FILES,
                os.path.basename(res['scl'])
            ),
            'FPATH_RGB_PREVIEW': os.path.join(
                Settings.SUBDIR_RGB_PREVIEWS,
                os.path.basename(res['preview'])
            ),
            'PRODUCT_URI': meta_blackfill.PRODUCT_URI.iloc[0],
            'SCENE_ID': meta_blackfill.SCENE_ID.iloc[0]
        }
        # bandstack_meta = bandstack_meta.append(
        #     df_row,
        #     ignore_index=True
        # )
        
>>>>>>> master
        # move to target archive
        shutil.move(
            res['bandstack'],
            os.path.join(
                target_s2_archive,
                os.path.basename(
                    res['bandstack']
                )
            )
        )
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
    
        # remove working directory in the end
        shutil.rmtree(
            os.path.join(
                target_s2_archive,
                'temp_blackfill'
            )
        )
    
<<<<<<< HEAD
    # write metadata of all stacked files to CSV
    return bandstack_meta
=======
        # write metadata of all stacked files to CSV
        bandstack_meta.to_csv(
            os.path.join(
                target_s2_archive,
                Settings.RESAMPLED_METADATA_FILE
            )
        )
>>>>>>> master
