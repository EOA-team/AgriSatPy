'''
Sentinel-2 records data in so-called datatakes. When a datatake is over and a new
begins the acquired image data is written to different files (based on the datatake
time). Sometimes, this cause scenes of a single acquisition date to be split into
two datasets which differ in their datatake. Thus, both datasets have a reasonable
amount of blackfill in those areas not covered by the datatake they belong to. For
users of satellite data, however, it is much more convenient to have those split
scenes merged into one since the division into two scenes by the datatake has
technical reasons only.
'''

import shutil
from pathlib import Path


from .resample_and_stack import resample_and_stack_s2
from agrisatpy.config import get_settings
from agrisatpy.analysis.mosaicing import merge_datasets

Settings = get_settings()
logger = Settings.logger

def merge_split_scenes(
        scene_1: Path,
        scene_2: Path,
        out_dir: Path,
        **kwargs
    ) -> dict:
    """
    merges two Sentinel-2 datasets in .SAFE formatof the same sensing date and tile
    split by the datatake beginning/ end.

    First, both scenes are resampled to 10m and stacked in a temporary working directory;
    second, they are merged together so that the blackfill of the first scenes is replaced
    by the values of the second one.
    SCL (if available) and previews are managed accordingly.

    :param scene_1:
        .SAFE directory containing the first of two scenes split by the datatake beginning/
        end of Sentinel-2
    :param scene_2:
        .SAFE directory containing the first of two scenes split by the datatake beginning/
        end of Sentinel-2
    :param out_dir:
        directory where to create a "working_dir" sub-directory to save the 
        intermediate products to.
    :param kwargs:
        key word arguments to pass to resample_and_stack_S2.
    """

    # check if working directory exists
    working_dir = out_dir.joinpath('temp_blackfill')
    if not working_dir.exists():
        working_dir.mkdir()

    # save the outputs of the two scenes to different sub-directories within the working
    # directory to avoid to override the output
    out_dirs = [working_dir.joinpath('1'), working_dir.joinpath('2')]
    for out_dir in out_dirs:
        if out_dir.exists():
            # for clean start of next loop iteration
            shutil.rmtree(out_dir)
        out_dir.mkdir()

    # do the spatial resampling for the two scenes
    # first scene
    try:
        scene_out_1 = resample_and_stack_s2(
            in_dir=scene_1,
            out_dir=out_dirs[0],
            **kwargs
        )
    except Exception as e:
        logger.error(f'Resampling of {scene_1} failed: {e}')
        return {}

    # second scene
    try:
        scene_out_2 = resample_and_stack_s2(
            in_dir=scene_2,
            out_dir=out_dirs[0],
            **kwargs
        )
    except Exception as e:
        logger.error(f'Resampling of {scene_1} failed: {e}')
        return {}

    # merge band stacks
    logger.info(f'Starting merging files {scene_out_1} and {scene_out_2}')

    # merged file has some name as the first scene
    out_file = working_dir.joinpath(scene_out_1['bandstack'])
    datasets = [scene_out_1['bandstack'], scene_out_2['bandstack']]

    try:
        merge_datasets(
            datasets=datasets,
            out_file=out_file
        )
    except Exception as e:
        logger.error(f'Merging of {datasets[0]} and {datasets[1]} failed: {e}')
        return {}

    # set band descriptions to output

    # generate RGB and SCL previews (if available)
    