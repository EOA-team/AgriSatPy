'''
Tests the automatic creation and update of an archive structure
for stored spatially re-sampled, band-stacked satellite imagery
'''

import os
import pytest

from agrisatpy.archive.sentinel2 import create_archive_struct


def test_archive_creation(datadir):
    """
    tests the creation of a new SAT archive and tries
    to update it with another tile and another year
    """
    year_selection = [2019, 2020, 2021]
    processing_levels = ['L1C', 'L2A']
    tile_selection = ['T31TGM', 'T32TMT']

    create_archive_struct(datadir,
                          processing_levels,
                          tile_selection,
                          year_selection)

    # check if all sub-directories were created
    for proc_level in processing_levels:
        for year in year_selection:
            for tile in tile_selection:
                expr = proc_level + os.sep + str(year) + os.sep + tile
                subdir = os.path.join(datadir, expr)
                assert os.path.isdir(subdir), \
                f'No sub-directory found for {proc_level} / {year} / {tile}: Expected: {subdir}'

    # add a tile afterwards
    tile = 'T32TLS'
    tile_selection.append(tile)

    create_archive_struct(datadir,
                          processing_levels,
                          tile_selection,
                          year_selection)

    # check if the tile was added to all processing levels and years
    for proc_level in processing_levels:
        for year in year_selection:
            expr = proc_level + os.sep + str(year) + os.sep + tile
            subdir = os.path.join(datadir, expr)
            assert os.path.isdir(subdir), \
            f'No sub-directory found for {proc_level} / {year} / {tile}: Expected: {subdir}'
    

    # add a year afterwards
    year = 2018
    year_selection.append(year)

    create_archive_struct(datadir,
                          processing_levels,
                          tile_selection,
                          year_selection)

    # check if the year was added to all processing levels and tiles
    for proc_level in processing_levels:
        for tile in tile_selection:
            expr = proc_level + os.sep + str(year) + os.sep + tile
            subdir = os.path.join(datadir, expr)
            assert os.path.isdir(subdir), \
            f'No sub-directory found for {proc_level} / {year} / {tile}: Expected: {subdir}'
