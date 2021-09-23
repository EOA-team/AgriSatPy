"""
sample script showing how to create the file system structure
for storing AgriSatPy satellite data
"""

from pathlib import Path
from agrisatpy.archive.sentinel2 import create_archive_struct


if __name__ == '__main__':

    datadir = Path('/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Processed')
    year_selection = [2016, 2017, 2018, 2019, 2020, 2021]
    processing_levels = ['L1C', 'L2A']
    tile_selection = [
        'T32TPT',
        'T32TMS',
        'T32TLT',
        'T32TNR',
        'T32TLS',
        'T31UGP',
        'T32TPR',
        'T32UMU',
        'T32ULU',
        'T32TMR',
        'T32TMT',
        'T32TNS',
        'T32TPS',
        'T32UPU',
        'T32TLR',
        'T31TGN',
        'T31TGM',
        'T32UNU',
        'T32TNT',
        'T31TGL'
    ]
    
    create_archive_struct(
        datadir,
        processing_levels,
        tile_selection,
        year_selection
    )
