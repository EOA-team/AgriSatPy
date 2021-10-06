
from agrisatpy.metadata.sentinel2 import loop_s2_archive
from agrisatpy.metadata.sentinel2.database import meta_df_to_database
from agrisatpy.metadata.sentinel2.database import ql_info_to_database
from agrisatpy.metadata.sentinel2.database import update_raw_metadata
from pathlib import Path

import agrisatpy


if __name__ == '__main__':

    years = [2019]
    processing_levels = ['L1C']
    region = 'CH'
    # set to True if noise model parameters are required (slow!)
    extract_datastrip = True
    # set to True to UPDATe existing entries in the database
    update_only = True
    update_cols = ['datatakeidentifier']
    
    for processing_level in processing_levels:
        for year in years:

            sat_dir = Path(
                f'/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Rawdata/{processing_level}/{region}/{year}'
            )

            metadata, datastrip_ql = loop_s2_archive(
                in_dir=sat_dir,
                extract_datastrip=extract_datastrip
            )

            metadata['storage_device_ip'] = '//hest.nas.ethz.ch/green_groups_kp_public'
            metadata['storage_device_ip_alias'] = '//nas12.ethz.ch/green_groups_kp_public'
            metadata['storage_share'] = metadata['storage_share'].apply(lambda x: x.replace('/home/graflu/public/',''))

            # write metadata to database
            if extract_datastrip:
                ql_info_to_database(ql_df=datastrip_ql)

            if not update_only:
                meta_df_to_database(meta_df=metadata)
            else:
                update_raw_metadata(
                    meta_df=mmetadata,
                    columns_to_update=update_cols
                )

            # save to CSV as a backup and to support non-database based access
            fname_csv = sat_dir.joinpath('metadata.csv')
            metadata.to_csv(str(fname_csv))
