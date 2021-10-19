
from pathlib import Path
from datetime import date

from agrisatpy.metadata.sentinel2 import loop_s2_archive
from agrisatpy.metadata.sentinel2.database import meta_df_to_database
from agrisatpy.metadata.sentinel2.database import ql_info_to_database
from agrisatpy.metadata.sentinel2.database import update_raw_metadata
import agrisatpy


if __name__ == '__main__':

    years = [2018]
    processing_levels = ['L2A']
    region = 'CH'
    # set to True if noise model parameters are required (slow!)
    extract_datastrip = False
    # set to True to UPDATe existing entries in the database
    update_only = False
    update_cols = ['reflectance_conversion']

    # only update latest datasets
    last_execution_date = date(2021,10,10)
    get_newest_datasets = True
    
    for processing_level in processing_levels:
        for year in years:

            sat_dir = Path(
                f'/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Rawdata/{processing_level}/{region}/{year}'
            )

            metadata, datastrip_ql = loop_s2_archive(
                in_dir=sat_dir,
                extract_datastrip=extract_datastrip,
                get_newest_datasets=get_newest_datasets,
                last_execution_date=last_execution_date
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
                    meta_df=metadata,
                    columns_to_update=update_cols
                )

            # save to CSV as a backup and to support non-database based access
            fname_csv = sat_dir.joinpath('metadata.csv')
            metadata.to_csv(str(fname_csv))
