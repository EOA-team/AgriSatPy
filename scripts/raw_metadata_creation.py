
from agrisatpy.metadata.sentinel2 import loop_s2_archive
from agrisatpy.metadata.sentinel2.database import meta_df_to_database
from pathlib import Path


if __name__ == '__main__':

    years = [2017]
    processing_levels = ['L2A']
    region = 'CH'
    
    for processing_level in processing_levels:
        for year in years:

            sat_dir = Path(
                f'/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Rawdata/{processing_level}/{region}/{year}'
            )

            metadata = loop_s2_archive(in_dir=sat_dir)
            metadata['storage_device_ip'] = '//hest.nas.ethz.ch/green_groups_kp_public'
            metadata['storage_device_ip_alias'] = '//nas12.ethz.ch/green_groups_kp_public'
            metadata['storage_share'] = metadata['storage_share'].apply(lambda x: x.replace('/home/graflu/public/',''))
            meta_df_to_database(meta_df=metadata)

            # save to CSV as a backup and to support non-database based access
            fname_csv = sat_dir.joinpath('metadata.csv')
            metadata.to_csv(str(fname_csv))
