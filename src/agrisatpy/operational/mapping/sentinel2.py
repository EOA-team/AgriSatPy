'''
Mapping module for Sentinel-2 data
'''

import geopandas as gpd
import uuid

from pathlib import Path
from shapely.geometry import box
from sqlalchemy.exc import DatabaseError
from typing import Optional
from typing import Union

from agrisatpy.metadata.sentinel2.database.querying import find_raw_data_by_bbox
from agrisatpy.operational.mapping.mapper import Mapper
from agrisatpy.operational.resampling.utils import identify_split_scenes
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels
from agrisatpy.utils.exceptions import InputError
from agrisatpy.utils.types import S2Scenes
from agrisatpy.metadata.sentinel2.utils import identify_updated_scenes


class Sentinel2Mapper(Mapper):
    """
    Spatial mapping class for Sentinel-2 data.

    :param processing_level:
        Sentinel-2 data processing level (L1C or L2A)
    :param local_cloud_cover_threshold:
        local, i.e., AOI-specific, cloud cover threshold between 0 and 100%.
        Instead of relying on global (scene-wide) cloud coverage information,
        the cloud coverage is calculated within the AOI bounds (L2A processing level).
        In case of L1C processing level the cloud cover threshold is applied globally
        to pre-filter data-set based on scene-wide cloud cover information.
    :param use_latest_pdgs_baseline:
        since a scene can possible occur in different PDGS baseline numbers (the scene_id
        and product_uri will be different, which is supported by our data model)
        it is necessary to decide for a baseline in those cases where multiple scenes
        from the same sensing and data take time are available (originating from the
        same Sentinel-2 data but processed slightly differently). By default we use
        those scenes from the latest baseline. Otherwise, it is possible to use
        the baseline most scenes were processed with.
    """

    def __init__(
            self,
            processing_level: ProcessingLevels,
            local_cloud_cover_threshold: Optional[Union[int,float]] = 100,
            use_latest_pdgs_baseline: Optional[bool] = True,
            *args,
            **kwargs
        ):

        # initialize super-class
        Mapper.__init__(self, *args, **kwargs)
    
        self.processing_level = processing_level
        self.local_cloud_cover_threshold = local_cloud_cover_threshold
        self.use_latest_pdgs_baseline = use_latest_pdgs_baseline
        self.s2_scenes = S2Scenes({})


    def get_sentinel2_scenes(self) -> None:
        """
        Queries the Sentinel-2 metadata DB for a selected time period and
        field parcel geometry (masking all pixels not belonging to the parcel) and
        returns a ``DataFrame`` with entries for all scenes found.
    
        The scene selection and processing workflow contains several steps:
    
        1.  Query the metadata DB for **ALL** available scenes that overlap
            the bounding box of a given ``Polygon`` or ``MultiPolygon``
            feature
        2.  Check if for a single sensing date several scenes are available
        3.  If yes check if that's due to Sentinel-2 data take or tiling grid
            design. If yes flag these scenes as potential merge candidates. A
            second reason for multiple scenes are differences in PDGS baseline,
            i.e., the dataset builds upon the **same** Sentinel-2 data but
            was processed by different base-line version. In this case the
            `use_latest_pdgs_baseline` flag becomes relevant. By default we keep
            those scenes with the latest processing baseline and discard the
            other scenes with older baseline version.
        4.  If the scenes found have different spatial coordinate systems (CRS)
            (usually different UTM zones) flag the data accordingly. The target
            CRS is defined as that CRS the majority of scenes shares.
    
        In the returned ``DataFrame`` two attributes are added for each scene that
        are required for the further AOI extraction steps:
    
        *    `is_split` is a boolean flag indicating that for one image acquisition
            date several Sentinel-2 scenes from different data takes or tiles must be
            considered when extracting the Sentinel-2 data for that AOI
        *   `target_crs` is a integer attribute denoting the target EPSG code of a
            Sentinel-2 scene. If there are different spatial CRS and the scene is not
            in that CRS the majority of scenes has the `target_crs` differs from the
            `epsg` attribute. This indicates the subsequent extraction function that
            the scene must be reprojected from the original `epsg` into the
            `target_epsg`.
    
        
        :return:
            dictionary with found Sentinel-2 scenes including the `is_split` and
            `target_epsg` entry for each AOI (polygon feature) in `polygon_features`.
        """
    
        # read features and loop over them to process each feature separately
        is_file = isinstance(self.polygon_features, Path) or \
                    isinstance(self.polygon_features, str)
        if is_file:
            try:
                self.polygon_features = gpd.read_file(self.polygon_features)
            except Exception as e:
                raise InputError(
                    f'Could not read polygon features from file ' \
                    f'{self.polygon_features}: {e}'
                )
    
        # for the DB query, the geometries are required in geographic coordinates
        # however, we keep the original coordinates as well to avoid to many reprojections
        self.polygon_features['geometry_wgs84'] = \
            self.polygon_features['geometry'].to_crs(4326)
    
        # check if there is a unique feature column
        # otherwise create it using uuid
        if self.unique_id_attribute is not None:
            if not self.polygon_features[self.unique_id_attribute].is_unique:
                raise ValueError(
                    f'"{self.unique_id_attribute}" is not unique for each feature'
                )
        else:
            self.unique_id_attribute = '_uuid4'
            self.polygon_features[self.unique_id_attribute] = [
                uuid.uuid4() for _ in self.polygon_features.iterrows()
            ]
    
        for _, feature in self.polygon_features.iterrows():
    
            feature_uuid = str(feature[self.unique_id_attribute])
    
            # get available Sentinel-2 data-sets using the maximum cloud coverage
            cloud_cover_threshold = 100
            if self.processing_level.name == 'L1C':
                cloud_cover_threshold = self.local_cloud_cover_threshold
    
            # determine bounding box of the current feature (AOI)
            bbox = box(*feature.geometry_wgs84.bounds)
    
            # use the resulting bbox to query the bounding box
            try:
                scenes_df = find_raw_data_by_bbox(
                    date_start=self.date_start,
                    date_end=self.date_end,
                    processing_level=self.processing_level,
                    bounding_box=bbox,
                    cloud_cover_threshold=cloud_cover_threshold
                )
            except Exception as e:
                raise DatabaseError(f'Querying metadata DB failed: {e}')
    
            if scenes_df.empty:
                raise UserWarning(
                    f'The query for feature {feature_uuid} returned now results'
                )
                continue
    
            # check if the satellite data is in different projections
            in_single_crs = scenes_df.epsg.unique().shape[0] == 1
    
            # check if there are several scenes available for a single sensing date
            # in this case merging of different datasets might be necessary
            scenes_df_split = identify_split_scenes(scenes_df)
            scenes_df['is_split'] = False
    
            # check if the data comes from different PDGS baseline versions
            # by default, we always aim for the highest baseline version
            if not scenes_df_split.empty:
                scenes_df_updated_baseline = identify_updated_scenes(
                    metadata_df=scenes_df_split,
                    return_highest_baseline=self.use_latest_pdgs_baseline
                )
                # drop those scenes that were dis-selected because of the PDGS baseline
                drop_idx = scenes_df[
                    scenes_df.product_uri.isin(scenes_df_split.product_uri) &
                    ~scenes_df.product_uri.isin(scenes_df_updated_baseline.product_uri)
                ].index
                if not drop_idx.empty:
                    scenes_df.drop(drop_idx, inplace=True)
                # check again for split scenes; these scenes are then "really" split
                scenes_df_split = identify_split_scenes(scenes_df)
                if not scenes_df_split.empty:
                    scenes_df.loc[
                        scenes_df.product_uri.isin(scenes_df_split.product_uri),
                        'is_split'
                    ] = True
    
            # in case the scenes have different projections (most likely different UTM
            # zone numbers) figure out which will be target UTM zone. To avoid too many
            # reprojection operations of raster data later, the target CRS is that CRS
            # most scenes have (expressed as EPSG code)
            scenes_df['target_crs'] = scenes_df.epsg
            if not in_single_crs:
                most_common_epsg = scenes_df.epsg.mode().values
                scenes_df.loc[
                    ~scenes_df.epsg.isin(most_common_epsg),
                    'target_crs'
                ] = most_common_epsg
        
            # add the scenes_df DataFrame to the dictionary that contains the data for
            # all AOIs (features)
            self.s2_scenes[feature_uuid] = scenes_df


    def extract_aois_from_s2(
            self,
            band_selection: Optional[str] = None,
            **reducers_options
        ):
        """
        This function takes the dictionary with Sentinel-2 scenes retrieved from
        the metadata DB query in `~Mapper.get_sentinel2_scenes` and extracts the
        Sentinel-2 data from the original .SAFE archives
    
        :param band_selection:
            selection of Sentinel-2 bands to extract. Per default all 10 and 20m are
            read
        """
        pass
        
