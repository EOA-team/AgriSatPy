'''
Mapping module for Sentinel-2 data
'''

import geopandas as gpd
import uuid

from datetime import date
import pandas as pd
from pathlib import Path
from shapely.geometry import box
from sqlalchemy.exc import DatabaseError
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from agrisatpy.io.sentinel2 import Sentinel2Handler
from agrisatpy.metadata.sentinel2.database.querying import find_raw_data_by_bbox
from agrisatpy.operational.mapping.mapper import Mapper, Feature
from agrisatpy.operational.resampling.utils import identify_split_scenes
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels
from agrisatpy.utils.exceptions import InputError, BlackFillOnlyError,\
    DataNotFoundError
from agrisatpy.metadata.sentinel2.utils import identify_updated_scenes
from agrisatpy.metadata.utils import reconstruct_path


return_types = ['xarray', 'GeoDataFrame']


class Sentinel2Mapper(Mapper):
    """
    Spatial mapping class for Sentinel-2 data.

    :attrib processing_level:
        Sentinel-2 data processing level (L1C or L2A)
    :attrib cloud_cover_threshold:
        global (scene-wide) cloud coverage threshold between 0 and 100% cloud cover.
        Scenes with cloud coverage reported higher than the threshold are discarded.
        To obtain *all* scenes in the archive use the default of 100%.
    :attrib use_latest_pdgs_baseline:
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
            cloud_cover_threshold: Optional[Union[int,float]] = 100,
            use_latest_pdgs_baseline: Optional[bool] = True,
            *args,
            **kwargs
        ):
        """
        Initializes a new Sentinel-2 Mapper object

        :param processing_level:
            Sentinel-2 processing level to query
        :param cloud_cover_threshold:
            cloud cover threshold in percent (0-100%). Default is 100% to
            consider all scenes in the archive
        :param use_latest_pdgs_baseline:
            if True (default) forces *AgriSatPy* to use the latest processing
            baseline in case a scene is available in different processing levels
        :param args:
            arguments to pass to the constructor of the ``Mapper`` super class
        :param kwargs:
            key-word arguments to pass to the constructor of the ``Mapper``
            super class
        """
        # initialize super-class
        Mapper.__init__(self, *args, **kwargs)

        object.__setattr__(self, 'processing_level', processing_level)
        object.__setattr__(self, 'cloud_cover_threshold', cloud_cover_threshold)
        object.__setattr__(self, 'use_latest_pdgs_baseline', use_latest_pdgs_baseline)

    def get_scenes(
            self,
            tile_ids: Optional[List[str]] = None,
            check_baseline: Optional[bool] = True
        ) -> None:
        """
        Queries the Sentinel-2 metadata DB for a selected time period and
        feature collection.

        NOTE:
            By passing a list of Sentinel-2 tiles you can explicitly control
            which Sentinel-2 tiles are considered. This might be useful for
            mapping tasks where your feature collection lies completely within
            a single Sentinel-2 tile but also overlaps with neighboring tiles. 

        The scene selection and processing workflow contains several steps:
    
        1.  Query the metadata DB for **ALL** available scenes that overlap
            the bounding box of a given ``Polygon`` or ``MultiPolygon``
            feature. **IMPORTANT**: By passing a list of Sentinel-2 tiles
            to consider (``tile_ids``) you can explicitly control which
            Sentinel-2 tiles are considered! 
        2.  Check if for a single sensing date several scenes are available
        3.  If yes check if that's due to Sentinel-2 data take or tiling grid
            design. If yes flag these scenes as potential merge candidates. A
            second reason for multiple scenes are differences in PDGS baseline,
            i.e., the dataset builds upon the **same** Sentinel-2 data but
            was processed by different base-line version. In this case the
            `use_latest_pdgs_baseline` flag becomes relevant. By default we keep
            those scenes with the latest processing baseline and discard the
            other scenes with older baseline version. NOTE: By setting
            ``check_baseline=False`` you can force AgriSatPy to ignore baseline
            checks (in this case all baselines available will be returned)
        4.  If the scenes found have different spatial coordinate systems (CRS)
            (usually different UTM zones) flag the data accordingly. The target
            CRS is defined as that CRS the majority of scenes shares.

        :param tile_id:
            optional list of Sentinel-2 tils (e.g., ['T32TMT','T32TGM']) to use for
            filtering. Only scenes belonging to these tiles are returned then
        :param check_baseline:
            if True (default) checks if a scene is available in different PDGS
            baseline versions
        """
        # read features and loop over them to process each feature separately
        is_file = isinstance(self.feature_collection, Path) or \
                    isinstance(self.feature_collection, str)
        if is_file:
            try:
                aoi_features = gpd.read_file(self.feature_collection)
            except Exception as e:
                raise InputError(
                    f'Could not read polygon features from file ' \
                    f'{self.aoi_features}: {e}'
                )
        else:
            aoi_features = self.feature_collection.copy()
    
        # for the DB query, the geometries are required in geographic coordinates
        # however, we keep the original coordinates as well to avoid to many reprojections
        aoi_features['geometry_wgs84'] = aoi_features['geometry'].to_crs(4326)
    
        # check if there is a unique feature column
        # otherwise create it using uuid
        if self.unique_id_attribute is not None:
            if not aoi_features[self.unique_id_attribute].is_unique:
                raise ValueError(
                    f'"{self.unique_id_attribute}" is not unique for each feature'
                )
        else:
            object.__setattr__(self, 'unique_id_attribute', '_uuid4')
            aoi_features[self.unique_id_attribute] = [
                str(uuid.uuid4()) for _ in aoi_features.iterrows()
            ]

        s2_scenes = {}
        # features implement the __geo_interface__
        features = []
        # extract all properties from the input features to preserve them
        cols_to_ignore = [self.unique_id_attribute, 'geometry', 'geometry_wgs84']
        property_columns = [x for x in aoi_features.columns if x not in cols_to_ignore]
        for _, feature in aoi_features.iterrows():
    
            feature_uuid = feature[self.unique_id_attribute]
            properties = feature[property_columns].to_dict()
            feature_obj = Feature(
                identifier=feature_uuid,
                geom=feature['geometry'],
                epsg=aoi_features.crs.to_epsg(),
                properties=properties
            )
            features.append(feature_obj.to_gdf())

            # determine bounding box of the current feature using
            # its representation in geographic coordinates
            bbox = box(*feature.geometry_wgs84.bounds)

            # use the resulting bbox to query the bounding box
            # TODO: if only a tile is passed query by tile, only!
            try:
                scenes_df = find_raw_data_by_bbox(
                    date_start=self.date_start,
                    date_end=self.date_end,
                    processing_level=self.processing_level,
                    bounding_box=bbox,
                    cloud_cover_threshold=self.cloud_cover_threshold
                )
            except Exception as e:
                raise DatabaseError(f'Querying metadata DB failed: {e}')

            # filter by tile if required
            if tile_ids is not None:
                other_tile_idx = scenes_df[~scenes_df.tile_id.isin(tile_ids)].index
                scenes_df.drop(other_tile_idx, inplace=True)
    
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
            if check_baseline:
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
                ] = most_common_epsg[0]

            # add the scenes_df DataFrame to the dictionary that contains the data for
            # all AOIs (features)
            s2_scenes[feature_uuid] = scenes_df

        # create feature collection
        features_gdf = pd.concat(features)
        # append raw scene count
        features_gdf['raw_scene_count'] = features_gdf.apply(
            lambda x, s2_scenes=s2_scenes: s2_scenes[x.name].shape[0], axis=1
        )
        features = features_gdf.__geo_interface__
        object.__setattr__(self, 'observations', s2_scenes)
        object.__setattr__(self, 'feature_collection', features)

    def _read_multiple_scenes(
            self,
            scenes_date: pd.DataFrame,
            feature_gdf: gpd.GeoDataFrame
        ) -> Union[gpd.GeoDataFrame, Sentinel2Handler]:
        """
        Backend method for processing and reading scene data if more than one scene
        is available for a given sensing date and area of interest
        """
        
        candidate_scene_ids = scenes_date.scene_id.astype(list)
        feature_id = feature_gdf['identifier'].iloc[0].values

        res = None

        # if the feature is a point we take the data set that is not blackfilled
        # if more than one data set is not blackfilled  we simply take the
        # first data set
        if feature_gdf['geometry'].iloc[0].type == 'Point':
            for _, candidate_scene in scenes_date.iterrows():

                feature_gdf = Sentinel2Handler.read_pixels_from_safe(
                    point_features=feature_gdf,
                    in_dir=candidate_scene.real_path,
                    band_selection=self.mapper_configs.band_names
                )
                # a empty data frame indicates black-fill
                if feature_gdf.empty:
                    continue

                # otherwise we take the first non-blackfilled pixel values
                selected_scene_id = candidate_scene.scene_id
                res = feature_gdf
                break
        # in case of a (Multi-)Polygon: check if one of the candidate scenes complete
        # contains the feature (i.e., its bounding box). If that's the case and the
        # returned data is not black-filled, we can take that data set. If none of the
        # candidate contains the scene complete, merging and (depending on the CRS)
        # re-reprojection might be required. The result is then saved to disk in a temporary
        # directory.
        else:
            pass

        # delete those scenes in the observations dataframe that were not used
        candidate_scene_ids.remove(selected_scene_id)
        drop_idx = self.observations[feature_id][
            self.observations[feature_id].scene_id.isin(candidate_scene_ids)
        ].index
        self.observations[feature_id].drop(drop_idx, inplace=True)

        return res


    def get_observation(
            self,
            feature_id: Any,
            sensing_date: date
        ) -> Union[gpd.GeoDataFrame, Sentinel2Handler, None]:
        """
        Returns the scene data (observations) for a selected feature and date.
        
        If for the date provided no scenes are found, the data from the scene(s)
        closest in time is returned

        :param feature_id:
            identifier of the feature for which to extract observations
        :param sensing_date:
            date for which to extract observations (or the closest date if
            no observations are available for the given date)
        :returns:
            depending on the geometry type of the feature either a
            ``GeoDataFrame`` (geometry type: ``Point``) or ``Sentinel2Handler``
            (geometry types ``Polygon`` or ``MultiPolygon``) is returned. if
            the observation contains nodata, only, None is returned.
        """
        # define variable for returning results
        res = None

        # get available observations for the AOI feature
        scenes_df = self.observations.get(feature_id, None)
        if scenes_df is None:
            raise DataNotFoundError(
                f'Could not find any scenes for feature with ID "{feature_id}"'
            )
        
        # get scene(s) closest to the sensing_date provided
        scenes_date = scenes_df.iloc[[
            abs((scenes_df.sensing_date - sensing_date)).argmin()
        ]]

        # map the dataset path(s)
        try:
            scenes_date['real_path'] = scenes_date.apply(
                lambda x: reconstruct_path(record=x), axis=1
            )
        except Exception as e:
            raise DataNotFoundError(
                f'Cannot find the scenes on the file system: {e}'
            )
        # get properties and geometry of the current feature from the collection
        feature_dict = self.get_feature(feature_id)
        feature_gdf = gpd.GeoDataFrame.from_features(feature_dict)
        feature_gdf.crs = feature_dict['features'][0]['properties']['epsg']

        # multiple scenes for a single date
        # check what to do (re-projection, merging)
        if scenes_date.shape[0] > 1:
            res = self._read_multiple_scenes(
                scenes_date=scenes_date,
                feature_dict=feature_dict
            )

        else:
            # if there is only one scene all we have to do is to read
            # read pixels in case the feature's dtype is point
            if feature_dict['features'][0]['geometry']['type'] == 'Point':
                res = Sentinel2Handler.read_pixels_from_safe(
                    point_features=feature_gdf,
                    in_dir=scenes_date['real_path'].iloc[0],
                    band_selection=self.mapper_configs.band_names
                )
            # or the feature
            else:
                handler = Sentinel2Handler()
                try:
                    handler.read_from_safe(
                        in_dir=scenes_date['realpath'].iloc[0],
                        aoi_features=feature_gdf,
                        band_selection=self.mapper_configs.band_names,
                        full_bounding_box_only=True,
                        int16_to_float=False
                    )
                except BlackFillOnlyError:
                    return res
                except Exception as e:
                    raise Exception from e
                res = handler

        # append date to GeoDataFrame
        if isinstance(res, gpd.GeoDataFrame):
            res['sensing_date'] = scenes_date['sensing_date']
            res['scene_id'] = scenes_date['scene_id']

        return res

    def get_complete_timeseries(
            self,
            feature_selection: Optional[List[Any]] = None
        ) -> None:
        """
        Extracts all observation with a time period for a feature collection.

        This function takes the Sentinel-2 scenes retrieved from the metadata DB query
        in `~Mapper.get_sentinel2_scenes` and extracts the Sentinel-2 data from the
        original .SAFE archives for all available scenes.
    
        :param feature_selection:
            optional subset of features (you can only select features included
            in the current feature collection)
        """

        assets = {}
        # loop over features (AOIs) in feature dict
        for feature, scenes_df in self.observations.items():

            # in case a feature selection is available check if the current
            # feature is part of it
            if feature_selection is not None:
                if feature in feature_selection:
                    continue

            # loop over scenes, they are already ordered by date (ascending)
            # and check for each date which scenes are relevant and require
            # potential reprojection or merging
            sensing_dates = scenes_df.sensing_date.unique()
            feature_res = []
            for sensing_date in sensing_dates:
                res = self.get_observation(
                    feature,
                    sensing_date
                )
                feature_res.append(res)

            # if res is a GeoDataFrame the list can be concated
            if isinstance(res, gpd.GeoDataFrame):
                assets[feature] = pd.concat(feature_res)

        return assets
