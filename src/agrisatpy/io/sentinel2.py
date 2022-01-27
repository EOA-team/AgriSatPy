'''
This module contains the ``Sentinel2Handler`` class that inherits from
AgriSatPy's ``SatDataHandler`` class.

The ``Sentinel2Handler`` enables reading one or more spectral bands from Sentinel-2
data. The data can be either band-stacked (i.e., AgriSatPy derived format) or in .SAFE
format which is ESA's standard format for distributing Sentinel-2 data.

The class handles data in L1C and L2A processing level.
'''

import numpy as np
import pandas as pd
import geopandas as gpd

from matplotlib.pyplot import Figure
from matplotlib import colors
from pathlib import Path
from typing import Optional
from typing import List
from typing import Tuple
from typing import Union

from agrisatpy.io import SatDataHandler
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels
from agrisatpy.utils.constants.sentinel2 import s2_band_mapping
from agrisatpy.utils.constants.sentinel2 import s2_gain_factor
from agrisatpy.utils.constants.sentinel2 import SCL_Classes
from agrisatpy.utils.constants.sentinel2 import band_resolution
from agrisatpy.utils.exceptions import BandNotFoundError, BlackFillOnlyError
from agrisatpy.utils.sentinel2 import get_S2_bandfiles_with_res
from agrisatpy.utils.sentinel2 import get_S2_platform_from_safe
from agrisatpy.utils.sentinel2 import get_S2_sclfile
from agrisatpy.utils.sentinel2 import get_S2_processing_level
from agrisatpy.utils.sentinel2 import get_S2_acquistion_time_from_safe


class Sentinel2Handler(SatDataHandler):
    """
    Class for storing Sentinel-2 band data read from bandstacks or
    .SAFE archives (L1C and L2A level) overwriting methods inherited
    from `~agrisatpy.utils.io.SatDataHandler`.
    """

    def __init__(self, *args, **kwargs):
        SatDataHandler.__init__(self, *args, **kwargs)
        self._is_l2a = True


    @staticmethod
    def _check_band_selection(
            band_selection: List[str]
        ) -> List[Tuple[str,str]]:
        """
        Returns the Sentinel-2 band and color names for a
        given band selection
    
        :param band_selection:
            list of selected bands
        :return:
            list of tuples of color and band names
        """

        # check how many bands were selected
        return [(k,v) for k,v in s2_band_mapping.items() if k in band_selection]


    def _generate_band_aliases(
            self,
            sel_bands: List[str]
        ) -> None:
        """
        Helper method to generate Sentinel-2 band aliases

        :param sel_bands:
            list of selected Sentinel-2 bands
        """
        old_band_names = self.bandnames
        self.reset_bandnames(new_bandnames=sel_bands)
        self.set_bandaliases(
            band_names=self.bandnames,
            band_aliases=old_band_names
        )


    def _int16_to_float(self) -> None:
        """
        Converts Sentinel-2 16 bit integer arrays to float and applies a gain
        factor to scale value between 0 and 1
        """

        for band_name in self.bandnames:
            # SCL band is not converted
            if band_name.upper() == 'SCL':
                continue
            if isinstance(self.data[band_name], np.ma.core.MaskedArray):
                # apply gain factor to data part only
                temp_data = self.data[band_name].data.astype(float) * \
                    s2_gain_factor
                temp_mask = self.data[band_name].mask
                self.data[band_name] = np.ma.masked_array(
                    data=temp_data,
                    mask=temp_mask
                )
            else:
                self.data[band_name] = self.data[band_name].astype(float) * \
                    s2_gain_factor


    def _get_band_files(
            self,
            in_dir: Path,
            band_selection: List[str],
            read_scl: bool
        ) -> pd.DataFrame:
        """
        Returns the file-paths to the selected Sentinel-2 bands in a .SAFE archive
        folder and checks the processing level of the data (L1C or L2A).

        :param in_dir:
            Sentinel-2 .SAFE archive folder from which to read data
        :param band_selection:
            selection of spectral Sentinel-2 bands to read
        :param read_scl:
            if True and the processing level is L2A the scene classification layer
            is read in addition to the spectral bands
        :return:
            ``DataFrame`` with paths to the jp2 files with the spectral band data
        """

        # check processing level
        processing_level = get_S2_processing_level(dot_safe_name=in_dir)

        # define also a local flag of the processing level so this method also works
        # when called from a classmethod
        is_l2a = True
        if processing_level == ProcessingLevels.L1C:
            self._is_l2a = False
            is_l2a = False

        # check if SCL should e read (L2A)
        if is_l2a and read_scl:
            scl_in_selection = 'scl' in band_selection or 'SCL' in band_selection
            if not scl_in_selection:
                band_selection.append('SCL')

        # determine native spatial resolution of Sentinel-2 bands
        band_res = band_resolution[processing_level]
        band_selection_spatial_res = [x for x in band_res.items() if x[0] in band_selection]

        # search band files depending on processing level and spatial resolution(s)
        band_df_safe = get_S2_bandfiles_with_res(
            in_dir=in_dir,
            band_selection=band_selection_spatial_res,
            is_l2a=is_l2a
        )

        return band_df_safe


    def plot_scl(
            self,
            colormap: Optional[str]=''
        ) -> Figure:
        """
        Wrapper around `plot_band` method to plot the Scene Classification
        Layer available from the L2A processing level. Raises an error if
        the band is not available

        :param colormap:
            optional matplotlib named colormap to use for visualization. If
            not provided uses a custom color map that tries to reproduce the
            standard SCL colors provided by ESA.
        :return:
            matplotlib figure object with the SCL band data
            plotted as map
        """

        # check if SCL is a masked array
        # if so, fill masked values with no-data class (for plotting we need to
        # manipulate the data directly)
        if isinstance(self.data['scl'], np.ma.core.MaskedArray):
            self.data['scl'] = self.data['scl'].filled(
                [k for k, v in SCL_Classes.values().items() if v == 'no_data'])

        # make a color map of fixed colors
        if colormap == '':
            
            # get only those colors required (classes in the layer)
            scl_colors = SCL_Classes.colors()
            scl_dict = SCL_Classes.values()
            scl_classes = list(np.unique(self.get_band('scl')))
            selected_colors = [x for idx,x in enumerate(scl_colors) if idx in scl_classes]
            scl_cmap = colors.ListedColormap(selected_colors)
            scl_ticks = [x[1] for x in scl_dict.items() if x[0] in scl_classes]
        try:
            return self.plot_band(
                band_name='scl',
                colormap=colormap,
                discrete_values=True,
                user_defined_colors=scl_cmap,
                user_defined_ticks=scl_ticks
            )
        except Exception as e:
            raise BandNotFoundError(f'Could not plot SCL: {e}')


    def mask_clouds_and_shadows(
            self,
            bands_to_mask: List[str],
            cloud_classes: Optional[List[int]] = [2, 3, 7, 8, 9, 10]
        ) -> None:
        """
        A Wrapper around the inherited ``mask`` method to mask clouds and
        shadows based on the SCL band. Works therefore on L2A data, only.

        Masks the SCL classes 2, 3, 7, 8, 9, 10.

        If another class selection is desired consider using the mask function
        from `agrisatpy.utils.io` directly or change the default
        ``cloud_classes``.

        :param bands_to_mask:
            list of bands on which to apply the SCL mask
        :param cloud_classes:
            list of SCL values to be considered as clouds/shadows and snow.
            By default, all three cloud classes and cloud shadows are considered
            plus snow.
        """

        mask_band = 'scl'
        try:
            self.mask(
                name_mask_band=mask_band,
                mask_values=cloud_classes,
                bands_to_mask=bands_to_mask
            )
        except Exception as e:
            raise Exception(f'Could not apply cloud mask: {e}')


    def get_scl_stats(self) -> pd.DataFrame:
        """
        Returns a ``DataFrame`` with the number of pixel for each
        class of the scene classification layer. Works for data in
        L2A processing level, only.

        :return:
            ``DataFrame`` with pixel count of SCL classes available
            in the currently loaded Sentinel-2 scene. Masked pixels
            are ignored and also not used for calculating the relative
            class occurences.
        """

        # check if SCL is available
        if not 'scl' in self.bandnames:
            raise BandNotFoundError(
                'Could not find scene classification layer. Is scene L2A?'
            )

        scl = self.get_band('SCL')

        # if the scl array is a masked array consider only those pixels
        # not masked out
        if isinstance(scl, np.ma.MaskedArray):
            scl = scl.compressed()

        # overall number of pixels; masked pixels are not considered
        n_pixels = scl.size

        # count occurence of SCL classes
        scl_classes, class_counts = np.unique(scl, return_counts=True)
        class_occurences = dict(zip(scl_classes, class_counts))

        # get SCL class name (in addition to the integer code in the data)
        scl_class_mapping = SCL_Classes.values()
        scl_stats_list = []
        for class_occurence in class_occurences.items():

            # unpack tuple
            class_code, class_count = class_occurence

            scl_stats_dict = {}
            scl_stats_dict['Class_Value'] = class_code
            scl_stats_dict['Class_Name'] = scl_class_mapping[class_code]
            scl_stats_dict['Class_Abs_Count'] = class_count
            # calculate percentage of the class count to overall number of pixels in %
            scl_stats_dict['Class_Rel_Count'] = class_count / n_pixels * 100

            scl_stats_list.append(scl_stats_dict)

        # convert to DataFrame
        scl_stats_df = pd.DataFrame(scl_stats_list)

        # append also those SCL classes not found in the scene so that always
        # all SCL classes are returned (this makes handling the DataFrame easier)
        # there are 12 SCL classes, so if the DataFrame has less rows append the
        # missing classes
        if scl_stats_df.shape[0] < len(scl_class_mapping):
            for scl_class in scl_class_mapping:
                if scl_class not in scl_stats_df.Class_Value.values:

                    scl_stats_dict = {}
                    scl_stats_dict['Class_Value'] = scl_class
                    scl_stats_dict['Class_Name'] = scl_class_mapping[scl_class]
                    scl_stats_dict['Class_Abs_Count'] = 0
                    scl_stats_dict['Class_Rel_Count'] = 0

                    scl_stats_df = scl_stats_df.append(
                        other=scl_stats_dict,
                        ignore_index=True
                    )

        return scl_stats_df
        

    def get_cloudy_pixel_percentage(
            self,
            cloud_classes: Optional[List[int]] = [3, 8, 9, 10]
        ) -> float:
        """
        Calculates the cloudy pixel percentage [0-100] for the current AOI
        (L2A processing level, only) considering all SCL classes that are
        not NoData.

        :param cloud_classes:
            list of SCL values to be considered as clouds. By default,
            all three cloud classes and cloud shadows are considered.
        :return:
            cloudy pixel percentage in the AOI [0-100%] related to the
            overall number of valid pixels (SCL != no_data)
        """

        # get SCL statistics
        scl_stats_df = self.get_scl_stats()

        # sum up pixels labeled as clouds or cloud shadows
        num_cloudy_pixels = scl_stats_df[
            scl_stats_df.Class_Value.isin(cloud_classes)
        ]['Class_Abs_Count'].sum()
        # and all other pixels
        num_noncloudy_pixels = scl_stats_df[
            ~scl_stats_df.Class_Value.isin(cloud_classes)
        ]['Class_Abs_Count'].sum()
        
        # relate it to the overall number of valid pixels in the scene and return
        # the value in %
        cloudy_pixel_percentage = num_cloudy_pixels / num_noncloudy_pixels * 100

        return cloudy_pixel_percentage


    def read_from_bandstack(
            self,
            fname_bandstack: Path,
            processing_level: Optional[ProcessingLevels] = None,
            in_file_scl: Optional[Path] = None,
            polygon_features: Optional[Union[Path, gpd.GeoDataFrame]] = None,
            full_bounding_box_only: Optional[bool] = False,
            int16_to_float: Optional[bool] = True,
            band_selection: Optional[List[str]] = list(s2_band_mapping.keys()),
            read_scl: Optional[bool] = True
        ) -> None:
        """
        Reads Sentinel-2 spectral bands from a band-stacked geoTiff by calling
        ``agrisatpy.io.Sat_Data_Reader`` as super-class. The bandstack is assumed
        to originate from AgriSatPy and its ``resample_and_stack_s2`` function.

        The function works on Sentinel-2 L1C and L2A processing level and reads by
        default all 10 and 20m bands. If the processing level is L2A it also searches
        for the scene classification layer (SCL) and provides it as additional band.

        :param fname_bandstack:
            file-path to the bandstacked geoTiff containing the Sentinel-2
            bands in a single spatial resolution
        :param processing_level:
            specification of the processing level of the Sentinel-2 data. If not
            provided will be determined from the name of the band-stack (works only
            if the file follows the AgriSatPy's naming conventions!)
        :param in_file_scl:
            if the SCL file location is already known (e.g., from AgriSatPy
            database query) or it is located not at its usual location
            pass it directly to the method, otherwise the method
            tries to find the corresponding SCL file from the filesystem assuming
            that AgriSatPy's default file system logic was used.
        :param polygon_features:
            vector file (e.g., ESRI shapefile or geojson) or ``GeoDataFrame`` defining
            geometry/ies for which to extract the Sentinel-2 data. Can contain
            one to many features of type Polygon or MultiPolygon
        :param full_bounding_box_only:
            if set to False, will only extract the data for those geometry/ies
            defined in in_file_aoi. If set to False, returns the data for the
            full extent (hull) of all features (geometries) in in_file_aoi.
        :param int16_to_float:
            if True (Default) converts the original UINT16 Sentinel-2 data
            to numpy floats ranging between 0 and 1. Set to False if you
            want to keep the UINT16 datatype and original value range.
        :param band_selection:
            selection of Sentinel-2 bands to read. Per default, all 10 and
            20m bands are processed. If you wish to read less, specify the
            band names accordingly, e.g., ['B02','B03','B04'] to read only the
            VIS bands.
        :param read_scl:
            read SCL file if available (default, L2A processing level).
        """

        # ignore SCL if provided in band selection. The SCL band is always
        # read when the processing level is L2A
        if 'scl' in band_selection:
            band_selection.remove('scl')
        if 'SCL' in band_selection:
            band_selection.remove('SCL')

        # call method from super class
        try:
            super().read_from_bandstack(
                fname_bandstack=fname_bandstack,
                in_file_aoi=polygon_features,
                full_bounding_box_only=full_bounding_box_only,
                band_selection=band_selection
            )
        except Exception as e:
            raise Exception from e

        # loop over bands, set band and color names and 
        sel_bands = self._check_band_selection(band_selection=band_selection)
        new_bandnames = [x[1] for x in sel_bands]
        old_bandnames = self.bandnames
        self.reset_bandnames(new_bandnames)
        self.set_bandaliases(new_bandnames, old_bandnames)

        # apply type casting if selected
        if int16_to_float:
            self._int16_to_float()
            # update datatype in meta
            self.data['meta']['dtype'] = 'float32'

        # check processing level (for SCL reading)
        if processing_level is None:
            processing_level = get_S2_processing_level(
                dot_safe_name=fname_bandstack
            )

        # read SCL if processing level is L2A and read_scl is True
        if processing_level == ProcessingLevels.L2A:

            if read_scl:

                if in_file_scl is None:
                    try:
                        parent_dir = fname_bandstack.parent
                        in_file_scl = get_S2_sclfile(
                            in_dir=parent_dir,
                            from_bandstack=True,
                            in_file_bandstack=fname_bandstack
                        )
                    except Exception as e:
                        raise Exception(f'Could not find SCL file: {e}')
                        # return if it's not possible to find the SCL
                        return
    
                # read SCL file into new reader and add it as new band
                scl_reader = SatDataHandler()
                try:
                    scl_reader.read_from_bandstack(
                        fname_bandstack=in_file_scl,
                        in_file_aoi=polygon_features,
                        full_bounding_box_only=full_bounding_box_only,
                        blackfill_value=0
                    )
                except Exception as e:
                    raise Exception(f'Could not read SCL file: {e}')
    
                # add SCL as new band
                band_name_scl = scl_reader.bandnames[0]
                self.add_band(
                    band_name='scl',
                    band_data=scl_reader.get_band(band_name_scl),
                    band_alias='SCL',
                    snap_band=self.bandnames[0]
                )
    
                # increase band count in meta
                self.data['meta']['count'] += 1


    def read_from_safe(
            self,
            in_dir: Path,
            polygon_features: Optional[Union[Path, gpd.GeoDataFrame]] = None,
            full_bounding_box_only: Optional[bool] = False,
            int16_to_float: Optional[bool] = True,
            band_selection: Optional[List[str]] = None,
            read_scl: Optional[bool] = True,
            skip_blackfilled_scenes: Optional[bool] = True
        ) -> None:
        """
        Reads Sentinel-2 spectral bands from a dataset in .SAFE format by calling
        ``agrisatpy.io.Sat_Data_Reader`` as super-class.. The .SAFE format is ESA's
        standard format for distributing Sentinel-2 data.

        The function works on Sentinel-2 L1C and L2A processing level and reads by
        default all 10 and 20m bands. If the processing level is L2A it also searches
        for the scene classification layer (SCL) and provides it as additional band.

        NOTE:
            By default, all 10 and 20m bands are loaded (10 bands in total).

        :param in_dir:
            file-path to the .SAFE directory containing Sentinel-2 data in
            L1C or L2A processing level
        :param polygon_features:
            vector file (e.g., ESRI shapefile or geojson) or ``GeoDataFrame`` defining
            geometry/ies for which to extract the Sentinel-2 data. Can contain
            one to many features of type Polygon or MultiPolygon
        :param full_bounding_box_only:
            if set to False, will only extract the data for those geometry/ies
            defined in in_file_aoi. If set to False, returns the data for the
            full extent (hull) of all features (geometries) in in_file_aoi.
        :param int16_to_float:
            if True (Default) converts the original UINT16 Sentinel-2 data
            to numpy floats ranging between 0 and 1. Set to False if you
            want to keep the UINT16 datatype and original value range.
        :param band_selection:
            selection of Sentinel-2 bands to read. Per default, all 10 and
            20m bands are processed. If you wish to read less or more, specify
            the band names accordingly, e.g., ['B02','B03','B04'] to read only the
            VIS bands. If the processing level is L2A the SCL band is **always**
            loaded unless you set ``read_scl`` to False.
        :param read_scl:
            read SCL file if available (default, L2A processing level).
        :param skip_blackfilled_scenes:
            if True (default) the dataset is skipped if the extracted band data
            contains blackfill (nodata), only. Sentinel-2 blackfill is indicated
            by a reflectance of zero in every pixel.
        """

        # load 10 and 20 bands by default
        if band_selection is None:
            band_selection = list(s2_band_mapping.keys())
            bands_to_exclude = ['B01', 'B09', 'B10']
            for band in bands_to_exclude:
                band_selection.remove(band)

        # determine which spatial resolutions are selected and check processing level
        band_df_safe = self._get_band_files(
            in_dir=in_dir,
            band_selection=band_selection,
            read_scl=read_scl
        )

        # loop over bands and add them to the reader object
        self._from_bandstack = False

        for idx, band_name in enumerate(band_selection):

            # get entry from dataframe with file-path of band
            band_safe = band_df_safe[band_df_safe.band_name == band_name]
            band_fpath = band_safe.band_path.values[0]

            # read band
            try:
                band_reader = SatDataHandler()
                band_reader.read_from_bandstack(
                    fname_bandstack=band_fpath,
                    in_file_aoi=polygon_features,
                    full_bounding_box_only=full_bounding_box_only
                )

                # add band meta and bounds (required for adding band data)
                band_reader_band = band_reader.bandnames[0]
                meta = band_reader.get_meta(band_reader_band)
                attrs = band_reader.get_attrs(band_reader_band)
                bounds = band_reader.get_bounds(
                    band_name=band_reader_band,
                    return_as_polygon=False
                )

                # get color name for saving meta and bounds
                color_name = s2_band_mapping[band_name]

                # set first_band_to_add to true in the first iteration
                first_band_to_add = idx == 0

                # add band data
                self.add_band(
                    band_name=color_name,
                    band_alias=band_name,
                    band_data=band_reader.get_band(band_reader_band),
                    band_meta=meta,
                    band_bounds=bounds,
                    band_attribs=attrs,
                    first_band_to_add=first_band_to_add
                )

            except Exception as e:
                raise Exception(
                    f'Could not add band {band_name} from .SAFE to handler: {e}'
                )

            # check for blackfill; if the whole band contains blackfill (nodata)
            # only, raise a warning message (blackfill is indicated by a reflectance
            # value of zero in each pixel in the current band
            if skip_blackfilled_scenes:
                if self.is_blackfilled():
                    raise BlackFillOnlyError(
                        f'The read subset of {in_dir} contains blackfill, only'
                    )

        # convert datatype if required
        if int16_to_float:
            self._int16_to_float()

        # set scene properties (platform, sensor, acquisition date)
        acqui_time = get_S2_acquistion_time_from_safe(dot_safe_name=in_dir)
        platform = get_S2_platform_from_safe(dot_safe_name=in_dir)
        processing_level = get_S2_processing_level(dot_safe_name=in_dir)

        self.scene_properties.acquisition_time = acqui_time
        self.scene_properties.platform = platform
        self.scene_properties.sensor = 'MSI'
        self.scene_properties.processing_level =processing_level
        self.scene_properties.dataset_uri = in_dir.name


    @classmethod
    def read_pixels_from_safe(
            cls,
            point_features: Union[Path, gpd.GeoDataFrame],
            in_dir: Path,
            band_selection: Optional[List[str]] = list(s2_band_mapping.keys()),
            read_scl: Optional[bool] = True
        ) -> gpd.GeoDataFrame:
        """
        Extracts Sentinel-2 raster values at locations defined by one or many
        point-like geometry features read from a vector file (e.g., ESRI shapefile).
        The Sentinel-2 data must be organized in .SAFE archive structure in either
        L1C or L2A processing level. Each selected Sentinel-2 band is returned as
        a column in the resulting ``GeoDataFrame``. Pixels outside of the band
        bounds are ignored and not returned as well as pixels set to blackfill
        (zero reflectance in all spectral bands).

        IMPORTANT:
            This function works for Sentinel-2 data organized in .SAFE format!
            If the Sentinel-2 data has been converted to multi-band tiffs, use
            `~Sentinel2Handler.read_pixels()` instead!

        NOTE:
            A point is dimension-less, therefore, the raster grid cell (pixel) closest
            to the point is returned if the point lies within the raster.
            Therefore, this method works on all Sentinel-2 bands **without** the need
            to do spatial resampling! The underlying ``rasterio.sample`` function always
            snaps to the closest pixel in the current spectral band.

        :param point_features:
            vector file (e.g., ESRI shapefile or geojson) or ``GeoDataFrame``
            defining point locations for which to extract pixel values
        :param raster:
            custom raster dataset understood by ``GDAL`` from which to extract
            pixel values at the provided point locations
        :param band_selection:
            list of bands to read. Per default all raster bands available are read.
        :param read_scl:
            read SCL file if available (default, L2A processing level).
        :return:
            ``GeoDataFrame`` containing the extracted raster values. The band values
            are appened as columns to the dataframe. Existing columns of the input
            `in_file_pixels` are preserved.
        """

        # check band selection and get file-paths to the single jp2 files
        band_df_safe = cls._get_band_files(
            cls,
            in_dir=in_dir,
            band_selection=band_selection,
            read_scl=read_scl
        )

        # loop over spectral bands and extract the pixel values
        band_gdfs = []
        for idx, band_name in enumerate(band_selection):

            # get entry from dataframe with file-path of band
            band_safe = band_df_safe[band_df_safe.band_name == band_name]
            band_fpath = Path(band_safe.band_path.values[0])

            # read band pixels
            try:
                gdf_band = cls.read_pixels(
                    point_features=point_features,
                    raster=band_fpath,
                )

                # rename the spectral band (rasterio returns None as band name from
                # the jp2 files that is translated to NaN in geopandas)
                gdf_band = gdf_band.rename(columns={None: band_name})

                # remove the geometry column from all GeoDataFrames but the first
                # since geopandas does not support multiple geometry columns
                # (they are the same for each band, anyways)
                if idx > 0:
                    gdf_band.drop('geometry', axis=1, inplace=True)
                band_gdfs.append(gdf_band)
            except Exception as e:
                raise Exception(
                    f'Could not extract pixels values from {band_name}: {e}'
                )

        # concat the single GeoDataFrames with the band data
        gdf = pd.concat(band_gdfs, axis=1)

        # clean the dataframe and remove duplicate column names after merging
        # to avoid (large) redundancies
        gdf = gdf.loc[:,~gdf.columns.duplicated()]

        # skip all pixels with zero reflectance (either blackfilled or outside of the
        # scene extent)
        gdf = gdf.loc[~(gdf[band_selection]==0).all(axis=1)]

        return gdf


if __name__ == '__main__':

    import cv2

    safe_archive = Path('../../../data/S2A_MSIL2A_20190524T101031_N0212_R022_T32UPU_20190524T130304.SAFE')
    field_parcels = Path('../../../data/sample_polygons/BY_Polygons_Canola_2019_EPSG32632.shp')

    # safe_archive = Path('/mnt/ides/Lukas/03_Debug/Sentinel2/S2A_MSIL2A_20171213T102431_N0206_R065_T32TMT_20171213T140708.SAFE')
    handler = Sentinel2Handler()
    handler.read_from_safe(
        in_dir=safe_archive,
        polygon_features=field_parcels,
        full_bounding_box_only=False
    )

    scl_stats_df = handler.get_scl_stats()
    band_stats = handler.get_band_summaries()

#
#     handler.add_bands_from_vector(
#         in_file_vector=field_parcels,         
#         snap_band='blue',                   # we use one of the 10m bands
#         attribute_selection=['crop_code'],  # we can use any selection of numeric attributes; each attribute becomes a new raster band
#         blackfill_value=0                   # here it is important to choose a value that not occurs in the data to rasterize
#     )
#
    # resample to 30m
    handler.resample(
        target_resolution=30,   # meter
        resampling_method=cv2.INTER_CUBIC
    )
    
#
#     bands_to_mask = handler.bandnames
#
#     handler.mask(
#         name_mask_band='crop_code',
#         mask_values=[1],  # 1 is the code for canola
#         bands_to_mask=bands_to_mask,
#         keep_mask_values=True  # we want to keep all canola pixels
#     )
#
#     gdf_canola_pixels = handler.to_dataframe()
    