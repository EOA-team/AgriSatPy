'''
This module contains the ``Sentinel2`` class that inherits from
AgriSatPy's core ``RasterCollection`` class.

The ``Sentinel2`` class enables reading one or more spectral bands from Sentinel-2
data in .SAFE format which is ESA's standard format for distributing Sentinel-2 data.

The class handles data in L1C and L2A processing level.
'''

import numpy as np
import pandas as pd
import geopandas as gpd

from matplotlib.pyplot import Figure
from matplotlib import colors
from pathlib import Path
from typing import (
    Optional,
    List,
    Tuple,
    Union
)

from agrisatpy.core.band import (
    Band,
    WavelengthInfo
)
from agrisatpy.core.raster import RasterCollection
from agrisatpy.utils.constants.sentinel2 import (
    band_resolution,
    band_widths,
    central_wavelengths,
    ProcessingLevels,
    s2_band_mapping,
    s2_gain_factor,
    SCL_Classes
)
from agrisatpy.utils.exceptions import BandNotFoundError
from agrisatpy.utils.sentinel2 import get_S2_bandfiles_with_res
from agrisatpy.utils.sentinel2 import get_S2_platform_from_safe
from agrisatpy.utils.sentinel2 import get_S2_processing_level
from agrisatpy.utils.sentinel2 import get_S2_acquistion_time_from_safe
from agrisatpy.core.scene import SceneProperties


class Sentinel2(RasterCollection):
    """
    Class for storing Sentinel-2 band data read from bandstacks or
    .SAFE archives (L1C and L2A level) overwriting methods inherited
    from `~agrisatpy.utils.io.SatDataHandler`.
    """

    @property
    def is_blackfilled(self) -> bool:
        """Checks if the scene is black-filled (nodata only)"""
        # if SCL is available use this layer
        if 'SCL' in self.band_names:
            scl_stats = self.get_scl_stats()
            no_data_count = scl_stats[
                scl_stats.Class_Value.isin([0])
            ]['Class_Abs_Count'].sum()
            all_pixels = scl_stats['Class_Abs_Count'].sum()
            return no_data_count == all_pixels
        # otherwise check the reflectance values from the first
        # band in the collection. If all values are zero then
        # the pixels are considered backfilled
        else:
            band_name = self.band_names[0]
            if self[band_name].is_masked_array:
                return (self[band_name].values.date == 0).all()
            elif self[band_name].is_ndarray:
                return (self[band_name].values == 0).all()
            elif self[band_name].is_zarr:
                raise NotImplementedError()

    @staticmethod
    def _get_band_files(
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
        :returns:
            ``DataFrame`` with paths to the jp2 files with the spectral band data
        """
        # check processing level
        processing_level = get_S2_processing_level(dot_safe_name=in_dir)

        # define also a local flag of the processing level so this method also works
        # when called from a classmethod
        is_l2a = True
        if processing_level == ProcessingLevels.L1C:
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

    @classmethod
    def from_safe(
            cls,
            in_dir: Path,
            band_selection: Optional[List[str]] = None,
            read_scl: Optional[bool] = True,
            **kwargs
        ):
        """
        Loads Sentinel-2 data from a .SAFE archive which is ESA's
        standard format for distributing Sentinel-2 data (L1C and L2A
        processing levels).

        :param in_dir:
            file-path to the .SAFE directory containing Sentinel-2 data in
            L1C or L2A processing level
        :param band_selection:
            selection of Sentinel-2 bands to read. Per default, all 10 and
            20m bands are processed. If you wish to read less or more, specify
            the band names accordingly, e.g., ['B02','B03','B04'] to read only the
            VIS bands. If the processing level is L2A the SCL band is **always**
            loaded unless you set ``read_scl`` to False.
        :param read_scl:
            read SCL file if available (default, L2A processing level).
        :param kwargs:
            optional key-word arguments to pass to `~agrisatpy.core.band.Band.from_rasterio`
        """
        # load 10 and 20 bands by default
        if band_selection is None:
            band_selection = list(s2_band_mapping.keys())
            bands_to_exclude = ['B01', 'B09']
            for band in bands_to_exclude:
                band_selection.remove(band)

        # determine which spatial resolutions are selected and check processing level
        band_df_safe = cls._get_band_files(
            in_dir=in_dir,
            band_selection=band_selection,
            read_scl=read_scl
        )
        # determine platform (S2A or S2B)
        platform = get_S2_platform_from_safe(dot_safe_name=in_dir)
        # set scene properties (platform, sensor, acquisition date)
        acqui_time = get_S2_acquistion_time_from_safe(dot_safe_name=in_dir)
        processing_level = get_S2_processing_level(dot_safe_name=in_dir)

        scene_properties = SceneProperties(
            acquisition_time = acqui_time,
            platform = platform,
            sensor = 'MSI',
            processing_level = processing_level,
            product_uri = in_dir.name
        )

        # set AREA_OR_POINT to Point (see here: https://gis.stackexchange.com/a/263329)
        # TODO: make sure this is really true
        kwargs.update({'area_or_point': 'Area'})
        # set nodata to zero (unfortunately the S2 img metadata is incorrect here)
        kwargs.update({'nodata': 0})
        # set correct scale factor (unfortunately not correct in S2 img metadata)
        kwargs.update({'scale': s2_gain_factor})

        # loop over bands and add them to the collection of bands
        sentinel2 = cls(scene_properties=scene_properties)
        for band_name in band_selection:

            # get entry from dataframe with file-path of band
            band_safe = band_df_safe[band_df_safe.band_name == band_name]
            band_fpath = band_safe.band_path.values[0]

            # get color name and set it as alias
            color_name = s2_band_mapping[band_name]

            # store wavelength information per spectral band
            if band_name != 'SCL':
                central_wvl = central_wavelengths[platform][band_name]
                wavelength_unit = central_wavelengths['unit']
                band_width = band_widths[platform][band_name]
                wvl_info = WavelengthInfo(
                    central_wavelength=central_wvl,
                    wavelength_unit=wavelength_unit,
                    band_width=band_width
                )
                kwargs.update({'wavelength_info': wvl_info})
            # read band
            try:
                sentinel2.add_band(
                    Band.from_rasterio,
                    fpath_raster=band_fpath,
                    band_idx=1,
                    band_name_dst=band_name,
                    band_alias=color_name,
                    **kwargs
                )
            except Exception as e:
                raise Exception(
                    f'Could not add band {band_name} from {in_dir.name}: {e}'
                )

        return sentinel2

    # TODO: adopt this function to latest API version
    @classmethod
    def read_pixels_from_safe(
            cls,
            point_features: Union[Path, gpd.GeoDataFrame],
            in_dir: Path,
            band_selection: Optional[List[str]] = None,
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

        if band_selection is None:
            band_selection = list(s2_band_mapping.keys())

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
        # check if SCL is a masked array. If so, fill masked values with no-data
        # class (for plotting we need to manipulate the data directly),
        #  therefore we work on a copy of SCL
        scl = self['SCL'].copy()
        if scl.is_masked_array:
            scl.values = scl.values.filled(
                [k for k, v in SCL_Classes.values().items() if v == 'no_data']
            )

        # make a color map of fixed colors
        if colormap == '':
            # get only those colors required (classes in the layer)
            scl_colors = SCL_Classes.colors()
            scl_dict = SCL_Classes.values()
            scl_classes = list(np.unique(self.get_band('SCL')))
            selected_colors = [x for idx,x in enumerate(scl_colors) if idx in scl_classes]
            scl_cmap = colors.ListedColormap(selected_colors)
            scl_ticks = [x[1] for x in scl_dict.items() if x[0] in scl_classes]
        try:
            return scl.plot(
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
        A Wrapper around the inherited ``mask`` method to mask clouds,
        shadows, water and snow based on the SCL band. Works therefore on L2A data,
        only.

        Masks the SCL classes 2, 3, 7, 8, 9, 10, 11.

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
        mask_band = 'SCL'
        try:
            self.mask(
                name_mask_band=mask_band,
                mask_values=cloud_classes,
                bands_to_mask=bands_to_mask,
                inplace=True
            )
        except Exception as e:
            raise Exception(f'Could not apply cloud mask: {e}')


    def get_scl_stats(self) -> pd.DataFrame:
        """
        Returns a ``DataFrame`` with the number of pixel for each
        class of the scene classification layer. Works for data in
        L2A processing level, only.

        :returns:
            ``DataFrame`` with pixel count of SCL classes available
            in the currently loaded Sentinel-2 scene. Masked pixels
            are ignored and also not used for calculating the relative
            class occurences.
        """
        # check if SCL is available
        if not 'SCL' in self.band_names:
            raise BandNotFoundError(
                'Could not find scene classification layer. Is scene L2A?'
            )

        scl = self.get_band('SCL')
        # if the scl array is a masked array consider only those pixels
        # not masked out
        if scl.is_masked_array:
            scl_values = scl.values.compressed()
        else:
            scl_values = scl.values

        # overall number of pixels; masked pixels are not considered
        n_pixels = scl_values.size

        # count occurence of SCL classes
        scl_classes, class_counts = np.unique(scl_values, return_counts=True)
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
            cloud_classes: Optional[List[int]] = [2, 3, 7, 8, 9, 10, 11],
        ) -> float:
        """
        Calculates the cloudy pixel percentage [0-100] for the current AOI
        (L2A processing level, only) considering all SCL classes that are
        not NoData.

        :param cloud_classes:
            list of SCL values to be considered as clouds. By default,
            all three cloud classes and cloud shadows are considered.
        :param check_for_snow:
            if True (default) also counts snowy pixels as clouds.
        :returns:
            cloudy pixel percentage in the AOI [0-100%] related to the
            overall number of valid pixels (SCL != no_data)
        """

        # get SCL statistics
        scl_stats_df = self.get_scl_stats()

        # sum up pixels labeled as clouds or cloud shadows
        num_cloudy_pixels = scl_stats_df[
            scl_stats_df.Class_Value.isin(cloud_classes)
        ]['Class_Abs_Count'].sum()
        # check for nodata (e.g., due to blackfill)
        nodata_pixels = scl_stats_df[
            scl_stats_df.Class_Value.isin([0])
        ]['Class_Abs_Count'].sum()
        
        # and relate it to the overall number of pixels
        all_pixels = scl_stats_df['Class_Abs_Count'].sum()
        cloudy_pixel_percentage = num_cloudy_pixels / \
            (all_pixels - nodata_pixels) * 100
        return cloudy_pixel_percentage
