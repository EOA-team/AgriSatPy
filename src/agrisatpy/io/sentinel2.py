'''
This module contains the ``S2_Band_Reader`` class that inherits from
AgriSatPy's ``Sat_Data_Reader`` class.

The ``S2_Band_Reader`` enables reading one or more spectral bands from Sentinel-2
data. The data can be either band-stacked (i.e., AgriSatPy derived format) in .SAFE
format which is ESA's standard format for distributing Sentinel-2 data.

The class handles data in L1C and L2A processing level.
'''

import numpy as np

from matplotlib.pyplot import Figure
from matplotlib import colors
from pathlib import Path
from typing import Optional
from typing import List

from agrisatpy.utils.constants.sentinel2 import ProcessingLevels
from agrisatpy.utils.constants.sentinel2 import s2_band_mapping
from agrisatpy.utils.constants.sentinel2 import s2_gain_factor
from agrisatpy.utils.constants.sentinel2 import SCL_Classes
from agrisatpy.utils.sentinel2 import get_S2_bandfiles_with_res
from agrisatpy.utils.sentinel2 import get_S2_sclfile
from agrisatpy.utils.sentinel2 import get_S2_processing_level
from agrisatpy.utils.constants.sentinel2 import band_resolution
from agrisatpy.io import Sat_Data_Reader
from agrisatpy.utils.exceptions import BandNotFoundError


class S2_Band_Reader(Sat_Data_Reader):
    """
    Class for storing Sentinel-2 band data read from bandstacks or
    .SAFE archives (L1C and L2A level) overwriting methods inherited
    from `~agrisatpy.utils.io.Sat_Data_Reader`.
    """

    def __init__(self, *args, **kwargs):
        Sat_Data_Reader.__init__(self, *args, **kwargs)

    @staticmethod
    def _check_band_selection(
            band_selection: List[str]
        ) -> List[str]:
        """
        Returns the Sentinel-2 color names for a given band selection
    
        :param band_selection:
            list of selected bands
        :return:
            list of corresponding Sentinel-2 color names
        """

        # check how many bands were selected
        return [s2_band_mapping[k] for k in band_selection]


    def _generate_band_aliases(
            self,
            sel_bands: List[str]
        ) -> None:
        """
        Helper method to generate Sentinel-2 band aliases

        :param sel_bands:
            list of selected Sentinel-2 bands
        """
        old_band_names = self.get_bandnames()
        self.reset_bandnames(new_bandnames=sel_bands)
        self.set_bandaliases(
            band_names=self.get_bandnames(),
            band_aliases=old_band_names
        )


    def _int16_to_float(self) -> None:
        """
        Converts Sentinel-2 16 bit integer arrays to float and applies a gain
        factor to scale value between 0 and 1
        """

        for band_name in self.get_bandnames():
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


    def plot_scl(
            self,
            colormap: Optional[str]=''
        ) -> Figure:
        """
        Wrapper around `agrisatpy.utils.io` to plot the Scene Classification
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
        # if so, fill masked values with no-data class
        if isinstance(self.data['scl'], np.ma.core.MaskedArray):
            self.data['scl'] = self.data['scl'].filled(
                [k for k, v in SCL_Classes.values().items() if v == 'no_data'])

        # make a color map of fixed colors
        if colormap == '':
            
            # get only those colors required (classes in the layer)
            scl_colors = SCL_Classes.colors()
            scl_dict = SCL_Classes.values()
            scl_classes = list(np.unique(self.data['scl']))
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
            cloud_classes: Optional[List[int]]=[3, 8, 9, 10]
        ) -> None:
        """
        A Wrapper around the inherited ``mask`` method to mask clouds and
        shadows based on the SCL band. Works therefore on L2A data only where
        SCL data is available. Masks the SCL classes 3, 8, 9, 10.
        If another class selection is desired consider using the mask function
        from `agrisatpy.utils.io` directly.

        :param bands_to_mask:
            list of bands on which to apply the SCL mask
        :param cloud_classes:
            list of SCL values to be considered as clouds. By default,
            all three cloud classes and cloud shadows are considered.
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


    def get_cloudy_pixel_percentage(
            self,
            cloud_classes: Optional[List[int]]=[3, 8, 9, 10]
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

        # check if SCL is available
        if not 'scl' in self.data.keys():
            raise BandNotFoundError(
                'Could not find scene classification layer. Is scene L2A?'
            )

        # check if SCL is a masked array
        # if so, fill masked values with no-data class
        if isinstance(self.data['scl'], np.ma.core.MaskedArray):
            self.data['scl'] = self.data['scl'].filled(
                [k for k, v in SCL_Classes.values().items() if v == 'no_data'])

        # sum up pixels labeled as clouds or cloud shadows
        unique, counts = np.unique(self.data['scl'], return_counts=True)
        class_occurence = dict(zip(unique, counts))
        cloudy_pixels = [x[1] for x in class_occurence.items() if x[0] in cloud_classes]
        scl_nodata = 0
        non_cloudy_pixels = [
            x[1] for x in class_occurence.items() if x[0] not in cloud_classes \
            and x[0] != scl_nodata
        ]
        num_cloudy_pixels = sum(cloudy_pixels)
        num_noncloudy_pixels = sum(non_cloudy_pixels)

        # relate it to the overall number of valid pixels in the AOI
        cloudy_pixel_percentage = (num_cloudy_pixels / num_noncloudy_pixels) * 100

        return cloudy_pixel_percentage


    def read_from_bandstack(
            self,
            fname_bandstack: Path,
            processing_level: Optional[ProcessingLevels] = None,
            in_file_scl: Optional[Path] = None,
            in_file_aoi: Optional[Path] = None,
            full_bounding_box_only: Optional[bool] = False,
            int16_to_float: Optional[bool] = True,
            band_selection: Optional[List[str]] = list(s2_band_mapping.keys())
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
            database query) pass it directly to the method, otherwise the method
            tries to find the corresponding SCL file from the filesystem assuming
            that AgriSatPy's default file system logic was used.
        :param in_file_aoi:
            vector file (e.g., ESRI shapefile or geojson) defining geometry/ies
            (polygon(s)) for which to extract the Sentinel-2 data. Can contain
            one to many features.
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
        """

        # call method from super class
        try:
            super().read_from_bandstack(
                fname_bandstack=fname_bandstack,
                in_file_aoi=in_file_aoi,
                full_bounding_box_only=full_bounding_box_only,
                blackfill_value=0,
                band_selection=band_selection
            )
        except Exception as e:
            raise Exception from e

        # post-treatment: loop over bands, set band and color names and 
        # convert and rescale to float if selected
        sel_bands = self._check_band_selection(band_selection)
        sel_bands.remove('scl')
        # preserve "old" band names as aliases
        self._generate_band_aliases(sel_bands=sel_bands)

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

        # read SCL if processing level is L2A
        if processing_level == ProcessingLevels.L2A:

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
            scl_reader = Sat_Data_Reader()
            try:
                scl_reader.read_from_bandstack(
                    fname_bandstack=in_file_scl,
                    in_file_aoi=in_file_aoi,
                    full_bounding_box_only=full_bounding_box_only,
                    blackfill_value=0
                )
            except Exception as e:
                raise Exception(f'Could not read SCL file: {e}')

            # add SCL as new band
            self.add_band(
                band_name='scl',
                band_data=scl_reader.get_band('B1'),
                band_alias='SCL'
            )

            # increase band count in meta
            self.data['meta']['count'] += 1


    def read_from_safe(
            self,
            in_dir: Path,
            processing_level: Optional[ProcessingLevels] = None,
            in_file_aoi: Optional[Path] = None,
            full_bounding_box_only: Optional[bool] = False,
            int16_to_float: Optional[bool] = True,
            band_selection: Optional[List[str]] = list(s2_band_mapping.keys()),
            read_scl: Optional[bool] = True
        ) -> None:
        """
        Reads Sentinel-2 spectral bands from a dataset in .SAFE format by calling
        ``agrisatpy.io.Sat_Data_Reader`` as super-class.. The .SAFE format is ESA's
        standard format for distributing Sentinel-2 data.

        The function works on Sentinel-2 L1C and L2A processing level and reads by
        default all 10 and 20m bands. If the processing level is L2A it also searches
        for the scene classification layer (SCL) and provides it as additional band.

        :param in_dir:
            file-path to the .SAFE directory containing Sentinel-2 data in
            L1C or L2A processing level
        :param processing_level:
            specification of the processing level of the Sentinel-2 data. If not
            provided will be determined from the name of the .SAFE dataset.
        :param spatial resolution:
            target spatial resolution. Sentinel-2 bands not having the
            target spatial resolution are resampled on the fly...
        :param in_file_aoi:
            vector file (e.g., ESRI shapefile or geojson) defining geometry/ies
            (polygon(s)) for which to extract the Sentinel-2 data. Can contain
            one to many features.
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

        # determine which spatial resolutions are selected
        # (based on the native spatial resolution of Sentinel-2 bands)
        band_selection_spatial_res = [x[1] for x in band_resolution.items() if x[0] in band_selection]
        resolution_selection = list(np.unique(band_selection_spatial_res))
    
        # check processing level
        if processing_level is None:
            processing_level = get_S2_processing_level(dot_safe_name=in_dir)
        is_L2A = True
        if processing_level == ProcessingLevels.L1C:
            is_L2A = False
    
        # search band files depending on processing level and spatial resolution(s)
        band_df_safe = get_S2_bandfiles_with_res(
            in_dir=in_dir,
            resolution_selection=resolution_selection,
            is_L2A=is_L2A
        )

        # search SCL file if processing level is L2A
        if is_L2A and read_scl:
            band_selection.append('SCL')
            try:
                scl_file = get_S2_sclfile(in_dir)
                # append to dataframe
                record = {
                    'band_name': 'SCL',
                    'band_path': str(scl_file),
                    'band_resolution': 20
                }
                band_df_safe = band_df_safe.append(record, ignore_index=True)
            except Exception as e:
                raise Exception(f'Could not read SCL file: {e}')

        # loop over bands and add them to the reader object
        self._from_bandstack = False
        for band_name in band_selection:

            # get entry from dataframe with file-path of band
            band_safe = band_df_safe[band_df_safe.band_name == band_name]
            band_fpath = band_safe.band_path.values[0]

            # read band
            try:
                band_reader = Sat_Data_Reader()
                band_reader.read_from_bandstack(
                    fname_bandstack=band_fpath,
                    in_file_aoi=in_file_aoi,
                    full_bounding_box_only=full_bounding_box_only
                )

                # add band meta and bounds BEFORE assigning band data
                band_reader_band = band_reader.get_bandnames()[0]
                meta = band_reader.get_meta(band_reader_band)
                self.set_meta(
                    meta=meta,
                    band_name=band_name
                )
                bounds = band_reader.get_band_bounds(
                    band_name=band_reader_band,
                    return_as_polygon=False
                )
                self.set_bounds(
                    bounds=bounds,
                    band_name=band_name
                )

                # add band data
                self.add_band(
                    band_name=band_name,
                    band_data=band_reader.get_band(band_reader_band),
                )

            except Exception as e:
                raise Exception from e

        # convert datatype if required
        if int16_to_float:
            self._int16_to_float()

        # set band aliases
        sel_bands = self._check_band_selection(band_selection=band_selection)
        self._generate_band_aliases(sel_bands=sel_bands)


if __name__ == '__main__':

    # download test data (if not done yet)
    import cv2
    import requests
    from agrisatpy.downloader.sentinel2.utils import unzip_datasets

    reader = S2_Band_Reader()
    band_selection = ['B02', 'B03', 'B04', 'B08']
    in_file_aoi = Path('/home/graflu/git/agrisatpy/data/sample_polygons/ZH_Polygons_2020_ESCH_EPSG32632.shp')

    # bandstack testcase
    fname_bandstack = Path('/home/graflu/git/agrisatpy/data/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    processing_level = ProcessingLevels.L2A

    reader.read_from_bandstack(
        fname_bandstack=fname_bandstack,
        in_file_aoi=in_file_aoi
    )
    fig_rgb = reader.plot_rgb()
    fig_scl = reader.plot_scl()
    cc = reader.get_cloudy_pixel_percentage()
    fig_blue = reader.plot_band('blue')
    
    band_names = reader.get_bandnames()

    # L2A testcase
    url = 'https://data.mendeley.com/public-files/datasets/ckcxh6jskz/files/e97b9543-b8d8-436e-b967-7e64fe7be62c/file_downloaded'
    
    testdata_dir = Path('/mnt/ides/Lukas/debug/S2_Data')
    in_file_aoi = Path('/home/graflu/git/agrisatpy/data/sample_polygons/BY_AOI_2019_MNI_EPSG32632.shp')

    testdata_fname = testdata_dir.joinpath('S2A_MSIL2A_20190524T101031_N0212_R022_T32UPU_20190524T130304.zip')
    testdata_fname_unzipped = Path(testdata_fname.as_posix().replace('.zip', '.SAFE'))

    if not testdata_fname_unzipped.exists():
        
        # download dataset
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(testdata_fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=5096):
                fd.write(chunk)

        # unzip dataset
        unzip_datasets(download_dir=testdata_dir)

    reader = S2_Band_Reader()
    reader.read_from_safe(
        in_dir=testdata_fname_unzipped,
        band_selection=['B02']
    )
    blue = reader.get_band('B02')

    # check the RGB
    fig_rgb = reader.plot_rgb()

    # and the false-color near-infrared
    fig_nir = reader.plot_false_color_infrared()

    # check the scene classification layer
    fig_scl = reader.plot_scl()

    reader.resample(target_resolution=10, resampling_method=cv2.INTER_CUBIC, bands_to_exclude=['scl'])
    reader.calc_vi(vi='NDVI')
    reader.write_bands(
        out_file=testdata_dir.joinpath('scl.tif'),
        band_selection=['scl']
    )

    cloudy_pixels = reader.get_cloudy_pixel_percentage()

    # resample SCL
    reader.resample(target_resolution=10, pixel_division=True)

    # mask the clouds (SCL classes 8,9,10) and cloud shadows (class 3)
    reader.mask_clouds_and_shadows(bands_to_mask=['TCARI_OSAVI'])
    reader.plot_band('TCARI_OSAVI', colormap='summer')
