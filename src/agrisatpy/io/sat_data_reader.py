'''
The Sat_Data_Reader is a super class from which sensor-specific classes for reading
(satellite) raster image data might inherit. The Sat_Data_Reader class provides
methods to read arbitrary raster data from files (or URIs) using rasterio's GDAL drivers,
and stores the band data in a dict-like data structure preserving the geo-spatial metadata.
'''

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import rasterio as rio
import rasterio.mask

from rasterio import Affine
from rasterio.coords import BoundingBox
from rasterio.drivers import driver_from_extension
from rasterio.crs import CRS
from shapely.geometry import box
from shapely.geometry import Polygon
from pathlib import Path
from typing import Optional
from typing import Dict
from typing import List
from typing import Union
from typing import Tuple
from typing import NamedTuple
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import Figure
from matplotlib.figure import figaspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from collections import namedtuple

from agrisatpy.analysis.vegetation_indices import VegetationIndices
from agrisatpy.utils.exceptions import NotProjectedError
from agrisatpy.utils.exceptions import ResamplingFailedError
from agrisatpy.utils.exceptions import BandNotFoundError
from agrisatpy.utils.exceptions import BlackFillOnlyError
from agrisatpy.utils.reprojection import check_aoi_geoms
from agrisatpy.spatial_resampling import upsample_array


class Sat_Data_Reader(object):
    """reader class from which sensor-specific classes inherit"""

    def __init__(self):
        self.data = {}
        self._from_bandstack = False


    def from_bandstack(self) -> bool:
        """
        checks if the data was read from bandstack derived from
        `~agrisatpy.processing.resampling` or custom (i.e., sensor-
        specific) file system structure such as .SAFE in case of
        Sentinel-2
        """

        return self._from_bandstack


    def add_band(self, band_name: str, band_data: np.array):
        """
        Adds a band to an existing band dict instance

        :param band_name:
            name of the band to add (existing bands with the same name
            will be overwritten!); corresponds to the dict key
        :param band_data:
            band data to add; corresponds to the dict value
        """

        band_dict = {band_name: band_data}
        self.data.update(band_dict)

        # check if the data comes from a bandstack (nothing to do)
        # or from single band files where the metadata needs to be added
        # for each band
        if not self.from_bandstack():
            # find out, which band has the same shape (x, y dim) and copy
            # the 'meta' and 'bounds' from that band for the band to add
            band_data_shape = band_data.shape
            for other_band in self.data.keys():
                if other_band == 'meta' or other_band == 'bounds':
                    continue
                if self.data[other_band].shape == band_data_shape:
                    self.data['meta'][band_name] = self.data['meta'][other_band]
                    self.data['bounds'][band_name] = self.data['bounds'][other_band]
                    # leave loop and return
                    break


    def get_band(
            self,
            band_name
        ) -> np.array:
        """
        Returns the ``numpy.array`` containing the data for one band

        :param band_name:
            name of the band to extract
        :return:
            band data as ``numpy.array`` (two dimensional)
        """

        # check if band_name is available
        if not band_name in self.data.keys():
            raise BandNotFoundError(f'{band_name} not found in data dict')

        # return np array
        return self.data[band_name]


    def get_bandnames(self) -> List[str]:
        """
        Returns a list of all available band names. It is assumed that
        a numpy array attribute indicates a band

        :return:
            list of available band names
        """

        band_names = []
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                band_names.append(key)

        return band_names


    def get_meta(
            self,
            band_name: Optional[str] = None
        ) -> dict:
        """
        Returns the image metadata for all bands or a selected band

        :param band_name:
            optional band name for retrieving meta data for a specific
            band
        :return:
            meta dict with image meta data
        """

        # if data is band-stacked, meta is always the same
        # otherwise it must be avialable for each band as entry
        if not self.from_bandstack() and band_name is not None:
            try:
                return self.data['meta'][band_name]
            except Exception:
                raise BandNotFoundError(
                    f'Could not find "{band_name}" in data dict'
                )
        else:
            return self.data['meta']


    def get_spatial_resolution(
            self,
            band_name: str
        ) -> NamedTuple:
        """
        Returns the spatial resolution in x and y direction of a band

        :param band_name:
            band name for retrieving spatial resolution
            of a specific band
        :return:
            spatial resolution in units of the image coordinate system
            in x and y direction
        """

        transform = self.get_meta(band_name=band_name)['transform']
        Spatial_Resolution = namedtuple('Spatial_Resolution', 'x y')
        return Spatial_Resolution(transform[0], transform[4])


    def get_band_bounds(
            self,
            band_name: str,
            return_as_polygon: Optional[bool] = True
        ) -> Tuple[Polygon, BoundingBox]:
        """
        Returns the bounds (bounding box) of a band

        :param band_name:
            band name for retrieving bounds
        :param return_as_polygon:
            if True returns a ``shapely`` polygon, if False a ``rasterio``
            bounding box object
        :return:
            bounds of the band in the coordinate system of the dataset
        """

        bounds = self.get_band(band_name='bounds')

        # check if the file is from band stack or if bounds are the same for each band
        if not self.from_bandstack():
            try:
                bounds = bounds[band_name]
            except Exception:
                raise BandNotFoundError(
                    f'Could not find "{band_name}" in data dict'
                )

        # return bounding box or polygon
        if return_as_polygon:
            bounds = box(*bounds)

        return bounds


    def get_band_epsg(
            self,
            band_name: str
        ) -> CRS:
        """
        Returns the EPSG code of a band

        :param band_name:
            name of the band for which to extract the EPSG code
        :return:
            EPSG code of the band as ``rasterio.crs.CRS``
        """

        meta = self.get_meta(band_name=band_name)
        return meta['crs']


    def is_blackfilled(
            self,
            blackfill_value: Optional[Union[int,float]] = 0
        ) -> bool:
        """
        Checks if the read Sentinel-2 scene data contains black fill only.
        Black fill in the spectral bands corresponds to having zero everywhere.

        :param blackfill_value:
            value indicating black fill. Set to zero by default.
        :return:
            True of the data is black fill only, else False
        """

        # check the bands
        band_names = self.get_bandnames()
        blackfill_list = []
        for band_name in band_names:
            band_data = self.get_band(band_name)
            if isinstance(band_data, np.ma.core.MaskedArray):
                band_data = deepcopy(band_data.data)
            if (band_data == blackfill_value).all():
                blackfill_list.append(True)
            else:
                blackfill_list.append(False)

        return all(blackfill_list)


    @staticmethod
    def _masked_array_to_nan(band_data: np.array) -> np.array:
        """
        If the band array is a masked array, the masked values
        are replaced by NaNs in order to clip the plot to the
        correct value range (otherwise also masked values are
        considered for getting the upper and lower bound of the
        colormap)

        :param band_data:
            either a numpy masked array or ndarray
        :return:
            either the same array (if input was ndarray) or an
            array where masked values were replaced with NaNs
        """

        if isinstance(band_data, np.ma.core.MaskedArray):
            # check if datatype of the array supports NaN (int does not)
            if band_data.dtype != np.uint8 and band_data.dtype != np.uint16:
                return band_data.filled(np.nan)
        # otherwise return the input
        return band_data


    def plot_rgb(self) -> Figure:
        """
        Plots a RGB image of the loaded band data providing a simple
        wrapper around the `~plot_band` method. Requires the
        'red', 'green' and 'blue' bands.

        :return:
            matplotlib figure object with the band data
            plotted as map
        """
        return self.plot_band(band_name='RGB')


    def plot_false_color_infrared(self) -> Figure:
        """
        Plots a false color infrared image of the loaded band data providing
        a simple wrapper around the `~plot_band` method. Requires the
        'nir_1', 'green' and 'red' bands.

        :return:
            matplotlib figure object with the band data
            plotted as map
        """
        return self.plot_band(band_name='False-Color')


    def plot_band(
            self,
            band_name: str,
            colormap: Optional[str] = 'gray',
            discrete_values: Optional[bool] = False,
            user_defined_colors: Optional[ListedColormap] = None,
            user_defined_ticks: Optional[List[Union[str,int,float]]] = None
        ) -> Figure:
        """
        plots a custom band using matplotlib.pyplot.imshow and the
        extent in the projection of the image data. Returns
        a figure object with the plot.

        To get a RGB preview of your data, pass band_name='RGB'.
        To get a false color NIR preview, pass band_name='FALSE-COLOR'
        If the band takes discrete values, only (e.g., classification)
        set `discrete_values` to False

        :param band_name:
            name of the band to plot
        :param colormap:
            String identifying one of matplotlib's colormaps.
            The default will plot the band in gray values.
        :param discrete_values:
            if True (Default) assumes that the band has continuous values
            (i.e., ordinary spectral data). If False assumes that the
            data only takes a limited set of discrete values (e.g., in case
            of a classification or mask layer).
        :param user_defined_colors:
            possibility to pass a custom, i.e., user-created color map object
            not part of the standard matplotlib color maps. If passed, the
            ``colormap`` argument is ignored.
        :param user_defined_ticks:
            list of ticks to overwrite matplotlib derived defaults (optional).
        :return:
            matplotlib figure object with the band data
            plotted as map
        """

        # custom band or RGB or False-Color NIR plot?
        rgb_plot, nir_plot = False, False
        if band_name.upper() == 'RGB':
            band_names = ['red', 'green', 'blue']
            rgb_plot = True
        elif band_name.upper() == 'FALSE-COLOR':
            nir_plot = True
            band_names = ['nir_1', 'green', 'red']
        else:  
            if band_name not in list(self.data.keys()):
                raise BandNotFoundError(f'{band_name} not found in data')
            band_data = self.data[band_name]

        # read band data in case of RGB or false color NIR plot
        if band_name.upper() == 'RGB' or band_name.upper() == 'FALSE-COLOR':
            diff = list(set(band_names) - set(list(self.data.keys())))
            # check if all required bands are available
            if len(diff) > 0:
                raise BandNotFoundError(f'band "{diff[0]}" not found in band data')

            # get all RGB bands
            band_data_list = []
            for band_name in band_names:
                band_data = self._masked_array_to_nan(self.data[band_name])
                band_data_list.append(band_data)
            # add transparency layer
            band_data_list.append(np.zeros_like(band_data_list[0]))
            band_data = np.dstack(band_data_list)

            # use 'green' henceforth to extract the corresponding meta-data
            band_name = 'green'
            # no colormap required
            colormap = None

        # check if band data is a masked array
        # band_data = self._masked_array_to_nan(band_data=band_data)
        if isinstance(band_data, np.ma.core.MaskedArray):
            mask = band_data.mask
            band_data = deepcopy(band_data.data)
            band_data[mask] = np.nan

        # adjust transparency in case of RGBA arrays
        if len(band_data.shape) == 3:
            tmp = deepcopy(band_data[:,:,0])
            tmp[~np.isnan(tmp)] = 1.
            tmp[np.isnan(tmp)] = 0.
            band_data[:,:,3] = tmp

        # get bounds amd EPSG code
        if self.from_bandstack():
            bounds = self.data['bounds']
            epsg = self.data['meta']['crs'].to_epsg()
        else:
            bounds = self.data['bounds'][band_name]
            epsg = self.data['meta'][band_name]['crs'].to_epsg()

        # determine intervals for plotting and aspect ratio (figsize)
        east_west_dim = bounds.right - bounds.left
        if abs(east_west_dim) < 5000:
            x_interval = 500
        elif abs(east_west_dim) >= 5000 and abs(east_west_dim) < 100000:
            x_interval = 5000
        else:
            x_interval = 50000
        north_south_dim = bounds.top - bounds.bottom
        if abs(north_south_dim) < 5000:
            y_interval = 500
        elif abs(north_south_dim) >= 5000 and abs(north_south_dim) < 100000:
            y_interval = 5000
        else:
            y_interval = 50000

        w_h_ratio = figaspect(east_west_dim / north_south_dim)

        # open figure and axes for plotting
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=w_h_ratio,
            num=1,
            clear=True
        )

        # get colormap
        cmap = user_defined_colors
        if cmap is None:
            cmap = plt.cm.get_cmap(colormap)

        # check if data is continuous (spectral) or discrete (np.unit8)
        if discrete_values:
            # define the bins and normalize
            unique_values = np.unique(band_data)
            norm = mpl.colors.BoundaryNorm(unique_values, cmap.N)
            img = ax.imshow(
                band_data,
                cmap=cmap,
                norm=norm,
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
            )

        else:

            # clip data for displaying to central 90%, i.e., discard upper and
            # lower 5% of the data
            lower_bound = np.nanquantile(band_data, 0.05)
            upper_bound = np.nanquantile(band_data, 0.95)

            # be even stricter for RGB plot
            if rgb_plot:
                if lower_bound == 0:
                    lower_bound = np.nanquantile(band_data[band_data > 0], 0.05)

            img = ax.imshow(
                band_data,
                vmin=lower_bound,
                vmax=upper_bound,
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                cmap=cmap
            )

        # add colorbar (does not apply in RGB case)
        if colormap is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(img, cax=cax, orientation='vertical')
            # overwrite ticker if user defined ticks provided
            if user_defined_ticks is not None:
                cb.ax.locator_params(nbins=len(user_defined_ticks))
                cb.set_ticklabels(user_defined_ticks)     

        if colormap is None:
            if rgb_plot:
                ax.title.set_text('True Color Image')
            elif nir_plot:
                ax.title.set_text('False Color Nir-Infrared Image')
        else:
            ax.title.set_text(f'Band: {band_name.upper()}')

        # add axes labels
        ax.set_xlabel(f'X [m] (EPSG:{epsg})', fontsize=12)
        ax.xaxis.set_ticks(np.arange(bounds.left, bounds.right, x_interval))
        ax.set_ylabel(f'Y [m] (EPSG:{epsg})', fontsize=12)
        ax.yaxis.set_ticks(np.arange(bounds.bottom, bounds.top, y_interval))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

        return fig


    def calc_vi(
            self,
            vi: str
        ) -> None:
        """
        Calculates a vegetation index implemented in `agrisatpy.analysis.vegetation_indices`
        and adds the vegetation index as a new band to the data dict.

        Raises an error if the index calculation fails.

        :param vi:
            name of the vegetation index to calculate (e.g., NDVI)
        """
        try:
            vi_obj = VegetationIndices(reader=self)
            vi_data = vi_obj.calc_vi(vi)
            self.add_band(band_name=vi, band_data=vi_data)
        except Exception as e:
            raise Exception(f'Could not calculate vegetation index "{vi}": {e}')


    def resample(
            self,
            target_resolution: Union[int,float],
            resampling_method: Optional[int] = cv2.INTER_CUBIC,
            pixel_division: Optional[bool] = False,
            band_selection: Optional[List[str]] = [],
            bands_to_exclude: Optional[List[str]] = []
        ) -> None:
        """
        resamples band data on the fly if required into a user-definded spatial
        resolution. The resampling algorithm used is cv2.resize and allows the
        following options:

        INTER_NEAREST - a nearest-neighbor interpolation
        INTER_LINEAR - a bilinear interpolation (used by default)
        INTER_AREA - resampling using pixel area relation.
        INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        INTER_LANCZOS4 -  a Lanczos interpolation over 8x8 pixel neighborhood

        IMPORTANT: The method overwrites the original band data when resampling
        is required!

        :param target_resolution:
            target spatial resolution in image projection (i.e., pixel size
            in meters)
        :param resampling_method:
            opencv resampling method. Per default bicubic interpolation is used
            (``cv2.INTER_CUBIC``)
        :param pixel_division:
            if set to True then pixel values will be divided into n*n subpixels 
            (only even numbers) depending on the target resolution. Takes the 
            current band resolution (for example 20m) and checks against the desired
            target_resolution and applies a scaling_factor. 
            This works, however, only if the spatial resolution is increased, e.g.
            from 20 to 10m. The ``resampling_method`` argument is ignored then.
            Default value is False.
        :param band_selection:
            list of bands to consider. Per default all bands are used but
            not processed if the bands already have the desired spatial
            resolution
        :param bands_to_exclude:
            list of bands NOT to consider for resampling. Per default all
            bands are considered for resampling (see band_selection).
        """

        # loop over bands and resample those bands not in the desired
        # spatial resolution

        if len(band_selection) == 0:
            band_selection = list(self.data.keys())
            # remove those items not containing band data
            band_selection.remove('meta')
            band_selection.remove('bounds')

        # check if bands are to exclude
        if len(bands_to_exclude) > 0:
            set_to_exclude = set(bands_to_exclude)
            band_selection = [x for x in band_selection if x not in set_to_exclude]

        # if the data comes from a bandstack then the spatial resolution is
        # the same for all bands     
        if self.from_bandstack():
            meta = self.data['meta']
            bounds = self.data['bounds']

        snap_band_available = False
        for idx, band in enumerate(band_selection):

            pixres = self.get_meta(band)['transform'][0]

            # check if resampling is required, if the band has
            # already the target resolution use its extent to snap
            # the other rasters too
            if pixres == target_resolution:
                snap_band_available = True
                snap_shape = self.data[band].shape
                snap_meta = self.data['meta'][band]
                # check if the data is masked, in that case also provide
                # a snap mask
                if isinstance(self.data[band], np.ma.core.MaskedArray):
                    snap_mask = self.data[band].mask
                break

        # raise a warning if no snap raster is available. The results might then
        # not be perfectly aligned and differ in extent and localization!
        if not snap_band_available:
            raise Warning('No snap raster band found for resampling. '\
                          'Results might differ in shape!')

        for idx, band in enumerate(band_selection):

            meta = self.get_meta(band)
            pixres = meta['transform'][0]
            if pixres == target_resolution:
                continue

            # check if coordinate system is projected, geographic
            # coordinate systems are not supported
            if not meta['crs'].is_projected:
                raise NotProjectedError(
                    'Resampling from geographic coordinates is not supported'
            )

            # check if a 'snap' band is available
            if snap_band_available:
                nrows_resampled = snap_shape[0]
                ncols_resampled = snap_shape[1]
            # if not determine the extent from the bounds
            else:
                # calculate new size of the raster
                ncols_resampled = int(np.ceil((bounds.right - bounds.left) / target_resolution))
                nrows_resampled = int(np.ceil((bounds.top - bounds.bottom) / target_resolution))

            # opencv2 switches the axes order!
            dim_resampled = (ncols_resampled, nrows_resampled)
            band_data = self.data[band]

            # check if the band data is stored in a masked array
            # if so, replace the masked values with NaN
            if isinstance(band_data,  np.ma.core.MaskedArray):
                band_data = deepcopy(band_data.data)

            # resample the array using opencv resize or pixel_division
            band_res = self.get_meta(band_name=band)['transform'][0]
            scaling_factor = int(band_res / target_resolution)
            if pixel_division:
                # determine scaling factor as ratio between current and target spatial resolution
                res = upsample_array(
                    in_array=band_data,
                    scaling_factor=scaling_factor
                )
            else:
                try:
                    res = cv2.resize(
                        band_data,
                        dsize=dim_resampled,
                        interpolation=resampling_method
                    )
                except Exception as e:
                    raise ResamplingFailedError(e)

            # overwrite entries in self.data
            # if the array is masked, use the masked array from the snap raster
            # or create a new one in the target resolution using pixel division
            if isinstance(self.data[band], np.ma.core.MaskedArray):
                if not snap_band_available:
                    in_mask = deepcopy(self.data[band].mask)
                    snap_mask = upsample_array(
                        in_array=in_mask,
                        scaling_factor=scaling_factor
                    )
                # save as masked array to back to data dict
                self.data[band] = np.ma.masked_array(res, mask=snap_mask)
            else:
                self.data[band] = res

            # for data from bandstacks updating meta is required only once
            # since all bands have the same spatial resolution
            if self.from_bandstack() and idx > 0:
                continue

            # check if meta is available from band in target resolution
            # or has to be calculated
            if snap_band_available:
                meta_resampled = snap_meta
            else:
                meta_resampled = deepcopy(meta)
                # update width, height and the transformation
                meta_resampled['width'] = ncols_resampled
                meta_resampled['height'] = nrows_resampled
                affine_orig = meta_resampled['transform']
                affine_resampled = Affine(
                    a=target_resolution,
                    b=affine_orig.b,
                    c=affine_orig.c,
                    d=affine_orig.d,
                    e=-target_resolution,
                    f=affine_orig.f
                )
                meta_resampled['transform'] = affine_resampled

            if not self.from_bandstack():
                self.data['meta'][band] = meta_resampled

        if self.from_bandstack():
            # check if any band was resampled. else leave everything as it is
            if meta_resampled is not None:
                self.data['meta'] = meta_resampled


    def mask(
            self,
            name_mask_band: str,
            mask_values: List[Union[int,float]],
            bands_to_mask: List[str],
            keep_mask_values: Optional[bool]=False
        ) -> None:
        """
        Allows to mask parts of an image (i.e., single bands) based
        on a mask band. The mask band must have the same spatial extent and
        resolution as the bands to mask (you might have to consider
        spatial resampling using the `resampling` method) and must
        be an entry in the data dict (using `add_band`) if it is not yet part
        of it.

        Masking currently only support floating data types.

        :param name_mask_band:
            name of the band (key in data dict) that contains the mask
        :param mask_values:
            specify those value(s) of the mask that denote VALID or INVALID
            categories (per default: INVALID, but it can be switched by
            using the `keep_mask_values=True` option)
        :param bands_to_mask:
            list of bands to which to apply the mask. The bands must have the same
            extent and resolution as the mask layer.
        :param keep_mask_values:
            if False (Def) the provided `mask_values` are assumed to represent
            INVALID classes, if True the opposite is the case
        """

        # check if mask band is available
        if not name_mask_band in self.data.keys():
            raise BandNotFoundError(f'{name_mask_band} is not in data dict')

        # convert the mask to a temporary binary mask
        tmp = np.zeros_like(self.data[name_mask_band])
        # set valid classes to 1, the other ones are zero
        if keep_mask_values:
            tmp[np.isin(self.data[name_mask_band], mask_values)] = 1
        else:
            tmp[~np.isin(self.data[name_mask_band], mask_values)] = 1

        # loop over bands specified and mask the invalid pixels
        for band_to_mask in bands_to_mask:
            if band_to_mask not in self.data.keys():
                raise BandNotFoundError(f'{band_to_mask} is not in data dict')
            # set values to NaN where tmp is zero
            self.data[band_to_mask][tmp == 0] = np.nan
        

    def read_from_bandstack(
            self,
            fname_bandstack: Path,
            in_file_aoi: Optional[Path] = None,
            full_bounding_box_only: Optional[bool] = False,
            blackfill_value: Optional[Union[int,float]] = 0
        ) -> None:
        """
        Reads spectral bands from a band-stacked geoTiff file
        using the band description to extract the required spectral band
        and store them in a dict with the required band names.

        IMPORTANT: This method assumes that the band-stack was created in
        the way `~agrisatpy.processing.resampling` does, i.e., assigning
        a name to each band in the geoTiff stack.

        IMPORTANT: To map band names to color names it might be necessary
        to implement this method in the inheriting classes. See
        `agrisatpy.utils.io.sentinel2` for an example how to override this
        method

        The method populates the self.data attribute that is a
        dictionary with the following items:
            <name-of-the-band>: <np.array> denoting the spectral band data
            <meta>:<rasterio meta> denoting the georeferencation
            <bounds>: <BoundingBox> denoting the bounding box of the band data

        :param fname_bandstack:
            file-path to the bandstacked geoTiff file to read.
        :param in_file_aoi:
            vector file (e.g., ESRI shapefile or geojson) defining geometry/ies
            (polygon(s)) for which to extract the Sentinel-2 data. Can contain
            one to many features.
        :param full_bounding_box_only:
            if set to False, will only extract the data for those geometry/ies
            defined in in_file_aoi. If set to False, returns the data for the
            full extent (hull) of all features (geometries) in in_file_aoi.
        :param blackfill_value:
            value indicating black fill. Set to zero by default.
        """

        # check bounding box
        masking = False
        if in_file_aoi is not None:
            masking = True
            gdf_aoi = check_aoi_geoms(
                in_file_aoi=in_file_aoi,
                fname_sat=fname_bandstack,
                full_bounding_box_only=full_bounding_box_only
            )
    
        with rio.open(fname_bandstack, 'r') as src:
            # get geo-referencation information
            meta = src.meta
            # and bounds which are helpful for plotting
            bounds = src.bounds
            # read relevant bands and store them in dict
            band_names = src.descriptions
            self.data = dict.fromkeys(band_names)
            for idx, band_name in enumerate(band_names):
                if not masking:
                    self.data[band_name] = src.read(idx+1)
                else:
                    self.data[band_name], out_transform = rio.mask.mask(
                            src,
                            gdf_aoi.geometry,
                            crop=True, 
                            all_touched=True, # IMPORTANT!
                            indexes=idx+1,
                            filled=False
                        )
                    # update meta dict to the subset
                    meta.update(
                        {
                            'height': self.data[band_name].shape[0],
                            'width': self.data[band_name].shape[1], 
                            'transform': out_transform
                         }
                    )
                    # and bounds
                    left = out_transform[2]
                    top = out_transform[5]
                    right = left + meta['width'] * out_transform[0]
                    bottom = top + meta['height'] * out_transform[4]
                    bounds = BoundingBox(left=left, bottom=bottom, right=right, top=top)

        # check for black-fill
        is_blackfilled = self.is_blackfilled()
        if is_blackfilled:
            raise BlackFillOnlyError('AOI contains blackfill, only')

        # meta and bounds are saved as additional items of the dict
        meta.update(
            {'count': len(self.data)}
        )
        self.data['meta'] = meta
        self.data['bounds'] = bounds

        self._from_bandstack = True


    def write_bands(
            self,
            out_file: Path,
            band_selection: Optional[List[str]] = [],
            band_aliases: Optional[List[str]] = []
        ) -> None:
        """
        Writes one or multiple bands to a raster file using rasterio. By
        default a geoTiff is written since rasterio recommends this option over
        other geospatial image formats such as JPEG2000.

        IMPORTANT: The method can only write bands to file that have the same
        spatial resolution and extent. If that's not the case you eiher have to
        resample the data first using the ``resample`` method or write only those
        bands that fullfil the aforementioned criteria.

        IMPORTANT: If the bands do not have the same datatype they will be all set
        to ``numpy.float64``.

        :param out_file:
            file-path where to save the raster to. The file-ending will determine
            the type of raster generated; we recommend to use geoTiff (*.tif)
        :param driver:
            one of the GDAL drivers supported by rasterio. By default, the geoTiff
            driver is used since this driver is reported to be the most stable one.
        :param band_selection:
            list of bands to export (optional). If empty all bands available are
            exported to raster.
        :param band_aliases:
            optional list of alias band names to overwrite the (color) names of
            the bands available in the data dict (optional). If provided must
            have the same length as bands to export.
        """

        # check output file naming and driver
        try:
            driver = driver_from_extension(out_file)
        except Exception as e:
            raise ValueError(
                f'Could not determine GDAL driver for {out_file.name}: {e}'
            )

        # check band_selection, if not provided use all available bands
        if len(band_selection) > 0:
            if not all(band_name in self.data.keys() for band_name in band_selection):
                raise BandNotFoundError('Mismatch between selected and available bands')
        else:
            band_selection = self.get_bandnames()

        # check if band aliases (if provided) match the band_selection
        if len(band_aliases) > 0:
            if len(band_aliases) != len(band_selection):
                raise ValueError(
                    f'Number of selected bands ({len(band_selection)}) '\
                    f'does not match number of aliases ({len(band_aliases)})'
                )

        # check meta and update it with the selected driver for writing the result
        if self._from_bandstack:
            meta = self.get_band(band_name='meta')
        else:
            # if not from bandstack use the meta information of the first selected band
            meta = self.get_band(band_name='meta').get(band_selection[0])

        # check if all bands have the same shape, the first band determines the
        # shape all other bands have to follow
        first_shape = self.get_band(band_name=band_selection[0]).shape

        if len(band_selection) > 1:
            for band_name in band_selection[1:]:
                next_shape = self.get_band(band_name=band_name).shape
                if first_shape != next_shape:
                    raise ValueError(
                        f'The shapes of band "{band_selection[0]}" and "{band_name}"'\
                        f' differ: {first_shape} != {next_shape}'
                    )

        # check datatype of the bands and use the highest one
        dtype = self.get_band(band_name=band_selection[0]).dtype
        dtype_str = str(dtype)

        if len(band_selection) > 1:
            if not all(isinstance(self.get_band(x).dtype, int) for x in self.get_bandnames()):
                dtype = np.float64
                dtype_str = 'float64'

        # update driver and the number of bands
        meta.update(
            {
                'driver': driver,
                'count': len(band_selection),
                'dtype': dtype_str
            }
        )

        # open the result dataset and try to write the bands
        with rio.open(out_file, 'w+', **meta) as dst:

            for idx, band in enumerate(band_selection):

                # check with band name to set
                band_name = band
                if len(band_aliases) > 0:
                    band_name = band_aliases[idx]
                dst.set_band_description(idx+1, band_name)

                # write band data
                band_data = self.get_band(band_name=band).astype(dtype)
                band_data = self._masked_array_to_nan(band_data)
                dst.write(band_data, idx+1)
