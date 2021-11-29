
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import rasterio as rio
import rasterio.mask

from rasterio import Affine
from rasterio.coords import BoundingBox
from pathlib import Path
from typing import Optional
from typing import Dict
from typing import List
from typing import Union
from matplotlib.pyplot import Figure
from matplotlib.figure import figaspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from agrisatpy.utils.exceptions import NotProjectedError, ResamplingFailedError
from copy import deepcopy

from agrisatpy.utils.exceptions import BandNotFoundError
from agrisatpy.utils.reprojection import check_aoi_geoms


class Sat_Data_Reader(object):
    """abstract class from which to sensor-specific classes inherit"""

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
            return band_data.filled(np.nan)
        else:
            return band_data

    def plot_band(
            self,
            band_name: str,
            colormap: Optional[str]='gray'
        ) -> Figure:
        """
        plots a custom band using matplotlib.pyplot.imshow and the
        extent in the projection of the image data. Returns
        a figure object with the plot

        :param band_name:
            name of the band to plot
        :param colormap:
            String identifying one of matplotlib's colormaps.
            The default will plot the band in gray values.
        :return:
            matplotlib figure object with the band data
            plotted as map
        """

        if band_name not in list(self.data.keys()):
            raise BandNotFoundError(f'{band_name} not found in data')
        band_data = self.data[band_name]

        # check if band data is a masked array
        band_data = self._masked_array_to_nan(band_data=band_data)

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
        else:
            x_interval = 5000
        north_south_dim = bounds.top - bounds.bottom
        if abs(north_south_dim) < 5000:
            y_interval = 500
        else:
            y_interval = 5000

        w_h_ratio = figaspect(east_west_dim / north_south_dim)

        # open figure and axes for plotting
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=w_h_ratio,
            num=1,
            clear=True
        )

        # clip data for displaying to central 90%, i.e., discard upper and
        # lower 5% of the data
        lower_bound = np.nanquantile(band_data, 0.05)
        upper_bound = np.nanquantile(band_data, 0.95)
        img = ax.imshow(
            band_data,
            vmin=lower_bound,
            vmax=upper_bound,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            cmap=colormap
        )
        # add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img, cax=cax, orientation='vertical')
        ax.title.set_text(f'Band: {band_name.upper()}')

        # add axes labels
        ax.set_xlabel(f'X [m] (EPSG:{epsg})', fontsize=12)
        ax.xaxis.set_ticks(np.arange(bounds.left, bounds.right, x_interval))
        ax.set_ylabel(f'Y [m] (EPSG:{epsg})', fontsize=12)
        ax.yaxis.set_ticks(np.arange(bounds.bottom, bounds.top, y_interval))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

        return fig


    def resample(
            self,
            target_resolution: Union[int,float],
            resampling_method: Optional[int] = cv2.INTER_LINEAR,
            band_selection: Optional[List[str]] = []
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
            opencv resampling method. Per default linear interpolation is used
        :param band_selection:
            list of bands to consider. Per default all bands are used but
            not processed if the bands already have the desired spatial
            resolution
        """

        # loop over bands and resample those bands not in the desired
        # spatial resolution

        if len(band_selection) == 0:
            band_selection = list(self.data.keys())
            # remove those items not containing band data
            band_selection.remove('meta')
            band_selection.remove('bounds')

        # if the data comes from a bandstack then the spatial resolution is
        # the same for all bands     
        if self.from_bandstack():
            meta = self.data['meta']
            bounds = self.data['bounds']

        for idx, band in enumerate(band_selection):

            if not self.from_bandstack():
                meta = self.data['meta'][band]
                bounds = self.data['bounds'][band]

            # get original spatial resolution
            pixres = meta['transform'][0]

            # check if resampling is required
            if pixres == target_resolution:
                continue

            # check if coordinate system is projected, geographic
            # coordinate systems are not supported
            if not meta['crs'].is_projected:
                raise NotProjectedError(
                    'Resampling from geographic coordinates is not supported'
            )

            # calculate new size of the raster
            ncols_resampled = int(np.ceil((bounds.right - bounds.left) / target_resolution))
            nrows_resampled = int(np.ceil((bounds.top - bounds.bottom) / target_resolution))
            dim_resampled = (ncols_resampled, nrows_resampled)

            band_data = self.data[band]

            # check if the band data is stored in a masked array
            # if so, replace the masked values with NaN
            band_data = self._masked_array_to_nan(band_data=band_data)

            # resample the array using opencv resize
            try:
                res = cv2.resize(
                    band_data,
                    dsize=dim_resampled,
                    interpolation=resampling_method
                )
            except Exception as e:
                raise ResamplingFailedError(e)

            # overwrite entries in self.data
            self.data[band] = res

            # for data from bandstacks updating meta is required only once
            # since all bands have the same spatial resolution
            if self.from_bandstack() and idx > 0:
                continue

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


    def read_from_bandstack(
            self,
            fname_bandstack: Path,
            in_file_aoi: Optional[Path] = None,
            full_bounding_box_only: Optional[bool] = False
        ) -> Dict[str, np.array]:
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
            file-path to the bandstacked geoTiff to read
        :param in_file_aoi:
            vector file (e.g., ESRI shapefile or geojson) defining geometry/ies
            (polygon(s)) for which to extract the Sentinel-2 data. Can contain
            one to many features.
        :param full_bounding_box_only:
            if set to False, will only extract the data for those geometry/ies
            defined in in_file_aoi. If set to False, returns the data for the
            full extent (hull) of all features (geometries) in in_file_aoi.
        :return:
            dictionary with band names and corresponding band data as np.array.
            In addition, two entries in the dict provide information about the
            geo-localization ('meta') and the bounding box ('bounds') in image
            coordinates
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

        # meta and bounds are saved as additional items of the dict
        meta.update(
            {'count': len(self.data)}
        )
        self.data['meta'] = meta
        self.data['bounds'] = bounds

        self._from_bandstack = True
