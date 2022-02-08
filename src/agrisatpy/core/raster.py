'''
This module defines the ``RasterDataHandler`` class which is the basic class for reading, plotting,
transforming, manipulating and writing (geo-referenced) raster data in an intuitive, object-oriented
way (in terms of software philosophy).

A ``RasterDataHandler`` is collection of to zero to N `~agrisatpy.core.Band` instances, where each
band denotes a two-dimensional array at its core. The ``RasterDataHandler`` class allows thereby
to handle ``Band`` instances with different spatial reference systems, spatial resolutions (i.e.,
grid cell sizes) and spatial extents.

Besides that, ``RasterDataHandler`` is a super class from which sensor-specific classes for reading
(satellite) raster image data inherit.
'''

import sys

import cv2
import datetime
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.mask
import xarray as xr

from collections.abc import MutableMapping
from copy import deepcopy
from collections import namedtuple
from matplotlib.colors import ListedColormap
from matplotlib.figure import figaspect
from matplotlib.pyplot import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numbers import Number
from pathlib import Path
from PIL import Image
from rasterio import Affine, band
from rasterio import features
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
from rasterio.drivers import driver_from_extension
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.geometry import Point
from shapely.geometry import Polygon
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union
from xarray import DataArray
import xarray as xarr
import zarr

from agrisatpy.core.band import Band
from agrisatpy.core.spectral_indices import SpectralIndices
from agrisatpy.config import get_settings
from agrisatpy.core.utils.raster import get_raster_attributes
from agrisatpy.core.utils.geometry import check_geometry_types
from agrisatpy.core.utils.geometry import convert_3D_2D
from agrisatpy.utils.arrays import upsample_array
from agrisatpy.utils.exceptions import NotProjectedError, DataExtractionError
from agrisatpy.utils.exceptions import InputError
from agrisatpy.utils.exceptions import ReprojectionError
from agrisatpy.utils.exceptions import ResamplingFailedError
from agrisatpy.utils.exceptions import BandNotFoundError
from agrisatpy.utils.exceptions import BlackFillOnlyError
from agrisatpy.utils.reprojection import check_aoi_geoms
from agrisatpy.utils.arrays import count_valid
from agrisatpy.utils.reprojection import reproject_raster_dataset
from agrisatpy.utils.decorators import check_band_names
from agrisatpy.utils.decorators import check_chunksize
from agrisatpy.utils.decorators import check_metadata
from agrisatpy.utils.constants import ProcessingLevels


logger = get_settings().logger

class SceneProperties(object):
    """
    A class for storing scene-relevant properties

    :attribute acquisition_time:
        image acquisition time
    :attribute platform:
        name of the imaging platform
    :attribute sensor:
        name of the imaging sensor
    :attribute processing_level:
        processing level of the remotely sensed data (if
        known and applicable)
    :attribute scene_id:
        unique scene identifier
    """

    def __init__(
            self, 
            acquisition_time: datetime.datetime = datetime.datetime(2999,1,1),
            platform: str = '',
            sensor: str = '',
            processing_level: ProcessingLevels = ProcessingLevels.UNKNOWN,
            scene_id: str = ''
        ):
        """
        Class constructor

        :param acquisition_time:
            image acquisition time
        :param platform:
            name of the imaging platform
        :param sensor:
            name of the imaging sensor
        :param processing_level:
            processing level of the remotely sensed data (if
            known and applicable)
        :param scene_id:
            unique scene identifier
        """
        # type checking first
        if not isinstance(acquisition_time, datetime.datetime):
            raise TypeError(
                f'A datetime.datetime object is required: {acquisition_time}'
            )
        if not isinstance(platform, str):
            raise TypeError(f'A str object is required: {platform}')
        if not isinstance(sensor, str):
            raise TypeError(f'A str object is required: {sensor}')
        if not isinstance(processing_level, ProcessingLevels):
            raise TypeError(
                f'A ProcessingLevels object is required: {processing_level}'
            )
        if not isinstance(scene_id, str):
            raise TypeError(f'A str object is required: {scene_id}')

        self.acquisition_time = acquisition_time
        self.platform = platform
        self.sensor = sensor
        self.processing_level = processing_level
        self.scene_id = scene_id

    @property
    def acquisition_time(self) -> datetime.datetime:
        """acquisition time of the scene"""
        return self._acquisition_time

    @acquisition_time.setter
    def acquisition_time(self, time: datetime.datetime) -> None:
        """acquisition time of the scene"""
        if not isinstance(time, datetime.datetime):
            raise TypeError('Expected a datetime.datetime object')
        self._acquisition_time = time

    @property
    def platform(self) -> str:
        """name of the imaging platform"""
        return self._platform

    @platform.setter
    def platform(self, value: str) -> None:
        """name of the imaging plaform"""
        if not isinstance(value, str):
            raise TypeError('Expected a str object')
        self._platform = value

    @property
    def sensor(self) -> str:
        """name of the sensor"""
        return self._sensor

    @sensor.setter
    def sensor(self, value: str) -> None:
        """name of the sensor"""
        if not isinstance(value, str):
            raise TypeError('Expected a str object')
        self._sensor = value

    @property
    def processing_level(self) -> ProcessingLevels:
        """current processing level"""
        return self._processsing_level

    @processing_level.setter
    def processing_level(self, value: ProcessingLevels):
        """current processing level"""
        if not isinstance(value, ProcessingLevels):
            raise TypeError(f'Expected {ProcessingLevels}')
        self._processing_level = value

    @property
    def scene_id(self) -> str:
        """unique scene identifier"""
        return self._scene_id

    @scene_id.setter
    def scene_id(self, value: str) -> None:
        """unique scene identifier"""
        if not isinstance(value, str):
            raise TypeError('Expected a str object')
        self._scene_id = value


class RasterCollection(MutableMapping):
    """
    Basic class for storing and handling single and multi-band raster
    data from which sensor- or application-specific classes inherit.

    A ``RasterDataHandler`` contains zero to N instances of
    `~agrisatpy.core.Band`. Bands are always indexed using their band
    name, therefore the band name must be **unique**!

    :attrib scene_properties:
        instance of `SceneProperties` for storing scene (i.e., dataset-wide)
        metadata. Designed for the usage with remote sensing data.
    """

    def __init__(
            self,
            band_constructor: Optional[Callable[..., Band]] = None,
            scene_properties: Optional[SceneProperties] = None,
            *args,
            **kwargs
        ):
        """
        Initializes a new `RasterCollection` with 0 or 1 band

        Examples:
        --------
        >>> from agrisatpy.core.raster import RasterDataHandler
        >>> from agrisatpy.core.band import Band
        >>> from agrisatpy.core.band import GeoInfo

        Empty `RasterDataHandler`:
        >>> handler = RasterDataHandler()
        >>> handler.empty

        New Handler from `numpy.ndarray` (default Band constructor)
        Define GeoInfo and Array first and use them to initialize a new Handler
        instance:
        >>> epsg = 32633
        >>> ulx, uly = 300000, 5100000
        >>> pixres_x, pixres_y = 10, -10
        >>> geo_info = GeoInfo(epsg=epsg,ulx=ulx,uly=uly,pixres_x=pixres_x,pixres_y=pixres_y)
        >>> band_name = 'random'
        >>> color_name = 'blue'
        >>> values = np.random.random(size=(100,120))
        >>> handler = RasterDataHandler(
        >>>         band_constructor=Band,
        >>>         band_name=band_name,
        >>>         values=values,
        >>>         color_name=color_name,
        >>>         geo_info=geo_info
        >>> )


        :param band_constructor:
            optional callable returning an `~agrisatpy.core.Band`
            instance.
        :param scene_properties:
            optional scene properties of the dataset handled by the
            current ``RasterCollection`` instance
        :param args:
            arguments to pass to `band_constructor` or one of its
            class methods (`Band.from_rasterio`, `Band.from_vector`)
        :param kwargs:
            key-word arguments to pass to `band_constructor`  or one of its
            class methods (`Band.from_rasterio`, `Band.from_vector`)
        """

        if scene_properties is None:
            scene_properties = SceneProperties()
        if not isinstance(scene_properties, SceneProperties):
            raise TypeError('scene_properties takes only objects ' \
                            'of type SceneProperties')
        self.scene_properties = scene_properties

        # bands are stored in a dictionary like collection
        self._frozen = False
        self.collection = dict()
        self._frozen = True

        self._band_aliases = []
        if band_constructor is not None:
            band = band_constructor.__call__(*args, **kwargs)
            if not isinstance(band, Band):
                raise TypeError('Only Band objects can be passed')
            self._band_aliases.append(band.band_alias)
            self.__setitem__(band)

    def __getitem__(self, key: str) -> Band:
        # check for band alias if any
        if self.has_band_aliases:
            if key not in self.band_names:
                if key in self.band_aliases:
                    band_idx = self.band_aliases.index(key)
                    key = self.band_names[band_idx]
        return self.collection[key]

    def __setitem__(self, item: Band):
        if not isinstance(item, Band):
            raise TypeError('Only Band objects can be passed')
        key = item.band_name
        if key in self.collection.keys():
            raise KeyError('Duplicate band names not permitted')
        value = item.copy()
        self.collection[key] = value

    def __delitem__(self, key: str):
        del self.collection[key]

    def __iter__(self):
        return iter(self.collection)
    
    def __len__(self) -> int:
        return len(self.collection)

    @property
    def band_names(self) -> List[str]:
        """band names in collection"""
        return list(self.collection.keys())

    @property
    def band_aliases(self) -> List[str]:
        """band aliases in collection"""
        return self._band_aliases

    @property
    def empty(self) -> bool:
        """Handler has bands loaded"""
        return len(self.collection) == 0

    @property
    def has_band_aliases(self) -> bool:
        """collection supports aliasing"""
        return len(self.band_aliases) > 0

    @property
    def collection(self) -> MutableMapping:
        """collection of the bands currently loaded"""
        return self._collection

    @collection.setter
    def collection(self, value):
        """collection of the bands currently loaded"""
        if not isinstance(value, dict):
            raise TypeError(
                'Only dictionaries can be passed'
            )
        if self._frozen:
            raise ValueError(
                'Existing collections cannot be overwritten'
            )
        if not self._frozen:
            self._collection = value

    @check_band_names
    def get_band_alias(
            self,
            band_name: str
        ) -> Union[Dict[str, str], None]:
        """
        Retuns the band_name-alias mapping of a given band
        in collection if the band has an alias, None instead

        :param band_name:
            name of the band for which to return the alias or
            its name if the alias is provided
        :returns:
            mapping of band_name:band_alias (band name is always the
            key and band_alias is the value)
        """
        if self[band_name].has_alias:
            idx = self.band_names.index(band_name)
            band_alias = self.band_aliases[idx]
            return {band_name: band_alias}

    @staticmethod
    def _bands_from_selection(
            fpath_raster: Path,
            band_idxs: Optional[List[int]],
            band_names_src: Optional[List[str]],
            band_names_dst: Optional[List[str]]
            ) -> Dict[str, Union[str,int]]:
        """
        Selects bands in a multi-band raster dataset based on a custom
        selection of band indices or band names

        :param fpath_raster:
            file-path to the raster file (technically spoken, this
            can also have just a single band)
        :param band_idxs:
            optional list of band indices in the raster dataset
            to read. If not provided (default) all bands are loaded.
            Ignored if `band_names_src` is provided.
        :param band_names_src:
            optional list of band names in the raster dataset to
            read. If not provided (default) all bands are loaded. If
            `band_idxs` and `band_names_src` are provided, the former
            is ignored.
        :param band_names_dst:
            optional list of band names in the resulting collection.
            Must match the length and order of `band_idxs` or
            `band_names_src`
        :returns:
            dictionary with band indices, and names based on the custom
            selection
        """
        # chech band selection
        if band_idxs is None:
            try:
                with rio.open(fpath_raster, 'r') as src:
                    band_names = list(src.descriptions)
                    band_count = src.count
            except Exception as e:
                raise IOError(f'Could not read {fpath_raster}: {e}')
            # use default band names if not provided in data set
            if len(band_names) == 0:
                band_names_src = [f'B{idx+1}' for idx in range(band_count)]
            # is a selection of bands provided? If no use all available bands
            # otherwise check the band indices
            if band_names_src is None:
                # get band indices of all bands, add 1 since GDAL starts
                # counting at 1
                band_idxs = [x+1 for x in range(band_count)]
            else:
                # get band indices of selected bands (+1 because of GDAL)
                band_idxs = [band_names.index(x)+1 for x in band_names_src \
                    if x in band_names]
                band_count = len(band_idxs)

        # make sure neither band_idxs nor band_names_src is None or empty
        if band_idxs is None or len(band_idxs) == 0:
            raise ValueError(
                'No band indices could be determined'
            )

        # set band_names_dst to values of band_names_src or default names
        if band_names_dst is None:
            if band_names_src is not None:
                band_names_dst = band_names_src
            else:
                band_names_dst = band_names

        return {
            'band_idxs': band_idxs,
            'band_names_src': band_names_src,
            'band_names_dst': band_names_dst,
            'band_count': band_count
        }

    @classmethod
    def from_multi_band_raster(
            cls,
            fpath_raster: Path,
            band_idxs: Optional[List[int]] = None,
            band_names_src: Optional[List[str]] = None,
            band_names_dst: Optional[List[str]] = None,
            band_aliases: Optional[List[str]] = None,
            **kwargs
        ):
        """
        Loads bands from a multi-band raster file into a new
        `RasterCollection` instance.

        Wrapper around `~agrisatpy.core.Band.from_rasterio` for
        1 to N raster bands.

        :param fpath_raster:
            file-path to the raster file (technically spoken, this
            can also have just a single band)
        :param band_idxs:
            optional list of band indices in the raster dataset
            to read. If not provided (default) all bands are loaded.
            Ignored if `band_names_src` is provided.
        :param band_names_src:
            optional list of band names in the raster dataset to
            read. If not provided (default) all bands are loaded. If
            `band_idxs` and `band_names_src` are provided, the former
            is ignored.
        :param band_names_dst:
            optional list of band names in the resulting collection.
            Must match the length and order of `band_idxs` or
            `band_names_src`
        :param band_aliases:
            optional list of aliases to use for *aliasing* of band names
        :param kwargs:
            optional key-word arguments accepted by
            `~agrisatpy.core.Band.from_rasterio`
        :returns:
            `RasterCollection` instance with loaded bands from the
            input raster data set.
        """
        # check band selection
        band_props = cls._bands_from_selection(
            fpath_raster=fpath_raster,
            band_idxs=band_idxs,
            band_names_src=band_names_src,
            band_names_dst=band_names_dst
        )

        # make sure band aliases match the length of bands
        if band_aliases is not None:
            if len(band_aliases) != band_props['band_count']:
                raise ValueError(
                    f'Number of band_aliases ({len(band_aliases)}) does ' \
                    f'not match number of bands to load ({band_props["band_count"]})'
                )
        else:
            band_aliases =['' for _ in range(band_props['band_count'])]

        # loop over the bands and add them to an empty handler
        handler = cls()
        for band_idx in range(band_props['band_count']):
            try:
                handler.add_band(
                    Band.from_rasterio,
                    fpath_raster=fpath_raster,
                    band_idx=band_props['band_idxs'][band_idx],
                    band_name_dst=band_props['band_names_dst'][band_idx],
                    band_alias=band_aliases[band_idx],
                    **kwargs
                )
            except Exception as e:
                raise Exception(
                    f'Could not add band {band_names_src[band_idx]} ' \
                    f'from {fpath_raster} to handler: {e}'
                )
        return handler

    @classmethod
    def read_pixels(
            cls,
            fpath_raster: Path,
            vector_features: Union[Path, gpd.GeoDataFrame],
            band_idxs: List[Optional[int]] = None,
            band_names_src: List[Optional[str]] = None,
            band_names_dst: List[Optional[str]] = None
        ) -> gpd.GeoDataFrame:
        """
        Wrapper around `~agrisatpy.core.band.read_pixels` for raster datasets
        with multiple bands

        NOTE:
            The pixels to read are defined by a ``GeoDataFrame`` or file with
            vector features understood by ``fiona``. If the geometry type is not
            ``Point`` the centroids will be used for extracting the closest
            grid cell value.

        :param fpath_raster:
            file-path to the raster dataset from which to extract pixel values
        :param vector_features:
            file-path or ``GeoDataFrame`` to features defining the pixels to read
            from a raster dataset. The geometries can be of type ``Point``,
            ``Polygon`` or ``MultiPolygon``. In the latter two cases the centroids
            are used to extract pixel values, whereas for point features the
            closest raster grid cell is selected.
        ::param band_idxs:
            optional list of band indices in the raster dataset to read. If not
            provided (default) all bands are loaded. Ignored if `band_names_src` is
            provided.
        :param band_names_src:
            optional list of band names in the raster dataset to read. If not provided
            (default) all bands are loaded. If `band_idxs` and `band_names_src` are
            provided, the former is ignored.
        :param band_names_dst:
            optional list of band names in the resulting collection.Must match the length
            and order of `band_idxs` or `band_names_src`.
        :returns:
            ``GeoDataFrame`` with extracted pixel values. If the vector features
            defining the sampling points are not within the spatial extent of the
            raster dataset the pixel values are set to nodata (inferred from
            the raster source)
        """
        # check band selection
        band_props = cls._bands_from_selection(
            fpath_raster=fpath_raster,
            band_idxs=band_idxs,
            band_names_src=band_names_src,
            band_names_dst=band_names_dst
        )

        # loop over bands and extract values from raster dataset
        for idx in range(band_props['band_count']):
            if idx == 0:
                gdf = Band.read_pixels(
                    fpath_raster=fpath_raster,
                    vector_features=vector_features,
                    band_idx=band_props['band_idxs'][idx],
                    band_name_src=band_props['band_names_src'][idx],
                    band_name_dst=band_props['band_names_dst'][idx]
                )
            else:
                gdf = Band.read_pixels(
                    fpath_raster=fpath_raster,
                    vector_features=gdf,
                    band_idx=band_props['band_idxs'][idx],
                    band_name_src=band_props['band_names_src'][idx],
                    band_name_dst=band_props['band_names_dst'][idx]
                )

        return gdf

    def drop_band(self, band_name: str):
        """
        Deletes a band from the current collection

        :param band_name:
            name of the band to drop
        """
        self.__delitem__(band_name)

    def is_bandstack(
            self,
            band_selection: Optional[List[str]] = None
        ) -> Union[bool, None]:
        """
        Checks if the rasters handled in the collection fulfill the bandstack
        criteria.

        These criteria are:
            - all bands have the same CRS
            - all bands have the same x and y dimension (number of rows and columns)
            - all bands must have the same upper left corner coordinates

        :param band_selection:
            if not None, checks only a list of selected bands. By default,
            all bands of the current object are checked.
        :returns:
            True if the current object fulfills the criteria else False;
            None if no bands are loaded into the handler's collection.
        """
        if band_selection is None:
            band_selection = self.band_names
        else:
            if not all(elem in self.band_names for elem in band_selection):
                raise BandNotFoundError(f'Invalid selection of bands')

        # return None if no bands are in collection
        if len(band_selection) == 0:
            return None

        # otherwise use the first band (that will then always exist)
        # as reference to check the other bands (if any) against
        first_geo_info = self[band_selection[0]].geo_info
        first_shape = (
            self[band_selection[0]].nrows,
            self[band_selection[0]].ncols
        )
        for idx in range(1, len(band_selection)):
            this_geo_info = self[band_selection[idx]].geo_info
            this_shape = (
                self[band_selection[idx]].nrows,
                self[band_selection[idx]].ncols
            )
            if this_shape != first_shape:
                return False
            if this_geo_info.epsg != first_geo_info.epsg:
                return False
            if this_geo_info.ulx != first_geo_info.ulx:
                return False
            if this_geo_info.uly != first_geo_info.uly:
                return False
            if this_geo_info.pixres_x != first_geo_info.pixres_x:
                return False
            if this_geo_info.pixres_y != first_geo_info.pixres_y:
                return False

        return True

    def add_band(
            self,
            band_constructor: Callable[..., Band],
            *args,
            **kwargs
        ) -> None:
        """
        Adds a band to the collection of raster bands.

        Raises an error if a band with the same name already exists (unique
        name constraint)

        :param band_constructor:
            callable returning a `~agrisatpy.core.Band` instance
        :param args:
            arguments to pass to `band_constructor` or one of its
            class methods (`Band.from_rasterio`, `Band.from_vector`)
        :param kwargs:
            key-word arguments to pass to `band_constructor`  or one of its
            class methods (`Band.from_rasterio`, `Band.from_vector`)
        """
        try:
            band = band_constructor.__call__(*args, **kwargs)
        except Exception as e:
            raise ValueError(f'Cannot initialize new Band instance: {e}')
        
        try:
            self.__setitem__(band)
            # forward band alias if any
            if band.has_alias:
                self._band_aliases.append(band.band_alias)
        except Exception as e:
            raise KeyError(f'Cannot add raster band: {e}')

    @check_band_names
    def plot_band(
            self,
            band_name: str,
            **kwargs
        ) -> Figure:
        """
        Plots a band in the collection of raster bands.

        Wrapper method around `~agrisatpy.core.Band.plot`.

        :param band_name:
            name of the band to plot. Aliasing is supported.
        :param kwargs:
            key-word arguments to pass to `~agrisatpy.core.Band.plot`
        :returns:
            `~matplotlib.pyplot.Figure` with band plotted as map
        """
        return self[band_name].plot(**kwargs)

    @check_band_names
    def plot_multiple_bands(
            self,
            band_selection: Optional[List[str]] = None,
            **kwargs
        ):
        """
        Plots three selected bands in a pseudo RGB with 8bit color-depth.

        IMPORTANT:
            The bands to plot **must** have the same spatial resolution,
            extent and CRS

        :param band_selection:
            optional list of bands to plot. If not provided takes the
            first three bands (or less) to plot
        :returns:
            `~matplotlib.pyplot.Figure` with band plotted as map in
            8bit color depth
        """
        # check passed band_selection
        if band_selection is None:
            band_selection = self.band_names
        # if one band was passed only call plot band
        if len(band_selection) == 1:
            return self.plot_band(band_name=band_selection[0], **kwargs)

        # if too many bands are passed take the first three
        if len(band_selection) > 3:
            band_selection = band_selection[0:3]
        # but raise an error when less than three bands are available
        # unless it's
        elif len(band_selection) < 3:
            raise ValueError('Need three bands to plot')

        # check if data can be stacked
        if not self.is_bandstack(band_selection):
            raise ValueError(
                'Bands to plot must share same spatial extent, pixel size and CRS'
            )

        # get bounds in the spatial coordinate system for plotting
        xmin, ymin, xmax, ymax = self[band_selection[0]].bounds.exterior.bounds
        # clip values to 8bit color depth
        array_list = []
        for band_name in band_selection:
            band_data = self.get_band(band_name).values
            new_arr = ((band_data - band_data.min()) * \
                       (1/(band_data.max() - band_data.min()) * \
                        255)).astype('uint8')
            array_list.append(new_arr)
        # stack arrays into 3d array
        stack = np.dstack(array_list)
        # get quantiles to improve plot visibility
        vmin = np.nanquantile(stack, 0.1)
        vmax = np.nanquantile(stack, 0.9)
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)
        ax.imshow(
            stack,
            vmin=vmin,
            vmax=vmax,
            extent=[xmin, xmax, ymin, ymax]
        )
        # set axis labels
        epsg = self[band_selection[0]].geo_info.epsg
        if self[band_selection[0]].crs.is_geographic:
            unit = 'deg'
        elif self[band_selection[0]].crs.is_projected:
            unit = 'm'
        fontsize = kwargs.get('fontsize', 12)
        ax.set_xlabel(f'X [{unit}] (EPSG:{epsg})', fontsize=fontsize)
        ax.set_ylabel(f'Y [{unit}] (EPSG:{epsg})', fontsize=fontsize)
        # add title str
        title_str = ", ".join(band_selection)
        ax.set_title(title_str, fontdict={'fontsize': fontsize})

        return fig

    @check_band_names
    def get_band(self, band_name: str) -> Union[Band, None]:
        """
        Returns a single band from the collection or None
        if the band is not found.

        :param band_name:
            band name (or its alias) to return
        :returns:
            ``Band`` instance from band name
        """
        return self.collection.get(band_name, None)

    # @check_band_names
    def get_pixels(
            self,
            vector_features: Union[Path, gpd.GeoDataFrame],
            band_selection: Optional[List[str]] = None
        ) -> gpd.GeoDataFrame:
        """
        Returns pixel values from bands in the collection as ``GeoDataFrame``.

        Since a pixel is a dimensionless object (``Point``) the extraction
        method works for raster bands with different pixel sizes, spatial
        extent and coordinate systems. If a pixel cannot be extracted from a
        raster band, the band's nodata value is inserted.

        :param band_selection:
            optional selection of bands to return
        :param vector_features:
            file-path or ``GeoDataFrame`` to features defining the pixels to read
            from the raster bands selected. The geometries can be of type ``Point``,
            ``Polygon`` or ``MultiPolygon``. In the latter two cases the centroids
            are used to extract pixel values, whereas for point features the
            closest raster grid cell is selected.
        :returns:
            ``GeoDataFrame`` with extracted raster values per pixel or
            Polygon centroid.
        """
        if band_selection is None:
            band_selection = self.band_names

        # loop over bands and extract the raster values into a GeoDataFrame
        for idx, band_name in enumerate(band_selection):
            # open a new GeoDataFrame for the first band and re-use it for
            # the other bands. This way we do not have to merge the single
            # GeoDataFrames afterwards
            if idx == 0:
                gdf = self[band_name].get_pixels(vector_features=vector_features)
            else:
                gdf = self[band_name].get_pixels(vector_features=gdf)

        return gdf

    @check_band_names
    def get_values(
            self,
            band_selection: Optional[List[str]] = None
        ) -> Union[np.ma.MaskedArray, np.ndarray]:
        """
        Returns raster values as stacked array in collection.

        NOTE:
            The selection of bands to return as stacked array
            **must** share the same spatial extent, pixel size
            and coordinate system

        :param band_selection:
            optional selection of bands to return
        :returns:
            raster band values in their underlying storage
            type (``numpy.ndarray``, ``numpy.ma.MaskedArray``,
            ``zarr``)
        """
        if band_selection is None:
            band_selection = self.band_names

        # check if the selected bands have the same spatial extent, pixel
        # cell size and spatial coordinate system (if not stacking fails)
        if not self.is_bandstack(band_selection):
            raise ValueError(
                'Cannot stack raster bands - they do not align spatially ' \
                'to each other.\nConsider reprojection/ resampling first.'
            )

        stack_bands = [self.get_band(x).values for x in band_selection]
        array_types = [type(x) for x in stack_bands]

        # stack arrays along first axis
        # we need np.ma in case the array is a masked array
        if set(array_types) == {np.ma.MaskedArray}:
            return np.ma.stack(stack_bands, axis=0)
        elif set(array_types) == {np.ndarray}:
            return np.stack(stack_bands, axis=0)
        elif set(array_types) == {zarr.core.Array}:
            raise NotImplementedError()
        else:
            raise ValueError('Unsupported array type')

    @check_band_names
    def reproject(
            self,
            band_selection: Optional[List[str]] = None,
            inplace: Optional[bool] = False,
            **kwargs
        ):
        """
        Reprojects band in the collection from one coordinate system
        into another

        :param band_selection:
            selection of bands to process. If not provided uses all
            bands
        :param inplace:
            if False returns a new `RasterCollection` (default) otherwise
            overwrites existing raster band entries
        :param kwargs:
            key-word arguments to pass to `~agrisatpy.core.Band.reproject`
        :returns:
            new RasterCollection if `inplace==False`, None otherwise
        """
        if band_selection is None:
            band_selection = self.band_names
        # initialize a new raster collection if inplace is False
        collection = None
        kwargs.update({'inplace': True})
        if not inplace:
            collection = RasterCollection()
            kwargs.update({'inplace': False})

        # loop over band reproject the selected ones
        for band_name in band_selection:
            if inplace:
                self.collection[band_name].reproject(**kwargs)
            else:
                band = self.get_band(band_name)
                collection.add_band(
                    band_constructor=band.reproject,
                    **kwargs
                )

        return collection

    @check_band_names
    def resample(
            self,
            band_selection: Optional[List[str]] = None,
            inplace: Optional[bool] = False,
            **kwargs
        ):
        """
        Resamples band in the collection into a different spatial resolution

        :param band_selection:
            selection of bands to process. If not provided uses all
            bands
        :param inplace:
            if False returns a new `RasterCollection` (default) otherwise
            overwrites existing raster band entries
        :param kwargs:
            key-word arguments to pass to `~agrisatpy.core.Band.resample`
        :returns:
            new RasterCollection if `inplace==False`, None otherwise
        """
        if band_selection is None:
            band_selection = self.band_names
        # initialize a new raster collection if inplace is False
        collection = None
        kwargs.update({'inplace': True})
        if not inplace:
            collection = RasterCollection()
            kwargs.update({'inplace': False})

        # loop over band reproject the selected ones
        for band_name in band_selection:
            if inplace:
                self.collection[band_name].resample(**kwargs)
            else:
                band = self.get_band(band_name)
                collection.add_band(
                    band_constructor=band.resample,
                    **kwargs
                )

        return collection

    def mask(
            self,
            mask: Union[str, np.ndarray],
            mask_values: Optional[List[Any]],
            keep_mask_values: Optional[bool] = False,
            bands_to_mask: Optional[List[str]] = None,
            inplace: Optional[bool] = False
        ):
        """
        Masks pixels of bands in the collection using a boolean array.

        IMPORTANT:
            The mask band (or mask array) and the bands to mask **must**
            have the same shape!

        :param mask:
            either a band out of the collection (identified through its
            band name) or a ``numpy.ndarray`` of datatype boolean.
        :param mask_values:
            if `mask` is a band out of the collection, a list of values
            **must** be specified to create a boolean mask. Ignored if `mask`
            is already a boolean ``numpy.ndarray``
        :param keep_mask_values:
            if False (default), pixels in `mask` corresponding to `mask_values`
            are masked, otherwise all other pixel values are masked.
            Ignored if `mask` is already a boolean ``numpy.ndarray``.
        :param bands_to_mask:
            bands in the collection to mask based on `mask`. If not provided,
            all bands are masked
        :param inplace:
            if False returns a new `RasterCollection` (default) otherwise
            overwrites existing raster band entries
        :returns:
            new RasterCollection if `inplace==False`, None otherwise
        """
        # check mask and prepare it if required
        if isinstance(mask, np.ndarray):
            if mask.dtype != 'bool':
                raise TypeError('When providing an array it must be boolean')
            if len(mask.shape) != 2:
                raise ValueError('When providing an array it must be 2-dimensional')
        elif isinstance(mask, str):
            try:
                mask = self.get_band(mask)
            except Exception as e:
                raise ValueError(f'Invalid mask band: {e}')
            # translate mask band into boolean array
            if mask_values is None:
                raise ValueError(
                    'When using a band as mask, you have to provide a list of mask values'
                )
            # convert the mask to a temporary binary mask
            tmp = np.zeros_like(mask)
            # set valid classes to 1, the other ones are zero
            if keep_mask_values:
                # drop all other values not in mask_values
                tmp[~np.isin(mask, mask_values)] = 1
            else:
                # drop all values in mask_values
                tmp[np.isin(mask, mask_values)] = 1
            mask = tmp.astype('bool')
        else:
            raise TypeError(
                f'Mask must be either band_name or np.ndarray not {type(mask)}'
            )

        # check shapes of bands and mask before applying the mask
        if not self.is_bandstack(band_selection=bands_to_mask):
            raise ValueError(
                'Can only mask bands that have the same spatial extent, pixel size and CRS'
            )
        if mask.shape[0] != self[bands_to_mask[0]].nrows:
            raise ValueError(
                f'Number of rows in mask ({mask.shape[0]}) does not match ' \
                f'number of rows in the raster data ({self[bands_to_mask[0]].nrows})'
            )
        if mask.shape[1] != self[bands_to_mask[0]].ncols:
            raise ValueError(
                f'Number of columns in mask ({mask.shape[1]}) does not match ' \
                f'number of columns in the raster data ({self[bands_to_mask[0]].ncols})'
            )

        # initialize a new raster collection if inplace is False
        collection = None
        if not inplace:
            collection = RasterCollection()

        # loop over band reproject the selected ones
        for band_name in bands_to_mask:
            if inplace:
                self.collection[band_name].mask(
                    mask=mask,
                    inplace=inplace
                )
            else:
                band = self.get_band(band_name)
                collection.add_band(
                    band_constructor=band.mask,
                    mask=mask,
                    inplace=inplace
                )

        return collection

    def calc_si(self, si_name: str):
        """
        Calculates a spectral index based on color-names (set as band aliases)
        """
        vi_values = SpectralIndices.calc_si(si_name, self.collection)
        # look for spectral band with same shape to take geo-info from
        geo_info = [
            self[x].geo_info for x in self.band_names if \
            self[x].values.shape == vi_values.shape
        ][0]
        self.add_band(
            band_constructor=Band,
            band_name=si_name.upper(),
            geo_info=geo_info,
            band_alias=si_name.lower(),
            values=vi_values
        )

    @check_band_names
    def to_dataframe(
            self,
            band_selection: Optional[List[str]] = None
        ) -> gpd.GeoDataFrame:
        """
        Converts the bands in collection to a ``GeoDataFrame``

        :param band_selection:
            selection of bands to process. If not provided uses all
            bands
        :returns:
            ``GeoDataFrame`` with point-like features denoting single
            pixel values across bands in the collection
        """
        if band_selection is None:
            band_selection = self.band_names
        # loop over bands and convert each band to a GeoDataFrame
        for idx, band_name in enumerate(band_selection):
            gdf_band = self[band_name].to_dataframe()
            if idx == 0:
                gdf = gdf_band
            else:
                # if the bands have the same extent, pixel size and
                # CRS we cam simply append the extracted band data
                if self.is_bandstack(band_selection):
                    gdf[band_name] = gdf_band[band_name]
                # otherwise we can try to merge the pixels passed on
                # their geometries
                else:
                    gdf = gdf.join(gdf_band[band_name, 'geometry'], on='geometry')
        return gdf

    def to_rasterio(
            self,
            fpath_raster: Path,
            band_selection: Optional[List[str]] = None,
            use_band_aliases: Optional[bool] = False
        ):
        """
        Writes bands in collection to a raster dataset on disk using
        ``rasterio`` drivers

        :param fpath_raster:
            file-path to the raster dataset (existing ones will be
            overwritten!)
        :param band_selection:
            selection of bands to process. If not provided uses all
            bands
        :param use_band_aliases:
            use band aliases instead of band names for setting raster
            band descriptions to the output dataset
        """
        # check output file naming and driver
        try:
            driver = driver_from_extension(fpath_raster)
        except Exception as e:
            raise ValueError(
                f'Could not determine GDAL driver for ' \
                f'{fpath_raster.name}: {e}'
            )

        # check band_selection, if not provided use all available bands
        if band_selection is None:
            band_selection = self.band_names
        if len(band_selection) == 0:
            raise ValueError('No band selected for writing to raster file')

        # make sure all bands share the same extent, pixel size and CRS
        if not self.is_bandstack(band_selection):
            raise ValueError(
                'Cannot write bands with different shapes, pixels sizes ' \
                'and CRS to raster data set')

        # check for band aliases if they shall be used
        if use_band_aliases:
            if not self.has_band_aliases:
                raise ValueError('No band aliases available')
            band_idxs = [self.band_names.index(x) for x in band_selection]
            band_selection = [self.band_aliases[x] for x in band_idxs]

        # check meta and update it with the selected driver for writing the result
        meta = deepcopy(self[band_selection[0]].meta)
        dtypes = [self[x].values.dtype for x in band_selection]
        if len(set(dtypes)) != 1:
            UserWarning(
                f'Multiple data types found in arrays to write ({set(dtypes)}). ' \
                f'Casting to highest data type'
            )

        if len(set(dtypes)) == 1:
            dtype_str = str(dtypes[0])
        else:
            # TODO: determine highest dtype
            dtype_str = 'float32'

        # update driver and the number of bands
        meta.update({
            'driver': driver,
            'count': len(band_selection),
            'dtype': dtype_str
        })

        # open the result dataset and try to write the bands
        with rio.open(fpath_raster, 'w+', **meta) as dst:
            for idx, band_name in enumerate(band_selection):
                # check with band name to set
                dst.set_band_description(idx+1, band_name)
                # write band data
                band_data = self.get_band(band_name).values.astype(dtype_str)
                dst.write(band_data, idx+1)

    @check_band_names
    def to_xarray(
            self,
            band_selection: Optional[List[str]] = None
        ) -> xr.DataArray:
        """
        Converts bands in collection a ``xarray.DataArray``
        """
        if band_selection is None:
            band_selection = self.band_names

        # bands must have same extent, pixel size and CRS
        if not self.is_bandstack(band_selection):
            raise ValueError(
                'Selected bands must share same spatial extent, pixel size ' \
                'and coordinate system'
            )

        # loop over bands and convert them to xarray
        band_xarr_list = []
        for band_name in band_selection:
            band_xarr = self[band_name].to_xarray()
            band_xarr_list.append(band_xarr)

        # merge the single xarrays in the list into a single big one
        return xr.concat(band_xarr_list, dim='band')
  

if __name__ == '__main__':

    import pytest
    from agrisatpy.core.band import GeoInfo

    scene_props = SceneProperties()

    handler = RasterCollection()
    assert handler.empty, 'RasterCollection is not empty'
    assert handler.scene_properties.acquisition_time == datetime.datetime(2999,1,1)
    assert len(handler) == 0, 'there should not be any items so far'
    assert handler.is_bandstack() is None, 'cannot check for bandstack without bands'

    # add band to empty handler
    epsg = 32633
    ulx, uly = 300000, 5100000
    pixres_x, pixres_y = 10, -10
    geo_info = GeoInfo(epsg=epsg,ulx=ulx,uly=uly,pixres_x=pixres_x,pixres_y=pixres_y)
    zeros = np.zeros((100,100))
    band_name_zeros = 'zeros'
    handler.add_band(Band, band_name=band_name_zeros, values=zeros, geo_info=geo_info)
    assert not handler.empty, 'handler is still empty'
    assert 'zeros' in handler.band_names, 'band not found'

    # test with band instance from np.ndarray
    band_name = 'random'
    color_name = 'blue'
    values = np.random.random(size=(100,120))

    handler = RasterCollection(
        band_constructor=Band,
        band_name=band_name,
        values=values,
        band_alias=color_name,
        geo_info=geo_info
    )

    assert len(handler) == 1, 'wrong number of bands in collection'
    assert isinstance(handler[band_name], Band), 'not a proper band in collection'
    assert handler[band_name].band_name == band_name, 'incorrect band name'
    assert handler.band_names == [band_name], 'wrong number of band names'
    assert handler.band_aliases != [''], 'band aliases were not set'

    # make sure the collection is protected properly
    with pytest.raises(TypeError):
        handler.collection = 'ttt'

    with pytest.raises(ValueError):
        handler.collection = dict()

    # add a second band
    zeros = np.zeros_like(values)
    band_name_zeros = 'zeros'
    handler.add_band(Band, band_name=band_name_zeros, values=zeros, geo_info=geo_info)

    # incorrect constructor call
    # with pytest.raises(ValueError):
    #     handler.add_band(
    #         Band,
    #         band_name=band_name_zeros,
    #         values=zeros,
    #         geo_info=geo_info
    #     )

    # add a band from rasterio
    fpath_raster = Path('../../../data/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    band_idx = 1
    handler.add_band(Band.from_rasterio, fpath_raster=fpath_raster, band_idx=band_idx)

    assert set(handler.band_names) == {band_name, band_name_zeros, 'B1'}, \
        'band names not set properly in collection'
    assert (handler[band_name_zeros].values == zeros).all(), \
        'values not inserted correctly'
    assert (handler[band_name].values == values).all(), \
        'values not inserted correctly'

    # bands should not fulfill the band stack criterion
    assert not handler.is_bandstack(), 'bands must not fulfill bandstack criterion'

    # drop one of the bands again
    handler.drop_band('random')
    assert 'random' not in handler.band_names, 'band name still exists after dropping it'
    # with pytest.raises(KeyError):
    #     handler['random']

    # drop non-existing band
    # with pytest.raises(KeyError):
    #     handler.drop_band('test')

    # read multi-band geoTiff into new handler
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster
    )

    assert gTiff_collection.band_names == \
        ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'], \
        'wrong list of band names in collection'
    assert gTiff_collection.is_bandstack(), 'collection must be bandstacked'
    gTiff_collection['B02'].crs == 32632, 'wrong CRS'

    # read multi-band geoTiff into new handler with custom destination names
    colors = ['blue', 'green', 'red', 'red_edge_1', 'red_edge_2', 'red_edge_3', \
              'nir_1', 'nir_2', 'swir_1', 'swir_2']
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster,
        band_names_dst=colors
    )
    assert gTiff_collection.band_names == colors, 'band names not set properly'
    gTiff_collection.calc_si('NDVI')
    assert 'NDVI' in gTiff_collection.band_names, 'SI not added to collection'
    assert gTiff_collection['NDVI'].ncols == gTiff_collection['red'].ncols, \
        'wrong number of columns in SI'
    assert gTiff_collection['NDVI'].nrows == gTiff_collection['red'].nrows, \
        'wrong number of rows in SI'

    gdf = gTiff_collection.to_dataframe(['NDVI', 'swir_2'])
    assert set(['NDVI', 'swir_2']).issubset(gdf.columns), 'bands not added as GeoDataFrame columns'

    # with band aliases
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster,
        band_aliases=colors
    )
    assert gTiff_collection.has_band_aliases, 'band aliases must exist'

    # try reprojection of raster bands to geographic coordinates
    reprojected = gTiff_collection.reproject(target_crs=4326)
    assert reprojected.band_names == gTiff_collection.band_names, 'band names not the same'

    # reproject inplace
    gTiff_collection.reproject(target_crs=4326, inplace=True)
    assert gTiff_collection.get_band('blue').crs == reprojected.get_band('blue').crs, \
        'CRS not updated properly'

    # plot RGB
    fig_rgb = reprojected.plot_multiple_bands(band_selection=['red', 'green', 'blue'])
    assert isinstance(fig_rgb, plt.Figure), 'not a matplotlib figure'

    # resample all bands to 5m
    gTiff_collection = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster,
        band_aliases=colors
    )
    resampled = gTiff_collection.resample(
        target_resolution=5
    )
    assert resampled['green'].geo_info.pixres_x == 5, 'resolution was not changed'
    assert resampled['green'].ncols == 2206, 'wrong number of columns'
    assert resampled['green'].nrows == 2200, 'wrong number of rows'

    fpath_out = Path('/tmp/test.jp2')
    resampled.to_rasterio(fpath_out)
