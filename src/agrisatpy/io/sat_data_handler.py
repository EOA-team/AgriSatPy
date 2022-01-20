'''
This module defines the ``SatDataHandler`` class which is the basic class for reading, plotting,
transforming, manipulating and writing (geo-referenced) raster data in an intuitive, object-oriented
way (in terms of software philosophy).

It relies on ``rasterio`` for all in- and output operations to read data from files (or URIs)
``GDAL`` drivers. ``SatDataHandler``  stores the band data in a dict-like data structure
preserving the geo-spatial and related image metadata. Additionally, the class implements methods
for converting image data to ``xarray`` datasets and ``geopandas`` dataframes.

Besides that, ``SatDataHandler`` is a super class from which sensor-specific classes for reading
(satellite) raster image data inherit.
'''

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

from copy import deepcopy
from collections import namedtuple
from matplotlib.colors import ListedColormap
from matplotlib.figure import figaspect
from matplotlib.pyplot import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numbers import Number
from pathlib import Path
from rasterio import Affine
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
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union
from xarray import DataArray


from agrisatpy.analysis.spectral_indices import SpectralIndices
from agrisatpy.config import get_settings
from agrisatpy.io.utils.raster import get_raster_attributes
from agrisatpy.io.utils.geometry import check_geometry_types
from agrisatpy.io.utils.geometry import convert_3D_2D
from agrisatpy.operational.resampling.utils import upsample_array
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
    """

    acquisition_time: datetime.datetime = None
    platform: str = ''
    sensor: str = ''
    processing_level: ProcessingLevels = ProcessingLevels.UNKNOWN

    @classmethod
    def set(
            cls,
            prop: str,
            value: Any 
        ):
        """
        sets a property value

        :param prop:
            name of the property for which to set a value
        :param value:
            value to set for the property
        """
        setattr(cls, prop, value)

    @classmethod
    def get(
            cls,
            prop: str        
        ) -> Any:
        """
        returns the value of a property

        :param prop:
            name of the property for which to get a value
        """

        return eval(f'cls.{prop}')
    

class SatDataHandler(object):
    """
    basic class for handling single and multi-band raster data
    from which sensor-specific classes inherit
    """

    def __init__(self):

        self.data = {'meta': None, 'bounds': None, 'attrs': None}
        self._from_bandstack = False
        self._has_bandaliases = False
        self._band_aliases = {}
        self.scene_properties = SceneProperties


    def from_bandstack(self) -> bool:
        """
        checks if the data was read from bandstack derived from
        `~agrisatpy.operational.resampling` or custom (i.e., sensor-
        specific) file system structure such as .SAFE in case of
        Sentinel-2
        """

        return self._from_bandstack


    def add_band(
            self,
            band_name: str,
            band_data: np.array,
            band_alias: Optional[str] = None,
            snap_band: Optional[str] = None,
            band_meta: Optional[Dict[str, Any]] = None,
            band_bounds: Optional[BoundingBox] = None,
            band_attribs: Dict[str, Any] = None,
            first_band_to_add: Optional[bool] = False
        ) -> None:
        """
        Adds a band to an existing band dict instance. To keep data
        integrity, it is either required to provide a snap band (that
        provides the band metadata and attributes required), or to pass
        these explicitly.

        NOTE:
            Band name and band alias (if provided) must be unique!

        :param band_name:
            name of the band to add (must not exist already)
        :param band_data:
            band data to add; corresponds to the dict value
        :param band_alias:
            optional band name alias (must not exist already)
        :param snap_band:
            name of the band used to provide the raster metadata, geo-info
            and attributes. If the current instance fulfills the bandstack
            criteria can be ignored since all existing bands share the same
            metadata, etc.
        :param band_meta:
            ``rasterio``-like raster metadata. Is ignored if `snap_band` is
            provided
        :param band_bounds:
            ``rasterio``-like bounding box of the band denoting its spatial
            extent. Ignored if `snap_band` is provided.
        :param band_attribs:
            ``rasterio``-like image attributes of the band denoting its
            nodata information etc. Ignored if `snap_band` is provided.
        :param first_band_to_add:
            if set to True, assumes that the current instance has no other
            bands available yet; therefore the band to add is also the first
            band in the handler instance.
        """

        # check if the band is really the first band in the current instance
        existing_bands = self.get_bandnames()
        if first_band_to_add:
            if len(existing_bands) > 0:
                raise ValueError(
                    f'Sorry, but your handler already contains bands: {self.get_bandnames()}\n' \
                    'Set `first_band_to_add` to False and try it again'
                )
            if (band_meta == None and band_bounds == None and band_attribs == None):
                raise InputError(
                    'When `first_band_to_add == True` band_meta, band_bounds and band_attribs ' \
                    'MUST be passed'
                )
            # set from bandstack to True since the first band always fulfills the bandstack criteria
            self._from_bandstack = first_band_to_add

        # check band name and alias if applicable
        if band_name in self.get_bandnames():
            raise KeyError(f'Duplicated name for band: {band_name}')
        if band_alias is not None:
            if band_alias in self.get_bandaliases().values():
                raise KeyError(
                    f'Duplicated alias for band {band_name}: {band_alias}'
                )

        # check if the instance currently is a bandstack
        if not first_band_to_add:
            was_bandstack =  self.check_is_bandstack()

        # if snap_band is not passed, band meta, bounds and attribs must be provided
        if snap_band is None:
            not_passed = band_meta == None and band_bounds == None and band_attribs == None
            if not_passed:
                raise ValueError(
                    'Either a snap band or band_meta, band_bounds, band_attribs must be provided'
                )

            # validate the entries
            @check_metadata
            def _validate(self, meta_key, meta_values):
                pass

            _validate(self, meta_key='meta', meta_values=band_meta)
            _validate(self, meta_key='bounds', meta_values=band_bounds)

        # else check if the snap band has the same shape as the data to add
        else:
            if self.get_band(snap_band).shape != band_data.shape:
                raise InputError(
                    f'The shape of the band to add ({band_data.shape}) does not '\
                    f'match the reference specified ({self.get_band(snap_band).shape})'
                )
            # if yes, then we can use the meta, bounds and attribs from the snap band
            band_attribs = self.get_attrs(snap_band)
            band_meta = self.get_meta(snap_band)
            band_bounds = self.get_bounds(snap_band, return_as_polygon=False)

        # insert of band data as new entry in the data dictionary
        # if the snap raster (if provided) is a masked array, apply the mask to the
        # band to add as well
        if snap_band is not None:
            if isinstance(self.get_band(snap_band), np.ma.MaskedArray):
                mask = self.get_band(snap_band).mask
                band_data = np.ma.MaskedArray(
                    data=band_data,
                    mask=mask
                )
        band_dict = {band_name: band_data}
        self.data.update(band_dict)

        # check band aliasing
        if self._has_bandaliases:
            if band_alias is None:
                band_alias = band_name
        else:
            if band_alias is not None:
                self._has_bandaliases = True

        if band_alias is not None:
            self._band_aliases[band_name] = band_alias

        # check if attributes are already populated. If not set them
        if first_band_to_add:
            self.set_attrs(
                attr=band_attribs,
                band_name=band_name
            )
            self.set_meta(
                band_name=band_name,
                meta=band_meta)
            self.set_bounds(
                bounds=band_bounds,
                band_name=band_name
            )
        # otherwise update the attributes since they are always band-specific
        else:
            for attr in self.data['attrs']:
                if isinstance(self.data['attrs'][attr], tuple):
                    # cast to list to make attribute entries mutable
                    update_list = list(self.data['attrs'][attr])
                    
                    if attr == 'descriptions':
                        update_list.append(band_name.upper())
                    else:
                        try:
                            update_list.append(band_attribs[attr][0])
                        except Exception as e:
                            del self.data[band_name]
                            del self._band_aliases[band_name]
                            raise KeyError(f'Could not add attribute {attr}. {e}')
    
                    updated_attr = tuple(update_list)
                    self.data['attrs'].update(
                        {attr: updated_attr}
                    )
            
        # we have to check if the current instance still fulfills
        # the bandstack criteria.
        # check if the band to add is the very first band, since in this
        # case meta and bounds are None
        if first_band_to_add:
            self.data['meta'] = band_meta
            self.data['bounds'] = band_bounds
        else:
            # if it hasn't been a band stack before it's easy
            if not was_bandstack:
                self.data['meta'][band_name] = band_meta
                self.data['bounds'][band_name] = band_bounds
            # otherwise we have to check manually
            else:
                meta_check = self.get_meta(existing_bands[0]) == band_meta
                bounds_check = self.get_bounds(existing_bands[0], return_as_polygon=False) == \
                    band_bounds
                # meta and/or bands do not match with the existing bands
                # we have to unset the bandstack
                if not meta_check or not bounds_check:
                    self._unset_bandstack()
                    # and add meta and bounds for the current band
                    self.data['meta'][band_name] = band_meta
                    self.data['bounds'][band_name] = band_bounds
                # or we have the case that data seems to be bandstacked but is still
                # organized as if it was not (e.g., after spatial resamoling)
                if set(self.data['meta'].keys()) == set(existing_bands):
                    self._set_bandstack()


    def add_bands_from_vector(
            self,
            in_file_vector: Path,
            snap_band: str,
            attribute_selection: Optional[List[str]] = None,
            blackfill_value: Optional[Union[int, float]] = None,
            default_float_type: Optional[str] = 'float32'
        ) -> None:
        """
        Adds data from a vector (e.g., ESRI shapefile) file by rasterizing it and
        adding its attributes (or a selection thereof) as band entries into the
        current ``SatDataHandler`` object.

        NOTE:
            Vector attributes **must** support type casting to float, i.e., they must
            be numeric. Since `geopandas` unfortunately handles all vector attributes 
            as object data types, the method tries to cast all numeric attributes
            to `float32` (default) or `float64` to avoid loss of precision.

        :param in_file_vector:
            any vector file format supported by `fiona` containing one to many
            features with one to many (numerical) attributes
        :param snap_band:
            band in the current `SatDataHandler` instance to use for aligning the
            rasterized vector features into the band stack.
        :param attribute_selection:
            selection of vector features' attributes to rasterize and add as new
            bands (each attribute is added as new band). If None (default) uses
            all available attributes. Attributes that cannot be casted to float
            are skipped and a warning is logged.
        :param blackfill_value:
            if None infers the blackfill value (no data value) from the image
            metadata of the `snap_band`.
        :param default_float_type:
            must be either `float32` (default) or `float64`
        """

        # get snap raster transformation first
        snap_affine = self.get_meta(snap_band)['transform']
        snap_shape = self.get_band_shape(snap_band)
        snap_shape = (snap_shape.nrows, snap_shape.ncols)

        # take blackfill value from image attributes (nodatavalue) if not provided
        if blackfill_value is None:
            blackfill_value = self.get_band_nodata(snap_band)

        # check default float type
        supported_floats = ['float32', 'float64']
        if default_float_type not in supported_floats:
            raise ValueError(
                f'Default floating dtype must be one of {supported_floats}'
            )

        # check input vector file
        if not in_file_vector.exists():
            raise FileNotFoundError(f'Could not find {in_file_vector}')

        # read vector data into a GeoDataFrame
        try:
            in_gdf = gpd.read_file(in_file_vector)
        except Exception as e:
            raise Exception from e

        # check feature selection (if provided), otherwise use all attributes (columns)
        # of the dataframe
        if attribute_selection is not None:
            if not all(elem in in_gdf.columns for elem in attribute_selection):
                raise AttributeError(
                    f'Some/ all of the passed attributes not found in {in_file_vector}'
                )
        else:
            attribute_selection = list(in_gdf.columns)
            # remove the geometry attribute from the feature selection list
            attribute_selection.remove('geometry')

        # check spatial coordinate system (CRS) of the GeoDataFrame and transform
        # it, if necessary, to the CRS of the current SatDataHandler
        in_epsg = in_gdf.crs
        if in_epsg != self.get_epsg(snap_band):
            try:
                in_gdf.to_crs(self.get_epsg(snap_band), inplace=True)
            except Exception as e:
                raise Exception(
                    f'Reprojection of input vector features failed: {e}'
                )

        # clip the input to the bounds of the snap band
        snap_bounds = self.get_bounds(snap_band, return_as_polygon=True)
        try:
            in_gdf_clipped = gpd.clip(
                gdf=in_gdf,
                mask=snap_bounds
            )
        except Exception as e:
            raise DataExtractionError(
                f'Could not clip input vector features to snap raster bounds: {e}'
            )

        # make sure there are still features left
        if in_gdf_clipped.empty:
            raise DataExtractionError(
                f'Seems there are no features to rasterize from {in_file_vector}'
            )

        # rasterization using rasterio. We have to conduct it for each attribute separately
        # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
        for attribute in attribute_selection:
            try:
                # infer the datatype (i.e., try if it is possible to cast the
                # attribute to float32, otherwise do not process the feature)
                try:
                    in_gdf_clipped[attribute].astype(default_float_type)
                except ValueError as e:
                    logger.warn(f'Skipped feature "{attribute}": {e}')
                    continue

                shapes = (
                    (geom,value) for geom, value in zip(
                        in_gdf_clipped.geometry,
                        in_gdf_clipped[attribute].astype(default_float_type)
                    )
                )
                rasterized = features.rasterize(
                    shapes=shapes,
                    out_shape=snap_shape,
                    transform=snap_affine,
                    all_touched=True,
                    fill=blackfill_value,
                    dtype=default_float_type
                )
            except Exception as e:
                raise Exception(
                    f'Could not rasterize attribute "{attribute}" from {in_file_vector}: {e}'
                )
            # add band to handler
            self.add_band(
                band_name=attribute,
                band_data=rasterized,
                snap_band=snap_band
            )


    @check_band_names
    def get_band(
            self,
            band_name: str
        ) -> np.array:
        """
        Returns the ``numpy.array`` containing the data for one band

        :param band_name:
            name of the band to extract
        :return:
            band data as ``numpy.array`` (two dimensional)
        """

        # return np array
        return self.data[band_name]


    @check_band_names
    def get_bands(
            self,
            band_names: Optional[List[str]] = None
        ) -> np.array:
        """
        Returns a stack of all or a selected number of bands
        as three-dimensional numpy array (bands are 2D).
        Calls ``np.dstack()``, therefore all bands MUST have the same
        shape.

        :param band_names:
            if not provided stacks all bands
        :return:
            3d array of stacked bands
        """

        if band_names is None:
            band_names = self.get_bandnames()

        # check band shapes
        band_shapes = []
        for band_name in band_names:
            band_shapes.append(self.get_band_shape(band_name))

        if len(set(band_shapes)) > 1:
            raise InputError('All bands must have the same shape')

        stack_bands = [self.get_band(x) for x in band_names]

        # stack arrays along first axis
        # we need np.ma in case the array is a masked array
        if isinstance(stack_bands[0], np.ma.MaskedArray):
            return np.ma.stack(stack_bands, axis=0)
        else:
            return np.stack(stack_bands, axis=0)


    @check_band_names
    def get_coordinates(
            self,
            band_name: str,
            shift_to_center: Optional[bool] = True
        ) -> Dict[str,np.array]:
        """
        Returns the coordinates in x and y dimension of a band.

        ATTENTION:
            GDAL provides pixel coordinates for the upper left corner
            of a pixel while other applications (xarray) require the coordinates
            of the pixel center

        :param band_name:
            name of the band for which to retrieve coordinates. If data was
            read from band stack the coordinates are valid for all bands
        :param shift_to_center:
            if True (default) return pixel center coordinates, if False uses
            the GDAL default and returns the upper left pixel coordinates.
        :return:
            dict with two entries "x" and "y" containing the numpy arrays
            of coordinates
        """

        coords = {}
        ny, nx = self.get_band(band_name).shape
        transform = self.get_meta(band_name)['transform']

        shift = 0.
        if shift_to_center:
            shift = 0.5

        # taken from xarray.backends.rasterio.py#323
        x, _ = transform * (np.arange(nx) + shift, np.zeros(nx) + shift)
        _, y = transform * (np.zeros(ny) + shift, np.arange(ny) + shift)
        
        coords['y'] = y
        coords['x'] = x

        return coords


    def _flatten_coordinates(
        self,
        band_name: str,
        pixel_coordinates_centered: bool,
        
        ) -> Dict[str, np.array]:
        """
        Flattens band coordinates. To be used when converting an array to
        ``geopandas.GeoDataFrame``.

        :param band_name:
            name of the band to convert
        :param pixel_coordinates_centered:
            if False the GDAL default is used an the upper left pixel corner is returned
            for point-like objects. If True the pixel center is used instead.
        :return:
            dict of ``numpy.ndarray`` containing the x and y coordinates
        """
             
        # get coordinates
        coords = self.get_coordinates(
            band_name=band_name,
            shift_to_center=pixel_coordinates_centered
        )
        # get band shape in terms of rows and columns
        band_shape = self.get_band_shape(band_name)
    
        # flatten x coordinates along the y-axis
        flat_x_coords = np.repeat(coords['x'], band_shape.nrows)
        # flatten y coordinates along the x-axis
        flat_y_coords = np.tile(coords['y'], band_shape.ncols)
    
        out_coords = {
            'x': flat_x_coords,
            'y': flat_y_coords
        }
    
        return out_coords


    @check_band_names
    def get_blackfill(
            self,
            band_name: str,
            blackfill_value: Optional[Union[int,float]] = 0
        ) -> np.array:
        """
        Returns a boolean ``numpy.array`` indicating those pixels that are
        black-filled in a user-selected band.

        :param band_name:
            name of the band
        :param blackfill_value:
            pixel value indicating black fill. Zero by default.
        :return:
            boolean black-fill mask as ``numpy.array`` (two dimensional)
        """

        return self.get_band(band_name) == blackfill_value


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


    def get_bandaliases(
            self
        ) -> Dict[str,str]:
        """
        Returns band aliases if any.
        """

        return self._band_aliases


    @check_band_names
    def set_bandaliases(
            self,
            band_names: List[str],
            band_aliases: List[str]
        ) -> None:
        """
        Sets alias band names. Existing aliases might be overwritten!

        :param band_names:
            selected band names
        :param band_aliases:
            band aliases to set. Must equal the number of band_names and applies
            in the same order as band_names
        """

        # create dict of names and aliases
        try:
            alias_dict = dict(zip(band_names, band_aliases))
        except Exception as e:
            raise Exception from e

        # check if self already has aliases
        if self._has_bandaliases:
            # update if aliases already exist
            self._band_aliases.update(alias_dict)
        else:
            self._band_aliases = alias_dict
            self._has_bandaliases = True


    @check_band_names
    def get_band_shape(
            self,
            band_name: str
        ) -> NamedTuple:
        """
        Returns the shape of a band in terms of rows and columns

        :param band_name:
            name of the band
        :return:
            tuple with number of rows and columns
        """

        Dimensions = namedtuple('dimensions', 'nrows ncols')
        return Dimensions(*self.get_band(band_name).shape)


    @check_band_names
    def get_band_nodata(
            self,
            band_name: str
        ) -> Number:
        """
        Returns the nodata value of a band extracted from the band's
        attributes

        :param band_name:
            name of the band
        :return:
            no data value of the band
        """
        try:
            return self.get_attrs(band_name)['nodatavals'][0]
        except Exception as e:
            raise Exception(
                f'Could not infer no data value for band {band_name}: {e}'
            )


    @check_band_names
    def get_meta(
            self,
            band_name: Optional[str] = None
        ) -> Dict[str, Any]:
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


    @check_band_names
    def get_attrs(
            self,
            band_name: Optional[str] = None
        ) -> Dict[str, Any]:
        """
        Returns the image attributes retrieved from GDAL datasets
        for a single or all bands.

        :param band_name:
            optional band name for retrieving image atrributes
        :return:
            meta dict with image attributes
        """

        # if a single band is selected, return the corresponding subset;
        # otherwise, return everything in 'attrs'
        if band_name is not None:
            band_idx = [idx for idx, name in enumerate(self.get_bandnames()) \
                        if name == band_name][0] 
            band_attrs = dict.fromkeys(self.data['attrs'].keys())
            for attr in band_attrs:
                if isinstance(self.data['attrs'][attr], tuple):
                    band_attrs[attr] = tuple([self.data['attrs'][attr][band_idx]])
                else:
                    band_attrs[attr] = self.data['attrs'][attr]
            return band_attrs
        else:
            return self.data['attrs']


    @check_band_names
    def get_spatial_resolution(
            self,
            band_name: Optional[str] = None
        ) -> Union[Dict[str,NamedTuple],NamedTuple]:
        """
        Returns the spatial resolution in x and y direction of all or
        a selected band

        :param band_name:
            band name for retrieving spatial resolution
            of a specific band
        :return:
            spatial resolution in units of the image coordinate system
            in x and y direction
        """
        
        transform = {}
        meta = self.get_meta(band_name=band_name)

        if not self.from_bandstack():
            if band_name is not None:
                transform = meta['transform']
            else:
                for band in self.get_bandnames():
                    transform[band] = meta[band]['transform']
        else:
            transform = meta['transform']

        # multiple bands
        if isinstance(transform, dict):
            res = {}
            for band in self.get_bandnames():
                Spatial_Resolution = namedtuple('Spatial_Resolution', 'x y')
                res[band] = Spatial_Resolution(
                    transform[band][0],
                    transform[band][4]
                )

        # single band direct access to Affine object
        else:
            Spatial_Resolution = namedtuple('Spatial_Resolution', 'x y')
            res = Spatial_Resolution(transform[0], transform[4])

        return res


    @check_band_names
    def get_bounds(
            self,
            band_name: Optional[str] = None,
            return_as_polygon: Optional[bool] = True
        ) -> Union[Union[BoundingBox,Polygon], Dict[str,Union[BoundingBox,Polygon]]]:
        """
        Returns the bounds (bounding box) of a band

        :param band_name:
            optional band name for retrieving bounds
        :param return_as_polygon:
            if True returns a ``shapely`` polygon, if False a ``rasterio``
            bounding box object
        :return:
            bounds of the band in the coordinate system of the dataset
        """

        bounds = self.data['bounds']

        # check if the file is from band stack or if bounds are the same for each band
        if not self.from_bandstack() and band_name is not None:
            try:
                bounds = bounds[band_name]
            except Exception:
                raise BandNotFoundError(
                    f'Could not find "bounds" in data bounds dict'
                )

        # return bounding box or polygon
        if return_as_polygon:
            if isinstance(bounds, dict):
                res = {}
                for band in self.get_bandnames():
                    res[band] = box(*bounds[band])
            else:
                res = box(*bounds)
            return res

        return bounds


    @check_band_names
    def get_epsg(
            self,
            band_name: Optional[str] = None
        ) -> Union[CRS,Dict[str,CRS]]:
        """
        Returns the EPSG code of all or a selected band

        :param band_name:
            optional band name for which to retrieve the EPSG code
        :return:
            EPSG code of the bands as ``rasterio.crs.CRS``
        """

        meta = self.get_meta(band_name=band_name)

        if self._from_bandstack:
            return meta['crs']
        else:
            if band_name is not None:
                return meta['crs']
            else:
                res = {}
                for band in self.get_bandnames():
                    res[band] = meta[band]['crs']
                return res


    @check_band_names
    def get_band_coordinates(
            self,
            band_name: str,
            shift_to_center: Optional[bool] = True
        ) -> Dict[str,np.array]:
        """
        Returns the coordinates in x and y dimension of a band.

        ATTENTION:
            GDAL provides pixel coordinates for the upper left corner
            of a pixel while other applications (xarray) require the coordinates
            of the pixel center

        :param band_name:
            name of the band for which to retrieve coordinates. If data was
            read from band stack the coordinates are valid for all bands
        :param shift_to_center:
            if True (default) return pixel center coordinates, if False uses
            the GDAL default and returns the upper left pixel coordinates.
        :return:
            dict with two entries "x" and "y" containing the numpy arrays
            of coordinates
        """

        coords = {}
        nx, ny = self.get_band(band_name).shape
        transform = self.get_meta(band_name)['transform']

        shift = 0.
        if shift_to_center:
            shift = 0.5

        # taken from xarray.backends.rasterio.py#323
        x, _ = transform * (np.arange(nx) + shift, np.zeros(nx) + shift)
        _, y = transform * (np.zeros(ny) + shift, np.arange(ny) + shift)
        
        coords['y'] = y
        coords['x'] = x

        return coords

    @check_metadata
    def _set_image_metadata(
            self,
            metadata_key: str,
            metadata_values: dict,
            band_name: Optional[str],
        ) -> None:
        """
        Back-end method called by set_meta, set_bounds and set_attr.

        :param metadata_key:
            must be one of 'meta', 'attrs', 'bounds'
        :param metadata_values:
            entries to set. Gets validated by the method's decorator
        :param band_name:
            name of the band for which to set the entries. Must be
            provided if the current instance does not fulfill the bandstack
            criteria.
        """

        # check if metadata key entry is already populated
        if metadata_key not in self.data.keys():
            self.data[metadata_key] = {}

        # check if the data is band stack
        if self._from_bandstack:
            self.data[metadata_key] = metadata_values
        else:
            if band_name is None:
                raise ValueError(
                    'Band name must be provided when not from bandstack'
                )
            if self.data[metadata_key] is None:
                self.data[metadata_key] = {}
            self.data[metadata_key][band_name] = metadata_values


    def set_meta(
            self,
            meta: dict,
            band_name: Optional[str] = None
        ) -> None:
        """
        Adds image metadata to the current object. Image metadata is an essential
        pre-requisite for writing image data to raster files.

        ATTENTION:
            Overwrites image metadata if already existing!

        :param meta:
            image metadata dict
        :param band_name:
            name of the band for which meta is added. If the current object
            is not a bandstack, specifying a band name is mandatory!
        """

        self._set_image_metadata(
            metadata_key='meta',
            metadata_values=meta,
            band_name=band_name
        )
        

    def set_attrs(
            self,
            attr: Dict[str, Any],
            band_name: Optional[str] = None
        ) -> None:
        """
        Adds image attributes to the current object.
        
        ATTENTION:
            Overwrites image attributes if already existing!

        :param meta:
            image attrib dict
        :param band_name:
            name of the band for which attributes are added. If the
            current object is not a bandstack, specifying a band name
            is mandatory!
        """

        self._set_image_metadata(
            metadata_key='attrs',
            metadata_values=attr,
            band_name=band_name
        )
        

    def set_bounds(
            self,
            bounds,
            band_name: Optional[str] = None
        ) -> None:
        """
        Adds image bounds to the current object. Image bounds are required for
        plotting.

        ATTENTION:
            Overwrites image bounds if already existing!

        :param meta:
            image metadata dict
        :param band_name:
            name of the band for which meta is added. If the current object
            is not a bandstack, specifying a band name is mandatory!
        """

        self._set_image_metadata(
            metadata_key='bounds',
            metadata_values=bounds,
            band_name=band_name
        )


    def reset_bandnames(
            self,
            new_bandnames: List[str]
        ) -> None:
        """
        Sets new band names. The length of the passed names must
        match the number of bands available. The replacement works in
        the order the new names are passed.

        :param new_bandnames:
            list of new band names. Must have the same length as the
            currently available bands.
        """

        # get old band names
        old_bandnames = self.get_bandnames()
        if len(old_bandnames) != len(new_bandnames):
            raise InputError(
                f'The number of new band names ({len(new_bandnames)}) ' \
                f'does not match the number of old band names (' \
                f'{len(old_bandnames)})'
            )

        # replace the band names element by element
        for old_bandname, new_bandname in list(zip(old_bandnames, new_bandnames)):
            try:
                self.data[new_bandname] = self.data.pop(old_bandname)
            except Exception as e:
                raise Exception from e

        # TODO: update band names in image meta!!


    @check_band_names
    def drop_band(
            self,
            band_name: str
        ) -> None:
        """
        Drops a selected band from the data dict. Also erases any
        references to the band in the metadata.

        :param band_name:
            name of the band to drop
        """

        try:
            band_idx = [x for x, name in enumerate(self.get_bandnames()) if name == band_name][0]
            del self.data[band_name]
        except Exception as e:
            raise Exception from e

        # handle band metadata
        if not self.check_is_bandstack():
            try:
                del self.data['meta'][band_name]
                del self.data['bounds'][band_name]
                del self.data['attrs'][band_name]
            except Exception as e:
                raise Exception from e
        else:
            # if the data is band-stacked, delete the band entries from the attributes
            for attr in self.data['attrs']:
                if isinstance(self.data['attrs'][attr], tuple):
                    update_list = list(self.data['attrs'][attr])
                    del update_list[band_idx]
                    self.data['attrs'][attr] = update_list


    def reproject_bands(
            self,
            target_crs: Union[int, CRS],
            blackfill_value: Optional[Union[int,float]] = 0,
            resampling_method: Optional[int] = Resampling.nearest,
            num_threads: Optional[int] = 1,
            dst_transform: Optional[Affine] = None
        ) -> None:
        """
        Reprojects all available bands from their source spatial reference
        system into another one.

        ATTENTION:
            The original band data is overwritten by this method!
        ATTENTION:
            When possible use ``dst_transform`` to align the
            re-projected raster with another raster that is already in the target
            CRS and has the same spatial extent as the band you are working on.

        :param target_crs:
            EPSG code denoting the target coordinate system
        :param blackfill_value:
            value indicating black-fill (aka no-data). No-Data pixels are
            not used for interpolation.
        :param resampling_method:
            resampling method used by rasterio. Default is nearest neighbor
            resampling which is recommended as long as the pixel size remains
            the same (e.g., from one UTM zone into another).
        :param num_threads:
            number of threads to use. The default is 1.
        :param dst_transform:
            per default AgriSatPy simply projects a raster from one CRS into
            another one. This might distort, however, the resulting image and
            change the pixel size. If you need the result to be aligned with
            another raster provide dst_transform as a target ("snap") transformation
        """

        # get band names
        band_names = self.get_bandnames()
        
        # loop over bands and reproject them
        for band_name in band_names:

            # get band bounds and Affine transformation
            src_bounds = self.get_bounds(band_name, return_as_polygon=False)
            src_crs = self.get_epsg(band_name)
            src_meta = deepcopy(self.get_meta(band_name))
            src_affine = src_meta['transform']

            resampling_options = {
                'src_crs': src_crs,
                'src_transform': src_affine,
                'dst_crs': target_crs,
                'src_nodata': blackfill_value,
                'resampling': resampling_method,
                'num_threads': num_threads,
                'dst_transform': dst_transform
            }

            # reproject raster data
            try:
                out_data, out_transform = reproject_raster_dataset(
                    raster=self.get_band(band_name),
                    **resampling_options
                )
            except Exception as e:
                raise ReprojectionError(f'Could not re-project band {band_name}: {e}')

            # reproject bounds
            try:
                out_bounds = transform_bounds(
                    src_crs=src_crs,
                    dst_crs=target_crs,
                    left=src_bounds.left,
                    bottom=src_bounds.bottom,
                    top=src_bounds.top,
                    right=src_bounds.right
                )
            except Exception as e:
                raise ReprojectionError(f'Could not re-project bounds of {band_name}: {e}')

            # overwrite band data
            self.data[band_name] = out_data[0,:,:]

            # adopt meta-entries (CRS, coordinates, etc.)
            src_meta.update(
                {
                    'crs' : rio.crs.CRS.from_epsg(target_crs),
                    'transform': out_transform,
                    'height': out_data.shape[1],
                    'width': out_data.shape[2]
                }
            )
            # overwrite meta and bounds
            if not self.from_bandstack():
                self.data['meta'][band_name] = src_meta
                self.data['bounds'][band_name] = out_bounds
            else:
                self.data['meta'] = src_meta
                self.data['bounds'] = out_bounds


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
    def _masked_array_to_nan(
            band_data: np.array
        ) -> np.array:
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


    @check_band_names
    def plot_band(
            self,
            band_name: str,
            colormap: Optional[str] = 'gray',
            discrete_values: Optional[bool] = False,
            user_defined_colors: Optional[ListedColormap] = None,
            user_defined_ticks: Optional[List[Union[str,int,float]]] = None,
            colorbar_label: Optional[str] = None,
            vmin: Optional[Union[int, float]] = None,
            vmax: Optional[Union[int, float]] = None
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
        :param colorbar_label:
            optional text label to set to the colorbar.
        :param vmin:
            lower value to use for ``plt.imshow()``. If None it is set to the
            lower 5% percentile of the data to plot.
        :param vmin:
            upper value to use for ``plt.imshow()``. If None it is set to the
            upper 95% percentile of the data to plot.
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
                # check if band data is still int16, if yes convert it to float
                if band_data.dtype == 'uint16':
                    band_data = band_data.astype(float)
                band_data_list.append(band_data)
            # add transparency layer (floats only)
            if band_data.dtype == float:
                band_data_list.append(np.zeros_like(band_data_list[0]))
            band_data = np.dstack(band_data_list)

            # use 'green' henceforth to extract the corresponding meta-data
            band_name = 'green'
            # no colormap required
            colormap = None

        # convert masked array to NaN (if applicable at all)
        band_data = self._masked_array_to_nan(band_data=band_data)

        # adjust transparency in case of RGBA arrays
        if len(band_data.shape) == 3 and band_data.dtype == float:
            tmp = deepcopy(band_data[:,:,0])
            # replace zero (blackfill) with nan 
            tmp[tmp <= 0.] = np.nan
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
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                interpolation='none'  # important, otherwise img will have speckle!
            )

        else:

            # clip data for displaying to central 90%, i.e., discard upper and
            # RGB and NIR plot work different because they contain multiple bands
            if (rgb_plot or nir_plot) and len(band_data.shape) == 3:
                vmin = np.nanquantile(band_data[:,:,0:3], 0.05)
                vmax = np.nanquantile(band_data[:,:,0:3], 0.95)
            else:
                if vmin is None:
                    vmin = np.nanquantile(band_data, 0.05)
                if vmax is None:
                    vmax = np.nanquantile(band_data, 0.95)

            # actual displaying of the band data
            img = ax.imshow(
                band_data,
                vmin=vmin,
                vmax=vmax,
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                cmap=cmap
            )

        # add colorbar (does not apply in RGB case)
        if colormap is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if discrete_values:
                cb = fig.colorbar(
                    img,
                    cax=cax,
                    orientation='vertical',
                    ticks=unique_values,
                    extend='max'
                )
            else:
                cb = fig.colorbar(
                    img,
                    cax=cax,
                    orientation='vertical'
                )
            # overwrite ticker if user defined ticks provided
            if user_defined_ticks is not None:
                # TODO: there seems to be one tick missing (?)
                cb.ax.locator_params(nbins=len(user_defined_ticks))
                cb.set_ticklabels(user_defined_ticks)
            # add colorbar label text if provided
            if colorbar_label is not None:
                cb.set_label(
                    colorbar_label,
                    rotation=270,
                    fontsize=14,
                    labelpad=20,
                    y=0.5
                )

        # set plot title (name of the band if not RGB or NIR plot)
        if colormap is None:
            if rgb_plot:
                ax.title.set_text('True Color Image')
            elif nir_plot:
                ax.title.set_text('False Color Nir-Infrared Image')
        else:
            ax.title.set_text(f'Band: {band_name.upper()}')

        # add axes labels and format ticker
        ax.set_xlabel(f'X [m] (EPSG:{epsg})', fontsize=12)
        ax.xaxis.set_ticks(np.arange(bounds.left, bounds.right, x_interval))
        ax.set_ylabel(f'Y [m] (EPSG:{epsg})', fontsize=12)
        ax.yaxis.set_ticks(np.arange(bounds.bottom, bounds.top, y_interval))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

        return fig


    def calc_si(
            self,
            si: str
        ) -> None:
        """
        Calculates a spectral index (SI) implemented in `agrisatpy.analysis.spectral_indices`
        and adds it as new band.

        To get a full list of SIs currently available, type

        >>> from agrisatpy.analysis.spectral_indices import SpectralIndices
        >>> SpectralIndices.get_si_list()

        :param si:
            name of the spectral index to calculate (e.g., NDVI). See ``
        """
        try:
            si_obj = SpectralIndices(reader=self)
            si_data = si_obj.calc_si(si)

            # we need a snap_band to comply with AgriSatPy's snap raster policy. Since
            # the spectral index is calculated from the current handler instance, we can
            # simply look for a band with the same shape as the returned SI raster
            snap_band = None
            for band_name in self.get_bandnames():
                band_dim = self.get_band_shape(band_name)
                if (
                    band_dim.nrows == si_data.shape[0] and
                    band_dim.ncols == si_data.shape[1]
                ):
                    snap_band = band_name
                    break

            # add the SI as new band (will fail if the band already exists)
            self.add_band(
                band_name=si,
                band_data=si_data,
                snap_band=snap_band
            )
        except Exception as e:
            raise Exception(f'Could not calculate spectral index "{si}": {e}')


    def resample(
            self,
            target_resolution: Union[int,float],
            resampling_method: Optional[int] = cv2.INTER_CUBIC,
            pixel_division: Optional[bool] = False,
            band_selection: Optional[List[str]] = [],
            bands_to_exclude: Optional[List[str]] = [],
            blackfill_value: Optional[Union[int,float]] = 0
        ) -> None:
        """
        resamples band data on the fly if required into a user-definded spatial
        resolution. The resampling algorithm used is `~cv2.resize` and allows the
        following options:

        - INTER_NEAREST - a nearest-neighbor interpolation
        - INTER_LINEAR - a bilinear interpolation (used by default)
        - INTER_AREA - resampling using pixel area relation.
        - INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        - INTER_LANCZOS4 -  a Lanczos interpolation over 8x8 pixel neighborhood

        ATTENTION:
            The method overwrites the original band data when resampling
            is required!

        ATTENTION: If using one of cv2 methods it is crucial to provide a
        blackfill (no-data) value. Blackfill is where the satellite image has no
        data.

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
        :param blackfill_value:
            value denoting blackfill (no-data). Pixels with that value are excluded
            from the resampling process. Can be ignored if ``pixel_division=True``
            because pixel division only divides pixels into smaller ones that
            align with the original pixels.
        """

        # loop over bands and resample those bands not in the desired
        # spatial resolution

        if len(band_selection) == 0:
            band_selection = self.get_bandnames()
        # check if bands are to exclude
        if len(bands_to_exclude) > 0:
            set_to_exclude = set(bands_to_exclude)
            band_selection = [x for x in band_selection if x not in set_to_exclude]

        snap_band_available = False
        for idx, band in enumerate(band_selection):

            pixres = self.get_spatial_resolution(band).x

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

        for idx, band in enumerate(band_selection):

            # check spatial resolution
            meta = self.get_meta(band)
            pixres = self.get_spatial_resolution(band).x
            if pixres == target_resolution:
                continue

            # get image boundaries
            bounds = self.get_bounds(
                band_name=band,
                return_as_polygon=False
            )

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
            scaling_factor = int(pixres / target_resolution)
            if pixel_division:
                # determine scaling factor as ratio between current and target spatial resolution
                res = upsample_array(
                    in_array=band_data,
                    scaling_factor=scaling_factor
                )
            else:
                # we have to take care about no-data pixels
                valid_pixels = count_valid(
                    in_array=band_data,
                    no_data_value=blackfill_value
                )
                all_pixels = band_data.shape[0] * band_data.shape[1]
                # if all pixels are valid, then we can directly proceed to the resampling
                if valid_pixels == all_pixels:
                    try:
                        res = cv2.resize(
                            band_data,
                            dsize=dim_resampled,
                            interpolation=resampling_method
                        )
                    except Exception as e:
                        raise ResamplingFailedError(e)
                else:
                    # blackfill pixel should be set to NaN before resampling
                    type_casting = False
                    if band_data.dtype == 'uint8' or band_data.dtype == 'uint16':
                        tmp = deepcopy(band_data).astype(float)
                        type_casting = True
                    else:
                        tmp = deepcopy(band_data)
                    tmp[tmp == blackfill_value] = np.nan
                    # resample data
                    try:
                        res = cv2.resize(
                            tmp,
                            dsize=dim_resampled,
                            interpolation=resampling_method
                        )
                    except Exception as e:
                        raise ResamplingFailedError(e)

                    # in addition, run pixel division since there will be too many NaN pixels
                    # when using only res from cv2 resize as it sets pixels without full
                    # spatial context to NaN
                    try:
                        res_pixel_div = upsample_array(
                            in_array=band_data,
                            scaling_factor=scaling_factor
                        )
                    except Exception as e:
                        raise ResamplingFailedError(e)

                    # replace NaNs with values from pixel division; thus we will get all
                    # pixel values and the correct blackfill; when working on AOIs this might
                    # fail because of shape mismatches; in this case keep the cv2 output, which means
                    # loosing a few pixels but AOIs can usually be filled with data from other
                    # scenes
                    if res.shape == res_pixel_div.shape:
                        res[np.isnan(res)] = res_pixel_div[np.isnan(res)]
                    else:
                        res[np.isnan(res)] = blackfill_value

                    # cast back to original datatype if required
                    if type_casting:
                        res = res.astype(band_data.dtype)

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
                meta_resampled['width'] = self.data[band].shape[1]
                meta_resampled['height'] = self.data[band].shape[0]
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
            keep_mask_values: Optional[bool] = False,
            nodata_values: Optional[Union[Number,List[Number]]] = None
        ) -> None:
        """
        Allows to mask parts of an image (i.e., single bands) based
        on a mask band. The mask band must have the same spatial extent and
        resolution as the bands to mask (you might have to consider
        spatial resampling using the `resampling` method) and must
        be an entry in the data dict (using `add_band`) if it is not yet part
        of it.

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
        :param nodata_values:
            no data values to set masked pixels. Can be set for each band by specifying
            a list of nodata values or a single value for all bands. If None (default),
            the no-data value is inferred from the band attributes.
        """

        # get band to use for masking
        mask_band = self.get_band(band_name=name_mask_band)
        if isinstance(mask_band, np.ma.MaskedArray):
            mask_band = mask_band.data

        # convert the mask to a temporary binary mask
        tmp = np.zeros_like(mask_band)
        # set valid classes to 1, the other ones are zero
        if keep_mask_values:
            # drop all other values not in mask_values
            tmp[~np.isin(mask_band, mask_values)] = 1
        else:
            # drop all values in mask_values
            tmp[np.isin(mask_band, mask_values)] = 1

        # loop over bands specified and mask the invalid pixels
        for idx, band_to_mask in enumerate(bands_to_mask):
            if band_to_mask not in self.get_bandnames() and \
            band_to_mask not in self.get_bandaliases().values():
                raise BandNotFoundError(
                    f'{band_to_mask} not found in available bands'
                )
            # check alias
            if band_to_mask not in self.get_bandnames():
                band_to_mask = [k for (k,v) in self.get_bandaliases().items() \
                                if v == band_to_mask][0]

            # check nodata value; take from image attributes if not provided
            if nodata_values is None:
                nodata_value = self.get_band_nodata(band_to_mask)
            else:
                if isinstance(nodata_values, list):
                    try:
                        nodata_value = nodata_values[idx]
                    except IndexError as e:
                        raise Exception from e

            # set values nodata where tmp is zero
            if isinstance(self.data[band_to_mask], np.ma.MaskedArray):
                # combine the two masks (previous mask and additionally masked pixels)
                orig_mask = self.data[band_to_mask].mask
                for row in range(orig_mask.shape[0]):
                    for col in range(orig_mask.shape[1]):
                        # ignore pixels already masked
                        if orig_mask[row, col]:
                            continue
                        else:
                            orig_mask[row, col] = tmp[row,col] == 1
                self.data[band_to_mask] = np.ma.MaskedArray(
                    data=self.data[band_to_mask].data,
                    mask=orig_mask
                )
            else:
                self.data[band_to_mask][tmp == 1] = nodata_value


    # TODO: rename in_file_aoi to vector_features
    def read_from_bandstack(
            self,
            fname_bandstack: Path,
            in_file_aoi: Optional[Union[Path, gpd.GeoDataFrame]] = None,
            full_bounding_box_only: Optional[bool] = False,
            check_for_blackfill: Optional[bool] = True,
            blackfill_value: Optional[Union[int,float]] = None,
            band_selection: Optional[List[str]] = None
        ) -> None:
        """
        Reads raster bands from a single or multi-band, geo-referenced file. All
        raster formats supported by ``GDAL`` can be read, however, we recommend to
        use `geoTiff` or `JPEG2000`. The data can be read from a local file or
        from a network resource (e.g., cloud-optimized geoTiff).

        The bands are into a ``numpy.ndarray`` or ``numpy.ma.MaskedArray``.

        There are several options for reading:

            - By passing a list of band names (`band_selection`) it is possible
              to read a subset of bands, only
            - By passing a file with vector geometries (e.g., ESRI shapefile or
              any other file format understood by ``fiona``) it is possible to
              read a spatial subset, only
            - When reading a spatial subset there are two further options:
                  a) shrinking the subset to the bounding box of the vector features
                  b) shrinking the subset to the vector features, only. In this case,
                     the band data is read ``np.ma.MaskedArray``

        NOTE:
            When reading a spatial subset (`in_file_aoi` provided) the vector features
            must be a single geometry or a series of geometries of type ``Polygon`` or
            ``MultiPolygon``! For reading single pixel values from a raster resource
            refer to ``SatDataHandler.read_pixels()``.

        :param fname_bandstack:
            file-path or URI to the single or multi-band raster file to read.
        :param in_file_aoi:
            vector file (e.g., ESRI shapefile or geojson) or ``GeoDataFrame``.
            Can contain one to many features, which must of type ``Polygon`` or
            ``MultiPolygon``. The spatial reference system of the features can differ
            from those of the raster data as long as a transformation is defined for
            re-projecting the vector features into the spatial reference system of
            the raster data.
        :param full_bounding_box_only:
            if set to False, will only extract the data for those geometry/ies
            defined in in_file_aoi. If set to False, returns the data for the
            full extent (hull) of all features (geometries) in in_file_aoi.
        :param blackfill_value:
            value indicating black fill (=no data). If None it is inferred from the image
            metadata.
        :param check_for_blackfill:
            optional (and default) check if the image data read contains any data besides
            blackfill (nodata) values. Satellite data can contain reasonable amounts of
            blackfill due to data processing and orbit design. Checking for blackfill
            (and potentially discarding these scenes) can therefore save memory and
            processing requirements.
        :param band_selection:
            list of bands to read. Per default all raster bands available are read.

        Examples
        --------
        >>> from agrisatpy.io import SatDataHandler
        >>> handler = SatDataHandler()
        >>> handler.read_from_bandstack('example.tif', in_file_aoi='parcels_boundaries.shp', band_selection=['B1'])
        """

        # check vector features if provided
        masking = False
        if in_file_aoi is not None:

            masking = True
            gdf_aoi = check_aoi_geoms(
                in_dataset=in_file_aoi,
                fname_raster=fname_bandstack,
                full_bounding_box_only=full_bounding_box_only
            )
            # check for third dimension (has_z) and flatten it to 2d
            gdf_aoi.geometry = convert_3D_2D(gdf_aoi.geometry)

            # check geometry types of the input features
            allowed_geometry_types = ['Polygon', 'MultiPolygon']
            gdf_aoi = check_geometry_types(
                in_dataset=gdf_aoi,
                allowed_geometry_types=allowed_geometry_types
            )

        # check band selection
        check_band_selection = False
        if band_selection is not None:
            check_band_selection = True
    
        with rio.open(fname_bandstack, 'r') as src:
            # get geo-referencation information
            meta = src.meta
            # and bounds which are helpful for plotting
            bounds = src.bounds
            # parse image attributes
            attrs = get_raster_attributes(riods=src)

            # read relevant bands and store them in dict
            band_names = src.descriptions
            self.data = dict.fromkeys(band_names)
            band_selection_idx = []
            for idx, band_name in enumerate(band_names):

                # handle empty band_names
                if band_name is None:
                    band_name = f'B{idx+1}'

                # check band selection if required
                if check_band_selection:
                    if band_name not in band_selection:
                        self.data.pop(band_name)
                        continue
                band_selection_idx.append(idx)
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

        # meta and bounds are the same single entry for all bands
        # meta and bounds are saved as additional items of the dict
        meta.update(
            {'count': len(self.get_bandnames())}
        )
        self._from_bandstack = True
        self.set_meta(meta)
        self.set_bounds(bounds)

        if band_selection is not None:
            attrs_filtered = {}
            for key in attrs.keys():
                if isinstance(attrs[key], tuple):
                    # use band_selection_idx to filter selected bands
                    attrs_filtered[key] = tuple(
                        [x for ii, x in enumerate(attrs[key]) if ii in band_selection_idx]
                    )
                else:
                    attrs_filtered[key] = attrs[key]
            self.set_attrs(attrs_filtered)
        # no band selection, use all available bands
        else:
            self.set_attrs(attrs)

        # check for black-fill (i.e., if the data only contains nodata an error will be raised)
        if check_for_blackfill:
            if blackfill_value is None:
                blackfill_value = self.get_band_nodata(self.get_bandnames()[0])
            is_blackfilled = self.is_blackfilled(blackfill_value=blackfill_value)
            if is_blackfilled:
                raise BlackFillOnlyError(
                    f'Region read from {in_file_aoi} contains blackfill (nodata), only'
                )


    @classmethod
    def read_pixels(
            cls,
            point_features: Union[Path, gpd.GeoDataFrame],
            raster: Path,
            band_selection: Optional[List[str]] = None
        ) -> gpd.GeoDataFrame:
        """
        Extracts raster values from an **external** raster dataset at
        locations defined by one or many point-like geometry features
        read from a vector file (e.g., ESRI shapefile) or ``GeoDataFrame``.

        IMPORTANT:
            This methods does **not** return pixel values from the current
            ``SatDataHandler`` instance but from the provided raster dataset
            (``raster``). If you want to get pixel values from bands currently
            read in the current ``SatDataHandler`` instance use
            `~SatDataHandler().get_pixels()` instead!
        
        NOTE:
            A point is dimension-less, therefore, the raster grid cell (pixel)
            closest to the point is returned if the point lies within the raster.
            For points outside of the raster extent pixel values are returned as
            nodata values in all raster bands.

        :param point_features:
            vector file (e.g., ESRI shapefile or geojson) or ``GeoDataFrame``
            defining point locations for which to extract pixel values.
        :param raster:
            custom raster dataset understood by ``GDAL`` from which to extract
            pixel values at the provided point locations
        :param band_selection:
            list of bands to read. Per default all raster bands available are read.
        :return:
            ``GeoDataFrame`` containing the extracted raster values. The band values
            are appened as columns to the dataframe. Existing columns of the input
            `in_file_pixels` are preserved.
        """

        # check input point features
        gdf = check_aoi_geoms(
            in_dataset=point_features,
            fname_raster=raster,
            full_bounding_box_only=False
        )
        allowed_geometry_types = ['Point']
        gdf = check_geometry_types(
            in_dataset=gdf,
            allowed_geometry_types=allowed_geometry_types
        )

        # check band selection. If not provided, uses all available bands
        band_selection_idx = None
        band_names = list(rio.open(raster, 'r').descriptions)

        if band_selection is not None:
            band_selection_idx = [idx+1 for idx, val in enumerate(band_names) \
                                  if val in band_selection]
            if len(band_selection_idx) == 0:
                raise BandNotFoundError(
                    f'The band selection provided did not match any of ' \
                    f'the band names in {raster}: {band_names}'
                )
        else:
            band_selection = band_names
                
        # use rasterio.sample to extract the pixel values from the raster
        # to do so, we need a list of coordinate tuples
        coord_list = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]
        with rio.open(raster, 'r') as src:
            try:
                # yield all values from the generator
                def _sample(src, coord_list, band_selection_idx):
                    yield from src.sample(coord_list, band_selection_idx)
                pixel_samples =list(_sample(src, coord_list, band_selection_idx))
            except Exception as e:
                raise Exception(f'Extraction of pixels from raster failed: {e}')

        # append the extracted pixels to the exisiting geodataframe. We can do so, because
        # we have passed the pixels in the same order as they occur in the dataframe
        for idx, band in enumerate(band_selection):
            # we need the idx-element of each pixel in the list of pixel samples
            band_list = [x[idx] for x in pixel_samples]
            gdf[band] = band_list

        return gdf


    def get_pixels(
            self,
            point_features: Union[Path, gpd.GeoDataFrame],
            band_selection: Optional[List[str]] = None,
            shift_to_center: Optional[bool] = True
        ):
        """
        Get pixel values from all or a subset of bands in the current
        ``SatDataHandler`` instance. Pixels are specified by point-like features
        provided in a vector file (e.g., ESRI shapefile) or by passing a
        ``GeoDataFrame``.

        The method supports extracting pixel data also from ``SatDataHandler``
        instances with bands differing in spatial resolution, spatial extent
        and coordinate system. Pixels outside of the current band bounds are
        ignored and not returned!

        NOTE:
            In contrast to `~SatDataHandler.read_pixels` (which is a class method)
            this is a(n instance) method to get pixel values from bands already
            **read** into a ``SatDataHandler`` instance.

        :param point_features:
            vector file (e.g., ESRI shapefile or geojson) or ``GeoDataFrame``
            defining point locations for which to extract pixel values
        :param band_selection:
            list of bands to read. Per default all raster bands available are read.
        :param shift_to_center:
            when True (default) return pixel center coordinates, if False uses
            the GDAL default and returns the upper left pixel coordinates.
        :return:
            ``GeoDataFrame`` containing the extracted raster values. The band values
            are appened as columns to the dataframe. Existing columns of the input
            `in_file_pixels` are preserved.
        """

        if band_selection is None:
            band_selection = self.get_bandnames()

        # define helper function for getting the closest array index for a coordinate
        # map the coordinates to array indices
        def _find_nearest_array_index(array, value):
            return np.abs(array - value).argmin()

        # To support extracting pixel values also from raster bands with different
        # resolution, projection and extent values are extracted for each band
        # separately
        band_gdfs = []
        for idx, band_name in enumerate(band_selection):

            # check input point features (and reproject them if necessary)
            gdf = check_aoi_geoms(
                in_dataset=point_features,
                raster_crs=self.get_epsg(band_name),
                full_bounding_box_only=False
            )

            # drop points outside of the band's bounding box
            band_bbox = self.get_bounds(
                band_name=band_name,
                return_as_polygon=False
            )
            gdf = gdf.cx[band_bbox.left:band_bbox.right, band_bbox.bottom:band_bbox.top]
            # the check for the correct geometry type is only required once
            # because the features do not change between the loop iterations
            if idx == 0:
                allowed_geometry_types = ['Point']
                gdf = check_geometry_types(
                    in_dataset=gdf,
                    allowed_geometry_types=allowed_geometry_types
                )

            # calculate the x and y array indices required to extract the pixel values
            gdf['x'] = gdf.geometry.x
            gdf['y'] = gdf.geometry.y

            # get band coordinates
            coords = self.get_band_coordinates(
                band_name=band_name,
                shift_to_center=shift_to_center
            )

            # get column (x) indices
            gdf['col'] = gdf['x'].apply(
                lambda x, coords=coords, find_nearest_array_index=_find_nearest_array_index:
                find_nearest_array_index(coords['x'], x)
            )
            # get row (y) indices
            gdf['row'] = gdf['y'].apply(
                lambda y, coords=coords, find_nearest_array_index=_find_nearest_array_index:
                find_nearest_array_index(coords['y'], y)
            )

            # add column to store band values
            gdf[band_name] = np.empty(gdf.shape[0])

            # loop over sample points and add them as new entries to the dataframe
            for _, record in gdf.iterrows():
                # get array value for the current column and row, continue on out-of-bounds error
                try:
                    array_value = self.get_band(band_name)[record.row, record.col]
                except IndexError:
                    continue
                gdf.loc[
                    (gdf.row == record.row) & (gdf.col == record.col),
                    band_name
                ] = array_value

            # append GeoDataFrame to List, if the iteration is larger zero drop the
            # geometry column to avoid an error later on
            if idx > 0:
                gdf.drop('geometry', axis=1, inplace=True)
            band_gdfs.append(gdf)

        # concat the single GeoDataFrames with the band data
        gdf = pd.concat(band_gdfs, axis=1)

        # clean the dataframe and remove duplicate column names after merging
        # to avoid (large) redundancies
        gdf = gdf.loc[:,~gdf.columns.duplicated()]

        return gdf


    def write_bands(
            self,
            out_file: Path,
            band_names: Optional[List[str]] = [],
            use_band_aliases: Optional[bool] = False
        ) -> None:
        """
        Writes one or multiple bands to a raster file using rasterio. By
        default a geoTiff is written since rasterio recommends this option over
        other geospatial image formats such as JPEG2000.

        ATTENTION: The method can only write bands to file that have the same
        spatial resolution and extent. If that's not the case you eiher have to
        resample the data first using the ``resample`` method or write only those
        bands that fullfil the aforementioned criteria.

        ATTENTION: If the bands do not have the same datatype they will be all set
        to ``numpy.float64``.

        :param out_file:
            file-path where to save the raster to. The file-ending will determine
            the type of raster generated; we recommend to use geoTiff (*.tif)
        :param band_names:
            list of bands to export (optional). If empty all bands available are
            exported to raster.
        :param use_band_aliases:
            use band aliases (if available) instead of actual band names for
            output band names
        """

        # check output file naming and driver
        try:
            driver = driver_from_extension(out_file)
        except Exception as e:
            raise ValueError(
                f'Could not determine GDAL driver for {out_file.name}: {e}'
            )

        # check band_selection, if not provided use all available bands
        if len(band_names) > 0:
            # check if band selection is valid
            if set(band_names).issubset(self.get_bandnames()):
                band_selection = band_names
            elif set(band_names).issubset(self.get_bandaliases().values()):
                band_selection = [k for (k,v) in self.get_bandaliases().items() if v in band_names]
        else:
            band_selection = self.get_bandnames()

        # check if band aliases shall be used
        if use_band_aliases:
            if self._has_bandaliases:
                band_selection = [k for (k,v) in self.get_bandaliases().items() if k in band_selection]

        if len(band_selection) == 0:
            raise ValueError('No band selected for writing to raster file')

        # check meta and update it with the selected driver for writing the result
        meta = deepcopy(self.get_meta(band_selection[0]))

        # check if all bands have the same shape, the first band determines the
        # shape all other bands have to follow
        first_shape = self.get_band(band_name=band_selection[0]).shape

        if len(band_selection) > 1:
            for band_name in band_selection[1:]:
                next_shape = self.get_band(band_name).shape
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

        # TODO: write attributes if any

        # open the result dataset and try to write the bands
        with rio.open(out_file, 'w+', **meta) as dst:

            for idx, band_name in enumerate(band_selection):

                # check with band name to set
                dst.set_band_description(idx+1, band_name)

                # write band data
                band_data = self.get_band(band_name).astype(dtype)
                band_data = self._masked_array_to_nan(band_data)
                dst.write(band_data, idx+1)


    def check_is_bandstack(
            self,
            band_selection: Optional[List[str]] = None
        ) -> bool:
        """
        Helper function that checks if a SatDataHandler object fulfills
        the ``from_bandstack()`` criteria. Optionally, the check can be carried
        out for a selected subset of bands.

        These criteria are:
            - all bands have the same CRS
            - all bands have the same x and y dimension (number of rows and columns)

        :param band_selection:
            if not None, checks only a list of selected bands. By default,
            all bands of the current object are checked.
        :return:
            True if the current object fulfills the criteria else False.
        """

        # loop over all bands and check the band CRS as well as their dims
        crs_list = []
        xdim_list = []
        ydim_list = []

        if band_selection is None:
            band_selection = self.get_bandnames()
        else:
            if not all(elem in self.get_bandnames() for elem in band_selection):
                raise BandNotFoundError(f'Invalid selection of bands')

        for band_name in band_selection:
            crs_list.append(self.get_epsg(band_name))
            xdim_list.append(self.get_band_shape(band_name).ncols)
            ydim_list.append(self.get_band_shape(band_name).nrows)

        return len(set(crs_list)) == 1 and len(set(xdim_list)) == 1 and len(set(ydim_list)) == 1


    def _set_bandstack(self):
        """
        Reorganizes the ``meta`` and ``bounds`` entries from band-wise organization
        to instance-wide coverage in a dict-like structure with a single entry of ``meta``
        and ``bounds`` for all bands.
        
        The opposite functionality is implemented in `~SatDataHandler()._unset_bandstack()`
        """

        entries = ['meta', 'bounds']
        band_names = self.get_bandnames()

        for entry in entries:
            src = deepcopy(self.data[entry])

            # replace the band-wise dict structure if the data from the first band
            try:
                band_entry = src[band_names[0]]
            except Exception as e:
                raise ValueError(f'{entry} is not organized by bands: {e}')

            self.data.update({entry: band_entry})

        self._from_bandstack = True


    def _unset_bandstack(self):
        """
        Reorganizes the ``meta`` and ``bounds`` entries from instance-wide coverage to
        band-wise organization in a dict-like structure with an entry of ``meta`` and
        ``bounds`` for each band.
        
        This allows to handle raster bands with different spatial resolution, extents
        and coordinate systems in one object.

        The opposite functionality is implemented in `~SatDataHandler()._set_bandstack()`
        """

        # re-populate the meta, bounds in self.data
        entries = ['meta', 'bounds']
        band_names = self.get_bandnames()
        
        for entry in entries:
            src = deepcopy(self.data[entry])
            # check if the entries are already organized by band names
            if isinstance(src, dict):
                if len([x for x in band_names if x in src.keys()]) > 0:
                    logger.warn(
                        'Band metadata seems to be already organized by bands'
                    )
                    return
            dst = {}
            for band_name in band_names:
                dst[band_name] = src
            self.data.update({entry: dst})

        self._from_bandstack = False


    def to_xarray(
            self,
            **kwargs
        ) -> DataArray:
        """
        Converts a ``SatDataHandler`` object to ``xarray`` in memory without
        having to dump anything to disk.

        ATTENTION:
            Works on bandstacked files only. I.e., all bands MUST
            have the same spatial extent and dimensions.

        :return:
            DataArray with spectral bands as dimensions and x and y
            coordinates
        """

        # check if data fulfills the bandstack criteria
        if not self.check_is_bandstack():
            raise ValueError(
                'Cannot convert SatDataHandler object to xarray when not bandstacked\n'
            )

        band_array_stack = {}
        for band_name in self.get_bandnames():
            band_data = deepcopy(self.get_band(band_name))
            # unfortunately, xarray does not support masked arrays, we have to convert
            # the image data to float (if not yet) and fill missing values with NaNs
            if isinstance(band_data, np.ma.core.masked_array):
                if band_data.dtype == 'uint8' or band_data.dtype == 'uint16':
                    band_data = band_data.astype(float)
                band_data = self._masked_array_to_nan(band_data)
                
            band_array_stack[band_name] = tuple([('y','x'), band_data])

        # get x, y coordinates and band names
        master_band = self.get_bandnames()[0]
        coords = self.get_coordinates(master_band)

        # get further attributes
        attrs = deepcopy(self.get_attrs())

        crs = self.get_epsg(master_band)

        # concat attributes across bands except for CRS and is_tiled
        attrs['crs'] = crs
        attrs['transform'] = tuple(self.get_meta(master_band)['transform'])
        attrs['time'] = self.scene_properties.get('acquisition_time')

        xds = xr.Dataset(
            band_array_stack,
            coords=coords,
            attrs=attrs,
            **kwargs
        )

        return xds


    @check_band_names
    def to_dataframe(
            self,
            band_names: Optional[List[str]] = None,
            pixel_coordinates_centered: Optional[bool] = False
        ) -> gpd.GeoDataFrame:
        """"
        Converts all or selected number of bands to a geopandas geodataframe.
        The resulting dataframe has as many rows as pixels in the current handler object
        and as many columns as bands plus a column denoting the pixel geometries.
        The spatial resolution of the bands selected must be the same as well as their
        spatial extent.

        NOTE:
            If the band data is a masked array, masked pixels are **not** written to the
            resulting ``GeoDataFrame``.

        The pixel geometries are returned either as shapely points or polygons.

        :param band_names:
            optional subset of bands for which to extract pixel values. Per
            default all bands are converted.
        :param pixel_coordinates_centered:
            if False the GDAL default is used an the upper left pixel corner is returned
            for point-like objects. If True the pixel center is used instead. This
            option is ignored if pixels are returned as polygons.
        :return:
            pandas or geopandas (geo) dataframe with pixel values and their coordinates
        """

        # use all bands if no selection is provided
        if band_names is None:
            band_names = self.get_bandnames()

        if band_names is None or len(band_names) == 0:
            raise BandNotFoundError(
                f'Invalid band selection or no bands available'
            )

        # two cases are possible: the data fulfills the bandstack criteria or not
        # if bandstacked, we can convert the data in a single rush
        if not self.check_is_bandstack(band_selection=band_names):
            raise InputError(
                'Bands selected for conversion to geodataframe must have same spatial extent'
            )
        
        # get coordinates of the first band in flattened format
        coords = self._flatten_coordinates(
            band_name=self.get_bandnames()[0],
            pixel_coordinates_centered=pixel_coordinates_centered
        )
        # get EPSG code
        epsg = self.get_epsg()

        # get image bands and reshape them from 3d to 2d
        stack_array = self.get_bands(band_names)
        new_shape = (stack_array.shape[0], stack_array.shape[1]*stack_array.shape[2])

        # if the band is a masked array, we need numpy.ma functions
        if isinstance(stack_array, np.ma.MaskedArray):
            flattened = np.ma.reshape(stack_array, new_shape, order='F')
            # save mask to array
            mask = flattened[0,:].mask
            # compress array (removes masked values) along the bands
            flattened = [f.compressed() for f in flattened]
            # mask band coordinates
            for coord in coords:
                coord_masked = np.ma.MaskedArray(data=coords[coord], mask=mask)
                coord_compressed = coord_masked.compressed()
                coords.update(
                    {
                        coord: coord_compressed
                    }
                )

        # otherwise we can use numpy ndarray's functions
        else:
            flattened = np.reshape(stack_array, new_shape, order='F')

        # convert the coordinates to shapely geometries
        coordinate_geoms = [Point(c[0], c[1]) for c in list(zip(coords['x'], coords['y']))]
        # call the GeoDataFrame constructor
        gdf = gpd.GeoDataFrame(geometry=coordinate_geoms, crs=epsg)

        # add band data
        gdf[band_names] = None
        if isinstance(flattened, list):
            for idx, band_name in enumerate(band_names):
                gdf[band_name] = flattened[idx]
        else:
            for idx, band_name in enumerate(band_names):
                gdf[band_name] = flattened[idx,:]

        return gdf
