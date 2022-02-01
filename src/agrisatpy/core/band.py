'''
A band is a two-dimensional array that can be located via a spatial coordinate system.
Each band thus has a name and an array of values, which are usually numeric.

AgriSatPy stores band data basically as ``numpy`` arrays. Masked arrays of the class
`~numpy.ma.MaskedArray` are also supported. For very large data sets that exceed the RAM of the
computer, ``zarr`` can be used.
'''

import cv2
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import rasterio as rio
import rasterio.mask
import xarray as xr
import zarr

from copy import deepcopy
from matplotlib.colors import ListedColormap
from matplotlib.figure import figaspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from rasterio import Affine
from rasterio import features
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from rasterio.enums import Resampling
from shapely.geometry import box
from shapely.geometry import Point
from shapely.geometry import Polygon
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from agrisatpy.core.utils.geometry import check_geometry_types
from agrisatpy.core.utils.geometry import convert_3D_2D
from agrisatpy.core.utils.raster import get_raster_attributes
from agrisatpy.utils.reprojection import check_aoi_geoms
from agrisatpy.utils.arrays import count_valid, upsample_array
from agrisatpy.utils.exceptions import BandNotFoundError, \
    DataExtractionError, ResamplingFailedError, ReprojectionError
from agrisatpy.utils.reprojection import reproject_raster_dataset
from abc import abstractstaticmethod


class GeoInfo(object):
    """
    Class for storing geo-localization information required to
    reference a raster band object in a spatial coordinate system.
    At its core this class contains all the attributes necessary to
    define a ``Affine`` transformation.

    :attrib epsg:
        EPSG code of the spatial reference system the raster data is projected
        to.
    :attrib ulx:
        upper left x coordinate of the raster band in the spatial reference system
        defined by the EPSG code. We assume ``GDAL`` defaults, therefore the coordinate
        should refer to the upper left *pixel* corner.
    :attrib uly:
        upper left y coordinate of the raster band in the spatial reference system
        defined by the EPSG code. We assume ``GDAL`` defaults, therefore the coordinate
        should refer to the upper left *pixel* corner.
    :attrib pixres_x:
        pixel size (aka spatial resolution) in x direction. The unit is defined by
        the spatial coordinate system given by the EPSG code.
    :attrib pixres_y:
        pixel size (aka spatial resolution) in y direction. The unit is defined by
        the spatial coordinate system given by the EPSG code.
    """

    def __init__(
            self,
            epsg: int,
            ulx: Union[int, float],
            uly: Union[int, float],
            pixres_x: Union[int, float],
            pixres_y: Union[int, float]
        ):
        """
        Class constructor to get a new ``GeoInfo`` instance.

        >>> geo_info = GeoInfo(4326, 11., 48., 0.02, 0.02)
        >>> affine = geo_info.as_affine()

        :param epsg:
            EPSG code identifying the spatial reference system (e.g., 4326 for
            WGS84).
        :param ulx:
            upper left x coordinate in units of the spatial reference system.
            Should refer to the upper left pixel corner.
        :param uly:
            upper left x coordinate in units of the spatial reference system.
            Should refer to the upper left pixel corner
        :param pixres_x:
            pixel grid cell size in x direction in units of the spatial reference
            system.
        :param pixres_y:
            pixel grid cell size in y direction in units of the spatial reference
            system.
        """
        # make sure the EPSG code is valid
        try:
            CRS.from_epsg(epsg)
        except Exception as e:
            raise ValueError(e)

        object.__setattr__(self, 'epsg', epsg)
        object.__setattr__(self, 'ulx', ulx)
        object.__setattr__(self, 'uly', uly)
        object.__setattr__(self, 'pixres_x', pixres_x)
        object.__setattr__(self, 'pixres_y', pixres_y)

    def __setattr__(self, *args, **kwargs):
        raise TypeError('GeoInfo object attributes are immutable')

    def __delattr__(self, *args, **kwargs):
        raise TypeError('GeoInfo object attributes are immutable')

    def __repr__(self) -> str:
        return str(self.__dict__)

    def as_affine(self) -> Affine:
        """
        Returns an ``rasterio.Affine`` compatible affine transformation

        :returns:
            ``GeoInfo`` instance as ``rasterio.Affine``
        """
        return Affine(
            a=self.pixres_x,
            b=0,
            c=self.ulx,
            d=0,
            e=self.pixres_y,
            f=self.uly
        )

    @classmethod
    def from_affine(
            cls,
            affine: Affine,
            epsg: int
        ):
        """
        Returns a ``GeoInfo`` instance from a ``rasterio.Affine`` object

        :param affine:
            ``rasterio.Affine`` object
        :param epsg:
            EPSG code identifying the spatial coordinate system
        :returns:
            new ``GeoInfo`` instance
        """
        return cls(
            epsg=epsg,
            ulx=affine.c,
            uly=affine.f,
            pixres_x=affine.a,
            pixres_y=affine.e
        )


class WavelengthInfo(object):
    """
    Class for storing information about the spectral wavelength of a
    raster band. Many optical sensors record data in spectral channels
    with a central wavelength and spectral band width.

    :attrib central_wavelength:
        central spectral wavelength.
    :attrib band_width:
        spectral band width. This is defined as the difference between
        the upper and lower spectral wavelength a sensor is recording
        in a spectral channel.
    :attrib wavelength_unit:
        physical unit in which `central_wavelength` and `band_width`
        are recorded. Usually 'nm' (nano-meters) or 'um' (micro-meters)
    """

    def __init__(
            self,
            central_wavelength: Union[int, float],
            wavelength_unit: str,
            band_width: Optional[Union[int, float]] = 0.,
        ):
        """
        Constructor to derive a new `WavelengthInfo` instance for
        a (spectral) raster band.

        :param central_wavelength:
            central wavelength of the band
        :param wavelength_unit:
            physical unit in which the wavelength is provided
        :param band_width:
            width of the spectral band (optional). If not provided
            assumes a width of zero wavelength units.
        """

        # wavelengths must be > 0:
        if central_wavelength <= 0.:
            raise ValueError('Wavelengths must be positive!')
        # band widths must be positive numbers
        if band_width < 0:
            raise ValueError('Bandwidth must not be negative')

        object.__setattr__(self, 'central_wavelength', central_wavelength)
        object.__setattr__(self, 'wavelength_unit', wavelength_unit)
        object.__setattr__(self, 'band_width', band_width)

    def __setattr__(self, *args, **kwargs):
        raise TypeError('WavelengthInfo object attributes are immutable')

    def __delattr__(self, *args, **kwargs):
        raise TypeError('WavelengthInfo object attributes are immutable')

    def __repr__(self) -> str:
        return str(self.__dict__) 
     

class Band(object):
    """
    Class for storing, accessing and modifying a raster band

    :attrib band_name:
        the band name identifies the raster band (e.g., 'B1'). It can be
        any character string
    :attrib color_name:
        if the raster band comes from an imaging sensor, the color name
        identifies the band in a color space (e.g., 'blue'). The color name
        can be seen as an alias to the actual `band_name`
    :attrib wavelength_info:
        optional wavelength info about the band to allow for localizing the
        band data in the spectral domain (mostly required for data from optical
        imaging sensors).
    :attrib nrows:
        number of rows of the raster band (extent in y dimension)
    :attrib ncols:
        number of columns of the raster band (extent in x dimension)
    :attrib epsg:
        EPSG code of the spatial reference system the raster data is projected
        to.
    :attrib bounds:
        spatial boundaries of the raster band (i.e., its footprint). The boundary
        is a rectangle of the size of the raster band and defines its location on
        Earth. The spatial reference system is defined by the `epsg` code.
    :attrib ulx:
        upper left x coordinate of the raster band in the spatial reference system
        defined by the EPSG code. We assume ``GDAL`` defaults, therefore the coordinate
        should refer to the upper left *pixel* corner.
    :attrib uly:
        upper left y coordinate of the raster band in the spatial reference system
        defined by the EPSG code. We assume ``GDAL`` defaults, therefore the coordinate
        should refer to the upper left *pixel* corner.
    :attrib scale:
        scale (aka gain) parameter of the raster data.
    :attrib offset:
        offset parameter of the raster data.
    :attrib values:
        the actual raster data as ``numpy.ndarray``, ``numpy.ma.MaskedArray`` or
        ``zarr``. The type depends on how the constructor is called.
    """

    def __init__(
            self,
            band_name: str,
            values: Union[np.ndarray, np.ma.MaskedArray, zarr.core.Array],
            geo_info: GeoInfo,
            color_name: Optional[str] = '',
            wavelength_info: Optional[WavelengthInfo] = None,
            scale: Optional[Union[int, float]] = 1.,
            offset: Optional[Union[int, float]] = 0.,
            unit: Optional[str] = '',
            nodata: Optional[Union[int,float]] = None,
            is_tiled: Optional[Union[int, bool]] = 0
        ):
        """
        Constructor to instantiate a new band object.

        :param band_name:
            name of the band.
        :param values:
            data of the band. Can be any numpy ``ndarray`` or ``maskedArray``
            as well as a ``zarr`` instance as long as its two-dimensional.
        :param geo_info:
            `~agrisatpy.core.band.GeoInfo` instance to allow for localizing
            the band data in a spatial reference system
        :param wavelength_info:
            optional `~agrisatpy.core.band.WavelengthInfo` instance denoting
            the spectral wavelength properties of the band. It is recommended
            to pass this parameter for optical sensor data.
        :param scale:
            optional scale (aka gain) factor for the raster band data. Many
            floating point datasets are scaled by a large number to allow for
            storing data as integer arrays to save disk space. The scale factor
            should allow to scale the data back into its original value range.
            For instance, Sentinel-2 MSI data is stored as unsigned 16-bit
            integer arrays but actually contain reflectance factor values between
            0 and 1. If not provided, `scale` is set to 1.
        :param offset:
            optional offset for the raster band data. As for the gain factor the
            idea is to scale the original band data in such a way that it's either
            possible to store the data in a certain data type or to avoid certain
            values. If not provided, `offset` is set to 0.
        :param unit:
            optional (SI) physical unit of the band data (e.g., 'meters' for
            elevation data)
        :param nodata:
            numeric value indicating no-data. If not provided the nodata value
            is set to ``numpy.nan`` for floating point data, 0 and -999 for
            unsigned and signed integer data, respectively.
        :param is_tiled:
            boolean flag indicating if the raster data is sub-divided into
            tiles. False (zero) by default.
        """

        # make sure the passed values are 2-dimensional
        if len(values.shape) != 2:
            raise ValueError('Only two-dimensional arrays are allowed')

        # check nodata value
        if nodata is None:
            if values.dtype in ['float16', 'float32', 'float64']:
                nodata = np.nan
            elif values.dtype in ['int16', 'int32', 'int64']:
                nodata = -999
            elif values.dtype in ['uint8', 'uint16', 'uint32', 'uint64']:
                nodata = 0

        object.__setattr__(self, 'band_name', band_name)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'geo_info', geo_info)
        object.__setattr__(self, 'color_name', color_name)
        object.__setattr__(self, 'wavelength_info', wavelength_info)
        object.__setattr__(self, 'scale', scale)
        object.__setattr__(self, 'offset', offset)
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'nodata', nodata)
        object.__setattr__(self, 'is_tiled', is_tiled)

    def __setattr__(self, *args, **kwargs):
        raise TypeError('Band object attributes are immutable')

    def __delattr__(self, *args, **kwargs):
        raise TypeError('Band object attributes immutable')

    @property
    def alias(self) -> Union[str, None]:
        """Alias of the band name (if available)"""
        if self.has_alias:
            return self.color_name

    @property
    def bounds(self) -> box:
        """Spatial bounding box of the band"""
        minx = self.geo_info.ulx
        maxx = minx + self.ncols * self.geo_info.pixres_x
        maxy = self.geo_info.uly
        miny = maxy + self.nrows * self.geo_info.pixres_y
        return box(minx, miny, maxx, maxy)

    @property
    def coordinates(self) -> Dict[str, np.ndarray]:
        """x-y spatial band coordinates"""
        nx, ny = self.ncols, self.nrows
        transform = self.transform
        x, _ = transform * (np.arange(nx), np.zeros(nx))
        _, y = transform * (np.zeros(ny), np.arange(ny))

        return {'x': x, 'y': y}

    @property
    def crs(self) -> CRS:
        """Coordinate Reference System of the band"""
        return CRS.from_epsg(self.geo_info.epsg)

    @property
    def has_alias(self) -> bool:
        """Checks if a color name can be used for aliasing"""
        return self.color_name != ''

    @property
    def is_zarr(self) -> bool:
        """Checks if the band values are a zarr array"""
        return isinstance(self.values, zarr.core.Array)

    @property
    def is_ndarray(self) -> bool:
        """Checks if the band values are a numpy ndarray"""
        return isinstance(self.values, np.ndarray) and not \
            self.is_masked_array

    @property
    def is_masked_array(self) -> bool:
        """Checks if the band values are a numpy masked array"""
        return isinstance(self.values, np.ma.MaskedArray)

    @property
    def meta(self) -> Dict[str, Any]:
        """
        Provides a ``rasterio`` compatible dictionary with raster
        metadata
        """
        return {
            'width': self.ncols,
            'height': self.nrows,
            'transform': self.geo_info.as_affine(),
            'count': 1,
            'dtype': str(self.values.dtype),
            'crs': self.crs
        }

    @property
    def nrows(self) -> int:
        """Number of rows of the band"""
        return self.values.shape[0]

    @property
    def ncols(self) -> int:
        """Number of columns of the band"""
        return self.values.shape[1]

    @property
    def transform(self) -> Affine:
        """Affine transformation of the band"""
        return self.geo_info.as_affine()

    @staticmethod
    def _get_pixel_geometries(
            vector_features: Union[Path, gpd.GeoDataFrame],
            fpath_raster: Optional[Path] = None,
            raster_crs: Union[int, CRS] = None
        ) -> gpd.GeoDataFrame:
        """
        Process passed pixel geometries including reprojection of the
        vector features (if required) into the spatial reference system
        of the raster band and extraction of centroid coordinates if
        the vector features are of type ``Polygon`` or ``MultiPolygon``

        :param vector_features:
            passed vector features to calling instance or class method
        :param fpath_raster:
            optional file path to the raster dataset. To be used when
            called from a classmethod
        :param raster_crs:
            optional raster EPSG code. To be used when called from an
            instance method.
        :returns:
            ``GeoDataFrame`` with ``Point`` features for extracting
            pixel values
        """
        # check input point features
        gdf = check_aoi_geoms(
            in_dataset=vector_features,
            fname_raster=fpath_raster,
            raster_crs=raster_crs,
            full_bounding_box_only=False
        )
        allowed_geometry_types = ['Point', 'Polygon', 'MultiPolygon']
        gdf = check_geometry_types(
            in_dataset=gdf,
            allowed_geometry_types=allowed_geometry_types
        )

        # convert to centroids if the geometries are not of type Point
        gdf.geometry = gdf.geometry.apply(
            lambda x: x.centroid if x.type in ['Polygon', 'MultiPolygon'] else x
        )

        return gdf

    @classmethod
    def from_rasterio(
            cls,
            fpath_raster: Path,
            band_idx: Optional[int] = 1,
            band_name_src: Optional[str] = '',
            band_name_dst: Optional[str] = 'B1',
            vector_features: Optional[Union[Path, gpd.GeoDataFrame]] = None,
            full_bounding_box_only: Optional[bool] = False,
            **kwargs
        ):
        """
        Creates a new ``Band`` instance from any raster dataset understood
        by ``rasterio``. Reads exactly **one** band from the input dataset!

        NOTE:
            To read a spatial subset of raster band data only pass
            `vector_features` which can be one to N (multi)polygon features.
            For Point features refer to the `read_pixels` method.

        :param fpath_raster:
            file-path to the raster file from which to read a band
        :param band_idx:
            band index of the raster band to read (starting with 1). If not
            provided the first band will be always read. Ignored if 
            `band_name_src` is provided.
        :param band_name_src:
            instead of providing a band index to read (`band_idx`) a band name
            can be passed. If provided `band_idx` is ignored.
        :param band_name_dst:
            name of the raster band in the resulting ``Band`` instance. If
            not provided the default value ('B1') is used. Whenever the band
            name is known it is recommended to use a meaningful band name!
        :param vector_features:
            ``GeoDataFrame`` or file with vector features in a format understood
            by ``fiona`` with one or more vector features of type ``Polygon``
            or ``MultiPolygon``. Unless `full_bounding_box_only` is set to True
            masks out all pixels not covered by the provided vector features.
            Otherwise the spatial bounding box encompassing all vector features
            is read as a spatial subset of the input raster band.
            If the coordinate system of the vector differs from the raster data
            source the vector features are projected into the CRS of the raster
            band before extraction.
        :param full_bounding_box_only:
            if False (default) pixels not covered by the vector features are masked
            out using ``maskedArray`` in the back. If True, does not mask pixels
            within the spatial bounding box of the `vector_features`.
        :param kwargs:
            further key-word arguments to pass to `~agrisatpy.core.band.Band`.
        :returns:
            new ``Band`` instance from a ``rasterio`` dataset.
        """

        # check vector features if provided
        masking = False
        if vector_features is not None:

            masking = True
            gdf_aoi = check_aoi_geoms(
                in_dataset=vector_features,
                fname_raster=fpath_raster,
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

        # read data using rasterio
        with rio.open(fpath_raster, 'r') as src:

            # parse image attributes
            attrs = get_raster_attributes(riods=src)
            transform = src.meta['transform']
            epsg = src.meta['crs'].to_epsg()

            # overwrite band_idx if band_name_src is provided
            band_names = list(src.descriptions)
            if band_name_src != '':
                if band_name_src not in band_names:
                    raise BandNotFoundError(
                        f'Could not find band "{band_name_src}" ' \
                        f'in {fpath_raster}'
                    )
                band_idx = band_names.index(band_name_src)

            # check if band_idx is valid
            if band_idx > len(band_names):
                raise IndexError(
                    f'Band index {band_idx} is out of range for a ' \
                    f'dataset with {len(band_names)} bands')
                

            # read selected band
            if not masking:
                # TODO: add zarr support here -> when is_tile == 1
                if attrs.get('is_tile', 0) == 1:
                    pass
                band_data = src.read(band_idx)
            else:
                band_data, transform = rio.mask.mask(
                    src,
                    gdf_aoi.geometry,
                    crop=True, 
                    all_touched=True, # IMPORTANT!
                    indexes=band_idx,
                    filled=False
                )
                # check if the mask contains any True value
                # if not cast the array from maskedArray to ndarray
                if np.count_nonzero(band_data.mask) == 0:
                    band_data = band_data.data

        # get scale, offset and unit (if available) from the raster attributes
        scale, scales = 1, attrs.get('scales', None)
        if scales is not None:
            scale = scales[band_idx-1]
        offset, offsets = 0, attrs.get('offsets', None)
        if offsets is not None:
            offset = offsets[band_idx-1]
        unit, units = '', attrs.get('unit', None)
        if units is not None:
            unit = units[band_idx-1]
        nodata, nodata_vals = None, attrs.get('nodatavals', None)
        if nodata_vals is not None:
            nodata = nodata_vals[band_idx-1]
        is_tiled = attrs.get('is_tiled', 0)

        # reconstruct geo-info
        geo_info = GeoInfo(
            epsg=epsg,
            ulx=transform.c,
            uly=transform.f,
            pixres_x=transform.a,
            pixres_y=transform.e
        )

        # create new Band instance
        return cls(
            band_name=band_name_dst,
            values=band_data,
            geo_info=geo_info,
            scale=scale,
            offset=offset,
            unit=unit,
            nodata=nodata,
            is_tiled=is_tiled,
            **kwargs
        )

    @classmethod
    def from_vector(
            cls,
            vector_features: Union[Path, gpd.GeoDataFrame],
            pixres_x: Union[int, float],
            pixres_y: Union[int, float],
            band_name_src: Optional[str] = '',
            band_name_dst: Optional[str] = 'B1',
            nodata_dst: Optional[Union[int, float]] = 0,
            snap_bounds: Optional[Polygon] = None,
            dtype_src: Optional[str] = 'float32',
            **kwargs
        ):
        """
        Creates a new ``Band`` instance from a ``GeoDataFrame`` or a file with
        vector features in a format understood by ``fiona`` with geometries
        of type ``Point``, ``Polygon`` or ``MultiPolygon`` using a single user-
        defined attribute (column in the data frame). The spatial reference
        system of the resulting band will be the same as for the input vector data.

        :param vector_featueres:
            file-path to a vector file or ``GeoDataFrame`` from which to convert
            a column to raster. Please note that the column must have a numerical
            data type.
        :param pixres_x:
            pixel grid cell size in x-directorion in the unit given by the spatial
            reference system of the vector features.
        :param pixres_y:
            pixel grid cell size in y-directorion in the unit given by the spatial
            reference system of the vector features.
        :param band_name_src:
            name of the attribute in the vector features' attribute table to
            convert to a new ``Band`` instance
        :param band_name_dst:
            name of the resulting ``Band`` instance. "B1" by default.
        :param nodata_dst:
            nodata value in the resulting band data to fill raster grid cells
            having no value assigned from the input vector features. If not
            provided the nodata value is set to 0 (rasterio default)
        :param dtype_src:
            datatype of the resulting raster array. Per default "float32" is used.
        :param kwargs:
            additional key-word arguments to pass to `~agrisatpy.core.Band`
        :returns:
            new ``Band`` instance from a vector features source
        """

        # check passed vector geometries
        if isinstance(vector_features, Path):
            gdf_aoi = gpd.read_file(vector_features)
        else:
            gdf_aoi = vector_features.copy()

        allowed_geometry_types = ['Point', 'Polygon', 'MultiPolygon']
        in_gdf = check_geometry_types(
            in_dataset=gdf_aoi,
            allowed_geometry_types=allowed_geometry_types
        )

        # check passed attribute selection
        if not band_name_src in in_gdf.columns:
            raise AttributeError(
                f'{band_name_src} not found in input vector dataset'
            )

        # infer the datatype (i.e., try if it is possible to cast the
        # attribute to float32, otherwise do not process the feature)
        try:
            in_gdf[band_name_src].astype(dtype_src)
        except ValueError as e:
            raise TypeError(
                f'Attribute "{band_name_src}" seems not to be numeric')

        # clip features to the spatial extent of a bounding box if available
        # clip the input to the bounds of the snap band
        if snap_bounds is not None:
            try:
                in_gdf = in_gdf.clip(
                    mask=snap_bounds
                )
            except Exception as e:
                raise DataExtractionError(
                    'Could not clip input vector features to ' \
                    f'snap raster bounds: {e}'
                )

        # make sure there are still features left
        if in_gdf.empty:
            raise DataExtractionError(
                'Seems there are no features to convert'
        )

        # infer shape and affine of the resulting raster grid if not provided
        if snap_bounds is None:
            if set(in_gdf.geometry.type.unique()) == set(['Polygon', 'MultiPolygon']):
                minx = in_gdf.geometry.bounds.minx.min()
                maxx = in_gdf.geometry.bounds.maxx.max()
                miny = in_gdf.geometry.bounds.miny.min()
                maxy = in_gdf.geometry.bounds.maxy.max()
            else:
                minx = in_gdf.geometry.x.min()
                maxx = in_gdf.geometry.x.max()
                miny = in_gdf.geometry.y.min()
                maxy = in_gdf.geometry.y.max()
            snap_bounds = box(minx, miny, maxx, maxy)
        else:
            minx, miny, maxx, maxy = snap_bounds.exterior.bounds

        # calculate number of columns from bounding box of all features
        # always round to the next bigger integer value to make sure no
        # value gets lost
        rows = int(np.ceil(abs((maxy - miny) / pixres_y)))
        cols = int(np.ceil(abs((maxx - minx) / pixres_x)))
        snap_shape = (rows, cols)

        # initialize new GeoInfo instance
        geo_info = GeoInfo(
            epsg=in_gdf.crs.to_epsg(),
            ulx=minx,
            uly=maxy,
            pixres_x=pixres_x,
            pixres_y=pixres_y
        )

        # rasterize the vector features
        try:
            shapes = (
                    (geom,value) for geom, value in zip(
                        in_gdf.geometry,
                        in_gdf[band_name_src].astype(dtype_src)
                    )
                )
            rasterized = features.rasterize(
                shapes=shapes,
                out_shape=snap_shape,
                transform=geo_info.as_affine(),
                all_touched=True,
                fill=nodata_dst,
                dtype=dtype_src
            )
        except Exception as e:
            raise Exception(
                    f'Could not process attribute "{band_name_src}": {e}'
                )

        # initialize new Band instance
        return cls(
            band_name=band_name_dst,
            values=rasterized,
            geo_info=geo_info,
            nodata=nodata_dst,
            **kwargs
        )

    @classmethod
    def read_pixels(
            cls,
            fpath_raster: Path,
            vector_features: Union[Path, gpd.GeoDataFrame],
            band_idx: Optional[int] = 1,
            band_name_src: Optional[str] = '',
            band_name_dst: Optional[str] = 'B1'
        ) -> gpd.GeoDataFrame:
        """
        Reads single pixel values from a raster dataset into a ``GeoDataFrame``

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
        :param band_idx:
            band index of the raster band to read (starting with 1). If not
            provided the first band will be always read. Ignored if 
            `band_name_src` is provided.
        :param band_name_src:
            instead of providing a band index to read (`band_idx`) a band name
            can be passed. If provided `band_idx` is ignored.
        :param band_name_dst:
            name of the raster band in the resulting ``GeoDataFrame`` (i.e.,
            column name)
        :returns:
            ``GeoDataFrame`` with extracted pixel values. If the vector features
            defining the sampling points are not within the spatial extent of the
            raster dataset the pixel values are set to nodata (inferred from
            the raster source)
        """

        # check input point features
        gdf = cls._get_pixel_geometries(
            vector_features=vector_features,
            fpath_raster=fpath_raster
        )

        # use rasterio.sample to extract the pixel values from the raster
        # to do so, we need a list of coordinate tuples
        coord_list = [(x,y) for x,y in zip(gdf.geometry.x , gdf['geometry'].y)]
        with rio.open(fpath_raster, 'r') as src:
            # overwrite band_idx if band_name_src is provided
            band_names = list(src.descriptions)
            if band_name_src != '':
                if band_name_src not in band_names:
                    raise BandNotFoundError(
                        f'Could not find band "{band_name_src}" ' \
                        f'in {fpath_raster}'
                    )
                band_idx = band_names.index(band_name_src)
            try:
                # yield all values from the generator
                def _sample(src, coord_list, band_idx):
                    yield from src.sample(coord_list, band_idx)
                pixel_samples =list(_sample(src, coord_list, band_idx))
            except Exception as e:
                raise Exception(f'Extraction of pixels from raster failed: {e}')

        # append the extracted pixels to the exisiting geodataframe. We can do
        # so, because we have passed the pixels in the same order as they occur
        # in the dataframe
        band_list = [x[0] for x in pixel_samples]
        gdf[band_name_dst] = band_list

        return gdf

    def _flatten_coordinates(self) -> Dict[str, np.array]:
        """
        Flattens band coordinates. To be used when converting an array to
        ``geopandas.GeoDataFrame``.

        :returns:
            dictionary of ``numpy.ndarray`` containing the x and y
            coordinates in flattened format to match the flattened band
            values in ``Fortran`` order
        """
             
        # get coordinates
        coords = self.coordinates
        # flatten x coordinates along the y-axis
        flat_x_coords = np.repeat(coords['x'], self.nrows)
        # flatten y coordinates along the x-axis
        flat_y_coords = np.tile(coords['y'], self.ncols)
    
        out_coords = {
            'x': flat_x_coords,
            'y': flat_y_coords
        }
    
        return out_coords

    def copy(self):
        """
        Returns a copy of the current ``Band`` instance
        """
        attrs = deepcopy(self.__dict__)
        return Band(**attrs)

    def get_attributes(
            self,
            **kwargs
        ) -> Dict[str, Any]:
        """
        Returns raster data attributes in ``rasterio`` compatible way

        :param kwargs:
            key-word arguments to insert into the raster attributes
        :returns:
            dictionary compatible with ``rasterio`` attributes
        """
        attrs = {}
        attrs['is_tiled'] = self.is_tiled
        attrs['nodatavals'] = (self.nodata, )
        attrs['scales'] = (self.scale, )
        attrs['offsets'] = (self.offset, )
        attrs['descriptions'] = (self.color_name, )
        attrs['crs'] = self.geo_info.epsg
        attrs['transform'] = tuple(self.transform)
        attrs['units'] = (self.unit, )
        attrs.update(kwargs)

        return attrs

    def get_meta(
            self,
            driver: Optional[str] = 'gTiff',
            **kwargs
        ) -> Dict[str, Any]:
        """
        Returns a ``rasterio`` compatible dictionary with raster dataset
        metadata.

        :param driver:
            name of the ``rasterio`` driver. `gTiff` (GeoTiff) by default
        :param kwargs:
            additional keyword arguments to append to metadata dictionary
        :returns:
            ``rasterio`` compatible metadata dictionary to be used for
            writing new raster datasets
        """
        meta = {}
        meta['height'] = self.nrows
        meta['width'] = self.ncols
        meta['crs'] = self.crs
        meta['dtype'] = str(self.values.dtype)
        meta['count'] = 1
        meta['nodata'] = self.nodata
        meta['transform'] = self.transform
        meta['is_tile'] = self.is_tiled
        meta['driver'] = driver
        meta.update(kwargs)

        return meta

    def get_pixels(
            self,
            vector_features: Union[Path, gpd.GeoDataFrame]
        ):
        """
        Returns pixel values from a ``Band`` instance raster values.

        The extracted band array values are stored in a new column in the
        returned `vector_features` ``GeoDataFrame`` named like the name
        of the band.

        If you do not want to read the entire raster data first consider
        using `~agrisatpy.core.Band.read_pixels` instead.

        NOTE:
            Masked pixels are set to the band's nodata value.

        :param vector_features:
            file-path or ``GeoDataFrame`` to features defining the pixels to read
            from the ``Band`` raster values. The geometries can be of type ``Point``,
            ``Polygon`` or ``MultiPolygon``. In the latter two cases the centroids
            are used to extract pixel values, whereas for point features the
            closest raster grid cell is selected.
        """

        # get pixel point features
        gdf = self._get_pixel_geometries(
            vector_features=vector_features,
            raster_crs=self.crs
        )

        # drop points outside of the band's bounding box (speeds up the process)
        band_bbox = BoundingBox(*self.bounds.exterior.bounds)
        gdf = gdf.cx[band_bbox.left:band_bbox.right, band_bbox.bottom:band_bbox.top]

        # define helper function for getting the closest array index for a coordinate
        # map the coordinates to array indices
        def _find_nearest_array_index(array, value):
            return np.abs(array - value).argmin()

        # calculate the x and y array indices required to extract the pixel values
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y

        # get band coordinates
        coords = self.coordinates

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
        gdf[self.band_name] = np.empty(gdf.shape[0])

        # loop over sample points and add them as new entries to the GeoDataFrame
        for _, record in gdf.iterrows():
            # get array value for the current column and row, continue on out-of-bounds error
            try:
                array_value = self.values[record.row, record.col]
            except IndexError:
                continue
            # ignore masked pixels
            if self.is_masked_array:
                if array_value.mask:
                    array_value = self.nodata
            gdf.loc[
                (gdf.row == record.row) & (gdf.col == record.col),
                self.band_name
            ] = array_value

        # clean up GeoDataFrame
        cols_to_drop = ['row', 'col', 'x', 'y']
        for col_to_drop in cols_to_drop:
            gdf.drop(col_to_drop, axis=1, inplace=True)

        return gdf
        
    def plot(
            self,
            colormap: Optional[str] = 'gray',
            discrete_values: Optional[bool] = False,
            user_defined_colors: Optional[ListedColormap] = None,
            user_defined_ticks: Optional[List[Union[str,int,float]]] = None,
            colorbar_label: Optional[str] = None,
            vmin: Optional[Union[int, float]] = None,
            vmax: Optional[Union[int, float]] = None,
            fontsize: Optional[int] = 12
        ) -> plt.Figure:
        """
        Plots the raster values using ``matplotlib``

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
            lower value to use for `~matplotlib.pyplot.imshow()`. If None it
            is set to the lower 5% percentile of the data to plot.
        :param vmin:
            upper value to use for `~matplotlib.pyplot.imshow()`. If None it
            is set to the upper 95% percentile of the data to plot.
        :param fontsize:
            fontsize to use for axes labels, plot title and colorbar label.
            12 pts by default.
        :returns:
            matplotlib figure object with the band data
            plotted as map
        """

        # get the bounds of the band
        bounds = BoundingBox(*self.bounds.exterior.bounds)

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
            unique_values = np.unique(self.values)
            norm = mpl.colors.BoundaryNorm(unique_values, cmap.N)
            img = ax.imshow(
                self.values,
                cmap=cmap,
                norm=norm,
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                interpolation='none'  # important, otherwise img will have speckle!
            )
        else:
            # clip data for displaying to central 90% percentile
            if vmin is None:
                vmin = np.nanquantile(self.values, 0.05)
            if vmax is None:
                vmax = np.nanquantile(self.values, 0.95)

            # actual displaying of the band data
            img = ax.imshow(
                self.values,
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
                    fontsize=fontsize,
                    labelpad=20,
                    y=0.5
                )

        ax.title.set_text(self.band_name.upper())
        # add axes labels and format ticker
        epsg = self.geo_info.epsg
        ax.set_xlabel(f'X [m] (EPSG:{epsg})', fontsize=fontsize)
        ax.xaxis.set_ticks(np.arange(bounds.left, bounds.right, x_interval))
        ax.set_ylabel(f'Y [m] (EPSG:{epsg})', fontsize=fontsize)
        ax.yaxis.set_ticks(np.arange(bounds.bottom, bounds.top, y_interval))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

        return fig

    def mask(
            self,
            mask: np.ndarray,
            inplace: Optional[bool] = False
        ):
        """
        Mask out pixels based on a boolean array.

        NOTE:
            If the band is already masked, the new mask updates the
            existing one. I.e., pixels already masked before remain
            masked.

        :param mask:
            ``numpy.ndarray`` of dtype ``boolean`` to use as mask.
            The mask must match the shape of the raster data.
        :param inplace:
            if False (default) returns a copy of the ``Band`` instance
            with the changes applied. If True overwrites the values
            in the current instance.
        :returns:
            ``Band`` instance if `inplace` is False, None instead.
        """

        # check shape of mask passed and its dtype
        if mask.dtype != 'bool':
            raise TypeError('Mask must be boolean')

        if mask.shape != self.values.shape:
            raise ValueError(
                f'Shape of mask {mask.shape} does not match ' \
                f'shape of band data {self.values.shape}')

        # check if array is already masked
        if self.is_masked_array:
            orig_mask = self.values.mask
            # update the existing mask
            for row in range(self.nrows):
                for col in range(self.ncols):
                    # ignore pixels already masked
                    if not orig_mask[row, col]:
                        orig_mask[row, col] = mask[row,col]
            # update band data array
            masked_array = np.ma.MaskedArray(
                data=self.values.data,
                mask=orig_mask
            )
        elif self.is_ndarray:
            masked_array = np.ma.MaskedArray(
                data=self.values,
                mask=mask
            )
        elif self.is_zarr:
            raise NotImplemented()

        if inplace:
            object.__setattr__(self, 'values', masked_array)
        else:
            attrs = deepcopy(self.__dict__)
            attrs.update({'values': masked_array})
            return Band(**attrs)

    def resample(
            self,
            target_resolution: Union[int, float],
            interpolation_method: Optional[int] = cv2.INTER_NEAREST_EXACT,
            target_shape: Optional[Tuple[int, int]] = None,
            inplace: Optional[bool] = False
        ):
        """
        Changes the raster grid cell (pixel) size.
        Nodata pixels are not used for resampling.

        :param target_resolution:
            spatial resolution (grid cell size) in units of the spatial
            reference system. Applies to x and y direction.
        :param interpolation_method:
            opencv interpolation method. Per default nearest neighbor
            interpolation is used (`~cv2.INTER_NEAREST_EXACT`). See the
            `~cv2` documentation for a list of available methods.
        :param target_shape:
            shape of the output in terms of number of rows and columns.
            If None (default) the `target_shape` parameter is inferred
            from the band data. If you want to make sure the output is
            *aligned* with another raster band (co-registered) provide
            this parameter.
        :param inplace:
            if False (default) returns a copy of the ``Band`` instance
            with the changes applied. If True overwrites the values
            in the current instance.
        :returns:
            ``Band`` instance if `inplace` is False, None instead.
        """

        # resampling currently works on grids with identitical x and y
        # grid cell size only
        if abs(self.geo_info.pixres_x) != abs(self.geo_info.pixres_y):
            raise NotImplementedError(
                'Resampling currently supports regular grids only '\
                'where the grid cell size is the same in x and y ' \
                'direction'
            )

        # check if a target shape is provided
        if target_shape is not None:
            nrows_resampled = target_shape[0]
            ncols_resampled = target_shape[1]
        # if not determine the extent from the bounds
        else:
            bounds = BoundingBox(*self.bounds.exterior.bounds)
            # calculate new size of the raster
            ncols_resampled = int(np.ceil((bounds.right - bounds.left) / target_resolution))
            nrows_resampled = int(np.ceil((bounds.top - bounds.bottom) / target_resolution))
            target_shape = (nrows_resampled, ncols_resampled)

        # opencv2 switches the axes order!
        dim_resampled = (ncols_resampled, nrows_resampled)

        # check if the band data is stored in a masked array
        # if so, replace the masked values with NaN
        if self.is_masked_array:
            band_data = deepcopy(self.values.data)
        elif self.is_ndarray:
            band_data = deepcopy(self.values)
        elif self.is_zarr:
            raise NotImplementedError()
            
        scaling_factor = abs(self.geo_info.pixres_x / target_resolution)
        blackfill_value = self.nodata

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
                    interpolation=interpolation_method
                )
            except Exception as e:
                raise ResamplingFailedError(e)
        else:
            # blackfill pixel should be set to NaN before resampling
            type_casting = False
            if band_data.dtype in ['uint8', 'uint16', 'uint32', \
            'int8', 'int16','int32', 'int64']:
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
                    interpolation=interpolation_method
                )
            except Exception as e:
                raise ResamplingFailedError(e)

            # in addition, run pixel division since there will be too many NaN pixels
            # when using only res from cv2 resize as it sets pixels without full
            # spatial context to NaN. This works, however, only if the target resolution
            # decreases the current pixel resolution by an integer scalar
            try:
                res_pixel_div = upsample_array(
                    in_array=band_data,
                    scaling_factor=int(scaling_factor)
                )
            except Exception as e:
                res_pixel_div = np.zeros(0)

            # replace NaNs with values from pixel division (if possible); thus we will
            # get all pixel values and the correct blackfill
            # when working on spatial subsets this might fail because of shape mismatches;
            # in this case keep the cv2 output, which means loosing a few pixels
            if res.shape == res_pixel_div.shape:
                res[np.isnan(res)] = res_pixel_div[np.isnan(res)]
            else:
                res[np.isnan(res)] = blackfill_value

            # cast back to original datatype if required
            if type_casting:
                res = res.astype(band_data.dtype)

        # if the array is masked, resample the mask as well
        if self.is_masked_array:
            # convert bools to int8 (cv2 does not support boolean arrays)
            in_mask = deepcopy(self.values.mask).astype('uint8') 
            out_mask = cv2.resize(
                in_mask,
                dim_resampled,
                cv2.INTER_NEAREST_EXACT
            )
            # convert mask back to boolean array
            out_mask = out_mask.astype(bool)
            # save as masked array
            res = np.ma.masked_array(data=res, mask=out_mask)

        # update the geo_info
        geo_info = deepcopy(self.geo_info.__dict__)
        geo_info.update({
            'pixres_x': np.sign(band.geo_info.pixres_x) * target_resolution,
            'pixres_y': np.sign(band.geo_info.pixres_y) * target_resolution
        })
        new_geo_info = GeoInfo(**geo_info)

        # TODO: think about ulx and uly if the target_shape is passed!!!
        if inplace:
            object.__setattr__(self, 'values', res)
            object.__setattr__(self, 'geo_info', new_geo_info)
        else:
            attrs = deepcopy(self.__dict__)
            attrs.update({
                'values': res,
                'geo_info': new_geo_info
            })
            return Band(**attrs)

    def reproject(
            self,
            target_crs: Union[int, CRS],
            dst_transform: Optional[Affine] = None,
            interpolation_method: Optional[int] = Resampling.nearest,
            num_threads: Optional[int] = 1,
            inplace: Optional[bool] = False,
            **kwargs
        ):
        """
        Projects the raster data into a different spatial coordinate system
        """

        # collect options for reprojection
        reprojection_options = {
            'src_crs': self.crs,
            'src_transform': self.transform,
            'dst_crs': target_crs,
            'src_nodata': self.nodata,
            'resampling': interpolation_method,
            'num_threads': num_threads,
            'dst_transform': dst_transform
        }
        reprojection_options.update(kwargs)

        # check for array type; masked arrays are not supported directly
        # also we have to cast to float for performing the reprojection
        if self.is_masked_array:
            band_data = deepcopy(self.values.data).astype(float)
            band_mask = deepcopy(self.values.mask).astype(float)
        elif self.is_ndarray:
            band_data = deepcopy(self.values).astype(float)
        elif self.is_zarr:
            raise NotImplementedError()

        try:
            out_data, out_transform = reproject_raster_dataset(
                raster=band_data,
                **reprojection_options
            )
        except Exception as e:
            raise ReprojectionError(
                f'Could not re-project band {self.band_name}: {e}'
            )

        # cast array back to original dtype
        out_data = out_data[0,:,:].astype(self.values.dtype)

        # reproject the mask separately
        if self.is_masked_array:
            out_mask, _ = reproject_raster_dataset(
                raster=band_mask,
                **reprojection_options
            )
            out_mask = out_mask.astype(bool)
            out_data = np.ma.MaskedArray(
                data=out_data,
                mask=out_mask[0,:,:]
            )

        new_geo_info = GeoInfo.from_affine(
            affine=out_transform,
            epsg=target_crs
        )
        if inplace:
            object.__setattr__(self, 'values', out_data)
            object.__setattr__(self, 'geo_info', new_geo_info)
        else:
            attrs = deepcopy(self.__dict__)
            attrs.update({
                'values': out_data,
                'geo_info': new_geo_info
            })
            return Band(**attrs)

    def reduce(
            self,
            method: Union[str,List[str]],
            by: Optional[Union[Path, gpd.GeoDataFrame]] = None
        ) -> Dict[str, Union[int, float]]:
        """
        Reduces the raster data to scalar values.

        :param method:
            any ``numpy`` function taking a two-dimensional array as input
            and returning a single scalar. Can be a single function name
            (e.g., "mean") or a list of function names (e.g., ["mean", "median"])
        :returns:
            a dictionary with scalar results
        """

        # TODO: implement reduce by vector features

        if isinstance(method, str):
            method = [method]

        # determine numpy prefix
        if self.is_masked_array:
            numpy_prefix = 'np.ma'
        elif self.is_ndarray:
            numpy_prefix = 'np.'
        elif self.is_zarr:
            raise NotImplemented()
        
        # compute statistics
        stats = {}
        for operator in method:
            # formulate numpy expression
            expression = f'{numpy_prefix}.{operator}'
            # numpy.ma has some different function names to consider
            if operator.startswith('nan'):
                expression = f'{numpy_prefix}.{operator[3::]}'
            elif operator.endswith('nonzero'):
                expression = f'{numpy_prefix}.count'
            try:
                # get function object and use its __call__ method
                numpy_function = eval(expression)
                stats[operator] = numpy_function.__call__(
                    self.values
                )
            except TypeError:
                raise Exception(
                    f'Unknown function name for {numpy_prefix}: {operator}'
                )

        return stats

    def scale_data(
            self,
            inverse: Optional[bool] = False,
            inplace: Optional[bool] = False
        ):
        """
        Applies scale and offset factors to the data.

        :param inverse:
            if True reverse the scaling (i.e., takes the inverse
            of the scale factor and changes the sign of the offset)
        :param inplace:
            if False (default) returns a copy of the ``Band`` instance
            with the changes applied. If True overwrites the values
            in the current instance.
        :returns:
            ``Band`` instance if `inplace` is False, None instead.
        """

        if inverse:
            scale = 1. / self.scale
            offset = -1. * self.offset
        else:
            scale, offset = self.scale, self.offset

        if self.is_masked_array:
            scaled_array = scale * self.values.data + offset
            scaled_array = np.ma.MaskedArray(
                data=scaled_array,
                mask=self.values.mask
            )
        elif self.is_ndarray:
            scaled_array = scale * self.values + offset
        elif self.is_zarr:
            raise NotImplemented()

        if inplace:
            object.__setattr__(self, 'values', scaled_array)
        else:
            attrs = deepcopy(self.__dict__)
            attrs.update({'values': scaled_array})
            return Band(**attrs)

    def to_dataframe(self) -> gpd.GeoDataFrame:
        """
        Returns a ``GeoDataFrame`` from the raster band data

        :returns:
            ``GeoDataFrame`` of raster values in the spatial coordinate
            system of the raster band data. The geometry type is always
            ``Point``.
        """
        # get coordinates of the first band in flattened format
        coords = self._flatten_coordinates()
        # get EPSG code
        epsg = self.geo_info.epsg

        # if the band is a masked array, we need numpy.ma functions
        new_shape = self.nrows * self.ncols
        if self.is_masked_array:
            flattened = np.ma.reshape(self.values, new_shape, order='F')
            # save mask to array
            mask = flattened.mask
            # compress array (removes masked values)
            flattened = flattened.compressed()
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
        elif self.is_ndarray:
            flattened = np.reshape(self.values, new_shape, order='F')
        elif self.is_zarr:
            raise NotImplemented()

        # convert the coordinates to shapely geometries
        coordinate_geoms = [Point(c[0], c[1]) for c in list(zip(coords['x'], coords['y']))]
        # call the GeoDataFrame constructor
        gdf = gpd.GeoDataFrame(geometry=coordinate_geoms, crs=epsg)
        # add band data
        gdf[self.band_name] = flattened

        return gdf

    def to_xarray(
            self,
            attributes: Dict[str, Any] = {},
            **kwargs
        ):
        """
        Returns a ``xarray.Dataset`` from the raster band data.

        NOTE:
            To ensure consistency with ``xarray`` pixel coordinates are
            shifted from the upper left pixel corner to the center.

        :param attributes:
            additional raster attributes to update or add
        :param kwargs:
            additional key-word arguments to pass to `~xarray.Dataset`
        :return:
            ``xarray.DataSet`` with x and y coordinates. Raster attributes
            are preserved.
        """

        band_data = deepcopy(self.values)
        # masked pixels are set to nodata
        if self.is_masked_array:
            try:
                band_data.filled(self.nodata)
            # on type error cast to float since this is most likely caused
            # by a data type miss-match (int <-> float)
            except TypeError:
                band_data = band_data.astype(float)
                band_data.filled(self.nodata)
            except Exception as e:
                raise ValueError(
                    f'Cannot set masked pixels to nodata: {e}'
                )

        # get coordinates and shift them half a pixel size
        shift_x = 0.5 * self.geo_info.pixres_x
        shift_y = 0.5 * self.geo_info.pixres_y
        coords = self.coordinates
        coords.update({
            'x': [val + shift_x for val in coords['x']],
            'y': [val + shift_y for val in coords['y']]
        })

        # define attributes
        attrs = self.get_attributes(**attributes)

        # call Dataset constructor
        band_tuple = tuple([('y','x'), band_data])
        band_dict = {self.band_name: band_tuple}
        xds = xr.Dataset(
            band_dict,
            coords=coords,
            attrs=attrs,
            **kwargs
        )

        return xds

    def to_rasterio(
            self,
            fpath_raster: Path,
            **kwargs
        ) -> None:
        """
        Writes the band data to a raster dataset using ``rasterio``.

        :param fpath_raster:
            file-path to the raster dataset to create. The ``rasterio``
            driver is identified by the file-name extension. In case
            jp2 is selected, loss-less compression is carried out.
        :param kwargs:
            additional keyword arguments to append to metadata dictionary
            used by ``rasterio`` to write datasets
        """

        # check output file naming and driver
        try:
            driver = driver_from_extension(fpath_raster)
        except Exception as e:
            raise ValueError(
                'Could not determine GDAL driver for ' \
                f'{fpath_raster.name}: {e}'
            )

        # construct meta dictionary required by rasterio
        meta = self.get_meta(driver, **kwargs)

        # make sure JPEG compression is loss-less
        if driver == 'JP2OpenJPEG':
            meta.update({
                'QUALITY': '100',
                'REVERSIBLE': 'YES'
            })

        # open the result dataset and try to write the bands
        with rio.open(fpath_raster, 'w+', **meta) as dst:
            # set band name
            dst.set_band_description(1, self.band_name)
            # write band data
            dst.write(self.values, 1)


if __name__ == '__main__':

    # read data from raster
    fpath_raster = Path('../../../data/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    vector_features = Path('../../../data/sample_polygons/ZH_Polygons_2020_ESCH_EPSG32632.shp')
    vector_features_2 = Path('../../../data/sample_polygons/BY_Polygons_Canola_2019_EPSG32632.shp')

    band = Band.from_rasterio(
        fpath_raster=fpath_raster,
        band_idx=1,
        band_name_dst='B02',
        vector_features=vector_features,
        full_bounding_box_only=False
    )

    assert band.geo_info.epsg == 32632, 'wrong EPSG code'
    assert band.band_name == 'B02', 'wrong band name'
    assert band.is_masked_array, 'array has not been masked'
    assert band.alias is None, 'band alias was not set'
    assert set(band.coordinates.keys()) == {'x', 'y'}, 'coordinates not set correctly'
    assert band.nrows == band.values.shape[0], 'number of rows does not match data array'
    assert band.ncols == band.values.shape[1], 'number of rows does not match data array'
    assert band.coordinates['x'].shape[0] == band.ncols, \
        'number of x coordinates does not match number of columns'
    assert band.coordinates['y'].shape[0] == band.nrows, \
        'number of y coordinates does not match number of rows'
    assert band.values.min() == 19, 'wrong minimum returned from data array'
    assert band.values.max() == 9504, 'wrong maximum returned from data array'
    assert band.geo_info.ulx == 475420.0, 'wrong upper left x coordinate'
    assert band.geo_info.uly == 5256840.0, 'wrong upper left y coordinate'

    band_bounds_mask = band.bounds
    assert band_bounds_mask.type == 'Polygon'
    assert band_bounds_mask.exterior.bounds[0] == band.geo_info.ulx, \
        'upper left x coordinate does not match'
    assert band_bounds_mask.exterior.bounds[3] == band.geo_info.uly, \
        'upper left y coordinate does not match'

    assert not band.values.mask.all(), 'not all pixels should be masked'

    band.plot(colorbar_label='Surface Reflectance')

    # reproject to Swiss Coordinate System (EPSG:2056)
    reprojected = band.reproject(target_crs=2056)

    assert reprojected.crs == 2056, 'wrong EPSg after reprojection'
    assert reprojected.is_masked_array, 'mask must not be lost after reprojection'
    assert np.round(reprojected.geo_info.ulx) == 2693130.0, 'wrong upper left x coordinate'
    assert np.round(reprojected.geo_info.uly) == 1257861.0, 'wrong upper left y coordinate'
    reprojected.to_rasterio('/tmp/reprojected.jp2')

    # get band statistics
    stats = band.reduce(method=['mean', 'min', 'max'])
    mean_stats = band.reduce(method='mean')
    assert mean_stats['mean'] == stats['mean'], 'miss-match of metrics'
    assert stats['min'] == band.values.min(), 'minimum not calculated correctly'
    assert stats['max'] == band.values.max(), 'maximum not calculated correctly'

    # convert to GeoDataFrame
    gdf = band.to_dataframe()
    assert (gdf.geometry.type == 'Point').all(), 'wrong geometry type'
    assert set(gdf.columns) == {'geometry', 'B02'}, 'wrong column labels'
    assert gdf.shape[0] == 29674, 'wrong number of pixels converted'
    assert gdf.B02.max() == stats['max'], 'band statistics not the same after conversion'
    assert gdf.B02.min() == stats['min'], 'band statistics not the same after conversion'
    assert gdf.B02.mean() == stats['mean'], 'band statistics not the same after conversion'

    # convert to xarray
    xarr = band.to_xarray()
    assert xarr.x.values[0] == band.geo_info.ulx + 0.5*band.geo_info.pixres_x, \
        'pixel coordinate not shifted to center of pixel in xarray'
    assert xarr.y.values[0] == band.geo_info.uly + 0.5*band.geo_info.pixres_y, \
        'pixel coordinate not shifted to center of pixel in xarray'
    assert (xarr.B02.values == band.values.astype(float)).all(), \
        'array values changed after conversion to xarray'
    assert np.count_nonzero(~np.isnan(xarr.B02)) == band.values.compressed().shape[0], \
        'masked values were not set to nan correctly'
    assert xarr.dims['y'] == band.nrows and xarr.dims['x'] == band.ncols, \
        'wrong number of rows and columns in xarray'

    # resample to 20m spatial resolution using bi-cubic interpolation
    resampled = band.resample(
        target_resolution=20,
        interpolation_method=cv2.INTER_CUBIC
    )
    assert resampled.geo_info.pixres_x == 20, 'wrong pixel size after resampling'
    assert resampled.geo_info.pixres_y == -20, 'wrong pixel size after resampling'
    assert resampled.geo_info != band.geo_info, 'geo info must not be the same'
    assert resampled.ncols < band.ncols, 'spatial resolution should decrease'
    assert resampled.nrows < band.ncols, 'spatial resolution should decrease'
    assert resampled.is_masked_array, 'mask should be preserved'

    # resample to 5m inplace
    old_shape = (band.nrows, band.ncols)
    band.resample(
        target_resolution=5,
        inplace=True
    )
    assert band.nrows == 588, 'wrong number of rows after resampling'
    assert band.ncols == 442, 'wrong number of columns after resampling'
    assert band.is_masked_array, 'mask should be preserved'

    # resample back to 10m and align to old shape
    band.resample(
        target_resolution=10,
        target_shape=old_shape,
        interpolation_method=cv2.INTER_CUBIC,
        inplace=True
    )

    assert (band.nrows, band.ncols) == old_shape, 'resampling to target shape did not work'

    # test masking with boolean mask (should mask everything)
    mask = np.ndarray(band.values.shape, dtype='bool')
    mask.fill(True)

    # try without inplace
    masked_band = band.mask(mask=mask, inplace=False)
    assert isinstance(masked_band, Band), 'wrong return type'
    assert (masked_band.values.mask == mask).all(), 'mask not applied correctly'
    assert (masked_band.values.data == band.values.data).all(), 'data not preserved correctly'

    band.mask(mask=mask, inplace=True)
    assert band.values.mask.all(), 'not all pixels masked'

    # test scaling -> nothing should happen at this stage
    values_before_scaling = band.values
    band.scale_data()
    assert (values_before_scaling.data == band.values.data).all(), 'scaling must not have an effect'
    band.scale_data(inverse=True)
    assert (values_before_scaling.data == band.values.data).all(), 'scaling must not have an effect'

    # read data with AOI outside of the raster bounds -> should raise a ValueError
    # band = Band.from_rasterio(
    #     fpath_raster=fpath_raster,
    #     band_idx=1,
    #     band_name_dst='B02',
    #     vector_features=vector_features_2,
    #     full_bounding_box_only=False
    # )

    # write band data to disk
    fpath_out = Path('/tmp/test.jp2')
    band.to_rasterio(fpath_raster=fpath_out)

    assert fpath_out.exists(), 'output dataset not written'
    band_read_again = Band.from_rasterio(fpath_out)
    assert (band_read_again.values == band.values.data).all(), \
        'band data not the same after writing'

    # read single pixels from raster dataset
    pixels = Band.read_pixels(
        fpath_raster=fpath_raster,
        vector_features=vector_features
    )
    assert 'B1' in pixels.columns, 'extracted band data not found'
    gdf = gpd.read_file(vector_features)
    assert pixels.shape[0] == gdf.shape[0], 'not all geometries extracted'
    assert pixels.geometry.type.unique().shape[0] == 1, 'there are too many different geometry types'
    assert pixels.geometry.type.unique()[0] == 'Point', 'wrong geometry type'

    # compare against results from instance method
    pixels_inst = band.get_pixels(vector_features)
    assert (pixels.geometry == pixels_inst.geometry).all(), \
        'pixel geometry must be always the same'
    assert band.band_name in pixels_inst.columns, 'extracted band data not found'

    # try features outside of the extent of the raster
    pixels = Band.read_pixels(
        fpath_raster=fpath_raster,
        vector_features=vector_features_2
    )
    assert (pixels.B1 == 0).all(), 'nodata not set properly to features outside of raster extent'

    # read with full bounding box (no masking just spatial sub-setting)
    band = Band.from_rasterio(
        fpath_raster=fpath_raster,
        band_idx=1,
        band_name_dst='B02',
        vector_features=vector_features,
        full_bounding_box_only=True
    )

    assert not band.is_masked_array, 'data should not be masked'
    assert band.is_ndarray, 'band data should be ndarray'
    assert band.bounds == band_bounds_mask, 'bounds should remain the same regardless of masking'

    mask[100:120,100:200] = False
    band.mask(mask=mask)
    assert band.is_masked_array, 'band must now be a masked array'
    assert not band.values.mask.all(), 'not all pixel should be masked'
    assert band.values.mask.any(), 'some pixels should be masked'

    # try some cases that must fail
    # reading from non-existing file
    # Band.from_rasterio(
    #     fpath_raster='not_existing_file.tif'
    # )
    #
    # reading wrong band index
    # Band.from_rasterio(
    #     fpath_raster=fpath_raster,
    #     band_idx=22,
    # )
    #

    resampled = band.resample(target_resolution=5)

    snap_bounds = band.bounds

    # read data from vector source
    band = Band.from_vector(
        vector_features=vector_features,
        pixres_x=10,
        pixres_y=-10,
        band_name_src='GIS_ID',
        band_name_dst='gis_id',
        snap_bounds=snap_bounds
    )

    assert band.band_name == 'gis_id', 'wrong band name inserted'
    assert band.values.dtype == 'float32', 'wrong data type for values'
    assert band.geo_info.pixres_x == 10, 'wrong pixel size in x direction'
    assert band.geo_info.pixres_y == -10, 'wrong pixel size in y direction'
    assert band.geo_info.epsg == 32632, 'wrong EPSG code'
    assert band.bounds == snap_bounds, 'bounds do not match'

    # without snap bounds
    band = Band.from_vector(
        vector_features=vector_features,
        pixres_x=10,
        pixres_y=-10,
        band_name_src='GIS_ID',
        band_name_dst='gis_id'
    )
    assert band.geo_info.pixres_x == 10, 'wrong pixel size in x direction'
    assert band.geo_info.pixres_y == -10, 'wrong pixel size in y direction'
    assert band.geo_info.epsg == 32632, 'wrong EPSG code'

    # with custom datatype
    band = Band.from_vector(
        vector_features=vector_features,
        pixres_x=10,
        pixres_y=-10,
        band_name_src='GIS_ID',
        band_name_dst='gis_id',
        dtype_src='uint16'
    )
    assert band.values.dtype == 'uint16', 'wrong data type'

    # test with point features
    point_gdf = gpd.read_file(vector_features)
    point_gdf.geometry = point_gdf.geometry.apply(lambda x: x.centroid)

    band_from_points = Band.from_vector(
        vector_features=point_gdf,
        pixres_x=10,
        pixres_y=-10,
        band_name_src='GIS_ID',
        band_name_dst='gis_id',
        snap_bounds=snap_bounds,
        dtype_src='uint16'
    )
    

    from shapely.geometry import Polygon

    epsg = 0
    ulx = 300000
    uly = 5100000
    pixres_x, pixres_y = 10, -10
    
    geo_info = GeoInfo(epsg=epsg, ulx=ulx, uly=uly, pixres_x=pixres_x, pixres_y=pixres_y)

    band_name = 'test'
    values = np.zeros(shape=(2,4))

    band = Band(band_name=band_name, values=values, geo_info=geo_info)
    assert type(band.bounds) == Polygon, 'band bounds must be a Polygon'
    assert not band.has_alias, 'when color name is not set, the band has no alias'
    assert band.band_name == band_name, 'wrong name for band'

    assert band.values[0,0] == 0., 'wrong value for band data'

    assert band.meta['height'] == band.nrows, 'wrong raster height in meta'
    assert band.meta['width'] == band.ncols, 'wrong raster width in meta'
    assert band.is_ndarray, 'must be of type ndarray'
    assert band.crs.is_epsg_code, 'EPSG code not valid'
    

    import zarr

    zarr_values = zarr.zeros((10,10), chunks=(5,5), dtype='float32')
    band = Band(band_name=band_name, values=zarr_values, geo_info=geo_info)
    

