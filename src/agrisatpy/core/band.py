'''
A band is a two-dimensional array that can be located via a spatial coordinate system.
Each band thus has a name and an array of values, which are usually numeric.

AgriSatPy stores band data basically as ``numpy`` arrays. Masked arrays of the class
`~numpy.ma.MaskedArray` are also supported. For very large data sets that exceed the RAM of the
computer, ``zarr`` can be used.
'''

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.mask
import zarr

from pathlib import Path
from rasterio import Affine
from rasterio import features
from rasterio.crs import CRS
from shapely.geometry import box
from shapely.geometry import Polygon
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from agrisatpy.core.utils.geometry import check_geometry_types
from agrisatpy.core.utils.geometry import convert_3D_2D
from agrisatpy.core.utils.raster import get_raster_attributes
from agrisatpy.utils.reprojection import check_aoi_geoms
from agrisatpy.utils.exceptions import BandNotFoundError, DataExtractionError


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
            nodata: Optional[Union[int,float]] = None
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
        shift = 0
        x, _ = transform * (np.arange(nx) + shift, np.zeros(nx) + shift)
        _, y = transform * (np.zeros(ny) + shift, np.arange(ny) + shift)

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
        return isinstance(self.values, np.ndarray)

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

    def plot(self):
        """
        Plots the raster values using ``matplotlib``
        """
        pass

    def resample(self):
        """
        Changes the raster grid cell (pixel) size
        """
        pass

    def reproject(self):
        """
        Projects the raster data into a different spatial coordinate system
        """
        pass

    def reduce(self):
        """
        Reduces the raster data to a scalar
        """
        pass

    def to_dataframe(self):
        """
        Returns a ``GeoDataFrame`` from the raster band data
        """

    def to_xarray(self):
        """
        Returns a ``xarray.Dataset`` from the raster band data¨
        """

        

if __name__ == '__main__':

    # read data from raster
    fpath_raster = Path('../../../data/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    vector_features = Path('../../../data/sample_polygons/ZH_Polygons_2020_ESCH_EPSG32632.shp')

    band = Band.from_rasterio(
        fpath_raster=fpath_raster,
        band_idx=1,
        band_name_dst='B02',
        vector_features=vector_features,
        full_bounding_box_only=False
    )

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
    

