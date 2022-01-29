'''
A band is a two-dimensional array that can be located via a spatial coordinate system.
Each band thus has a name and an array of values, which are usually numeric.

AgriSatPy stores band data basically as ``numpy`` arrays. Masked arrays of the class
`~numpy.ma.MaskedArray` are also supported. For very large data sets that exceed the RAM of the
computer, ``zarr`` can be used.
'''

import numpy as np
import zarr

from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import box
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union


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
        """

        # make sure the passed values are 2-dimensional
        if len(values.shape) != 2:
            raise ValueError('Only two-dimensional arrays are allowed')

        object.__setattr__(self, 'band_name', band_name)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'geo_info', geo_info)
        object.__setattr__(self, 'color_name', color_name)
        object.__setattr__(self, 'wavelength_info', wavelength_info)
        object.__setattr__(self, 'scale', scale)
        object.__setattr__(self, 'offset', offset)

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
        
        
        

if __name__ == '__main__':

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
    

