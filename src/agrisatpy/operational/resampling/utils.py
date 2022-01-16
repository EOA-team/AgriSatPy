'''
Some auxiliary functions for spatial resampling of raster data
'''

import itertools
import numpy as np


def upsample_array(
        in_array: np.array,
        scaling_factor: int,
    ) -> np.array:
    """
    takes a 2-dimensional input array (i.e., image matrix) and splits every
    array cell (i.e., pixel) into X smaller ones having all the same value
    as the "super" cell they belong to, where X is the scaling factor (X>=1).
    This way the input image matrix gets a higher spatial resolution without
    changing any of the original pixel values.

    The value of the scaling_factor determines the spatial resolution of the output.
    If scaling_factor = 1 then the input and the output array are the same.
    If scaling_factor = 2 then the output array has a spatial resolution two times
    higher then the input (e.g. from 20 to 10 m), and so on.

    :param array_in:
        2-d array (image matrix)
    :param scaling_factor:
        factor for increasing spatial resolution. Must be greater than/ equal to 1
    :return out_array:
        upsampled array with pixel values in target spatial resolution
    """

    # check inputs
    if scaling_factor < 1:
        raise ValueError('scaling_factor must be greater/equal 1')

    # define output image matrix array bounds
    shape_out = (in_array.shape[0]*scaling_factor,
                 in_array.shape[1]*scaling_factor)
    out_array = np.zeros(shape_out, dtype = in_array.dtype)

    # increase resolution using itertools by repeating pixel values
    # scaling_factor times
    counter = 0
    for row in range(in_array.shape[0]):
        column = in_array[row, :]
        out_array[counter:counter+scaling_factor,:] = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, scaling_factor) for x in column))
        counter += scaling_factor
    return out_array
