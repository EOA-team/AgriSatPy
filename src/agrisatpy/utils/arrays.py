'''
Created on Dec 10, 2021

@author: Lukas Graf
'''

import numpy as np

from numpy.ma.core import MaskedArray
from typing import Union
from typing import Optional
from agrisatpy.utils.exceptions import InputError

def count_valid(
        in_array: Union[np.array, MaskedArray],
        no_data_value: Optional[Union[int, float]] = 0.
    ) -> int:
    """
    Counts the number of valid (i.e., non no-data) elements in a 2-d
    array. If a masked array is provided, the number of valid elements is
    the number of not-masked array elements.

    :param in_array:
        two-dimensional array to analyze. Can be an ordinary ``numpy.ndarray``
        or a masked array.
    :param no_data_value:
        no data value indicating invalid array elements. Default set to zero.
        Ignored if ``in_array`` is a ``MaskedArray``
    :return:
        number of invalid array elements.
    """

    if len(in_array.shape) > 2:
        raise InputError(
            f'Expected a two-dimensional array, got {len(in_array.shape)} instead.'
        )

    # masked array already has a count method
    if isinstance(in_array, MaskedArray):
        return in_array.count()

    return in_array.count_nonzero(in_array != no_data_value)
    
    