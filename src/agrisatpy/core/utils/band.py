'''
Created on May 1, 2022

@author: graflu
'''

import numpy as np

from numbers import Number
from typing import List, Optional, Union

from agrisatpy.core.band import Band

class BandOperator:
    """
    Band operator supporting basic algebraic operations and Band objects
    """
    operators: List[str] = ['+', '-', '*', '/', '**', '<', '>', '==', '<=', '>=']

    class BandMathError(Exception):
        pass
    
    @classmethod
    def calc(
            cls,
            a: Band,
            other: Union[Number,np.ndarray],
            operator: str,
            inplace: Optional[bool] = False,
            band_name: Optional[str] = None
        ) -> Union[None,np.ndarray]:
        """
        executes a custom algebraic operator on Band objects

        :param a:
            `Band` object with values
        :param other:
            scalar or two-dimemsional `numpy.array` to use on the right-hand
            side of the operator. If a `numpy.array` is passed the array must
            have the same x and y dimensions as the current `Band` data.
        :param operator:
            symbolic representation of the operator (e.g., '+'
            for addition)
        :param inplace:
            returns a new `Band` object if False (default) otherwise overwrites
            the current `Band` data
        :param band_name:
            optional name of the resulting `Band` object if inplace is False.
        :returns:
            `numpy.ndarray` if inplace is False, None instead
        """
        if operator not in cls.operators:
            raise ValueError(f'Unknown operator "{operator}"')
        if isinstance(other, np.ndarray) or isinstance(other, np.ma.MaskedArray):
            if other.shape != a.values.shape:
                raise ValueError(
                    f'Passed array has wrong dimensions. Expected {a.values.shape}' \
                    + f' - Got {other.shape}'
                )
        # perform the operation
        try:
            expr = f'a.values {operator} other'
            res = eval(expr)
        except Exception as e:
            raise cls.BandMathError(f'Could not execute {expr}: {e}')
        # return result or overwrite band data
        if inplace:
            return a.__setattr__('values', res)
        else:
            attrs = dir(a)
            if band_name is not None:
                attrs.update('band_name', band_name)
            return Band(
                values=res,
                **attrs
            )
