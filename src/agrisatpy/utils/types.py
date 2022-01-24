'''
Custom types for *AgriSatPy*
'''

from pandas import DataFrame
from typing import Any
from typing import Dict
from typing import NewType


S2Scenes = NewType('S2Scenes', Dict[Any, DataFrame])
