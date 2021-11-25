
class Sat_Data_Reader(object):
    """abstract class from which to sensor-specific classes inherit"""

    def __init__(self):
        self.data = {}
        self._from_bandstack = False
