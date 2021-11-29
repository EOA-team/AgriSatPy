'''
Created on Nov 29, 2021

@author:    Lukas Graf (D-USYS, ETHZ)

@purpose:   Collection of exceptions raised by AgriSatPy's modules
'''

class NotProjectedError(Exception):
    pass

class ResamplingFailedError(Exception):
    pass

class BandNotFoundError(Exception):
    pass

class UnknownProcessingLevel(Exception):
    pass

class InputError(Exception):
    pass

class DataNotFoundError(Exception):
    pass

class ArchiveNotFoundError(Exception):
    pass

class MetadataNotFoundError(Exception):
    pass
