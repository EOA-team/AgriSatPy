'''
Collection of exceptions raised by AgriSatPy's modules
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

class RegionNotFoundError(Exception):
    pass

class ArchiveCreationError(Exception):
    pass

class BlackFillOnlyError(Exception):
    pass

class ReprojectionError(Exception):
    pass

class DataExtractionError(Exception):
    pass

class STACError(Exception):
    pass
