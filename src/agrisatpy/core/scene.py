'''
A scene is a collection of raster bands with an acquisition date, an unique identifier
and a (remote sensing) platform that acquired the raster data.
'''

import datetime

from agrisatpy.utils.constants import ProcessingLevels

class SceneProperties(object):
    """
    A class for storing scene-relevant properties

    :attribute acquisition_time:
        image acquisition time
    :attribute platform:
        name of the imaging platform
    :attribute sensor:
        name of the imaging sensor
    :attribute processing_level:
        processing level of the remotely sensed data (if
        known and applicable)
    :attribute scene_id:
        unique scene identifier
    """

    def __init__(
            self, 
            acquisition_time: datetime.datetime = datetime.datetime(2999,1,1),
            platform: str = '',
            sensor: str = '',
            processing_level: ProcessingLevels = ProcessingLevels.UNKNOWN,
            scene_id: str = ''
        ):
        """
        Class constructor

        :param acquisition_time:
            image acquisition time
        :param platform:
            name of the imaging platform
        :param sensor:
            name of the imaging sensor
        :param processing_level:
            processing level of the remotely sensed data (if
            known and applicable)
        :param scene_id:
            unique scene identifier
        """
        # type checking first
        if not isinstance(acquisition_time, datetime.datetime):
            raise TypeError(
                f'A datetime.datetime object is required: {acquisition_time}'
            )
        if not isinstance(platform, str):
            raise TypeError(f'A str object is required: {platform}')
        if not isinstance(sensor, str):
            raise TypeError(f'A str object is required: {sensor}')
        if not isinstance(processing_level, ProcessingLevels):
            raise TypeError(
                f'A ProcessingLevels object is required: {processing_level}'
            )
        if not isinstance(scene_id, str):
            raise TypeError(f'A str object is required: {scene_id}')

        self.acquisition_time = acquisition_time
        self.platform = platform
        self.sensor = sensor
        self.processing_level = processing_level
        self.scene_id = scene_id

    @property
    def acquisition_time(self) -> datetime.datetime:
        """acquisition time of the scene"""
        return self._acquisition_time

    @acquisition_time.setter
    def acquisition_time(self, time: datetime.datetime) -> None:
        """acquisition time of the scene"""
        if not isinstance(time, datetime.datetime):
            raise TypeError('Expected a datetime.datetime object')
        self._acquisition_time = time

    @property
    def platform(self) -> str:
        """name of the imaging platform"""
        return self._platform

    @platform.setter
    def platform(self, value: str) -> None:
        """name of the imaging plaform"""
        if not isinstance(value, str):
            raise TypeError('Expected a str object')
        self._platform = value

    @property
    def sensor(self) -> str:
        """name of the sensor"""
        return self._sensor

    @sensor.setter
    def sensor(self, value: str) -> None:
        """name of the sensor"""
        if not isinstance(value, str):
            raise TypeError('Expected a str object')
        self._sensor = value

    @property
    def processing_level(self) -> ProcessingLevels:
        """current processing level"""
        return self._processsing_level

    @processing_level.setter
    def processing_level(self, value: ProcessingLevels):
        """current processing level"""
        if not isinstance(value, ProcessingLevels):
            raise TypeError(f'Expected {ProcessingLevels}')
        self._processing_level = value

    @property
    def scene_id(self) -> str:
        """unique scene identifier"""
        return self._scene_id

    @scene_id.setter
    def scene_id(self, value: str) -> None:
        """unique scene identifier"""
        if not isinstance(value, str):
            raise TypeError('Expected a str object')
        self._scene_id = value
