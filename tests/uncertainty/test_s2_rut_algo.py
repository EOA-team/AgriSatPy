import pytest
from agrisatpy.uncertainty import S2RutAlgo as s2_rut_algo
import numpy as np


def test_simple_case_B8():
    rut_algo = s2_rut_algo()
    rut_algo.a = 6.22865527455779
    rut_algo.e_sun = 1036.39
    rut_algo.u_sun = 1.03418574554466
    rut_algo.tecta = 63.5552301619033
    rut_algo.quant = 10000.0
    rut_algo.alpha = 0.571
    rut_algo.beta = 0.04447
    rut_algo.u_ADC = 0.5

    spacecraft = 'Sentinel-2A'

    band_data = [100, 500, 1000, 2000, 5000, 10000, 15000.]
    rut_result = rut_algo.unc_calculation(
        band_data=np.array(band_data),
        band_id=7,
        spacecraft=spacecraft
    )

    assert [250, 85, 55, 39, 28, 24, 23] == list(rut_result)

def test_simple_case_B2():
    rut_algo = s2_rut_algo()
    rut_algo.a = 6.22865527455779
    rut_algo.e_sun = 1036.39
    rut_algo.u_sun = 1.03418574554466
    rut_algo.tecta = 63.5552301619033
    rut_algo.quant = 10000.0
    rut_algo.alpha = 0.571
    rut_algo.beta = 0.04447
    rut_algo.u_ADC = 0.5

    band_data = [100, 500, 1000, 2000, 5000, 10000, 15000.]
    rut_result = rut_algo.unc_calculation(np.array(band_data), 1)

    assert [250, 96, 61, 42, 31, 26, 25] == list(rut_result)

def test_simple_case_B1():
    rut_algo = s2_rut_algo()
    rut_algo.a = 6.22865527455779
    rut_algo.e_sun = 1036.39
    rut_algo.u_sun = 1.03418574554466
    rut_algo.tecta = 63.5552301619033
    rut_algo.quant = 10000.0
    rut_algo.alpha = 0.571
    rut_algo.beta = 0.04447
    rut_algo.u_ADC = 0.5

    band_data = [100, 500, 1000, 2000, 5000, 10000, 15000.]
    rut_result = rut_algo.unc_calculation(np.array(band_data), 0)

    assert [250, 97, 61, 42, 31, 26, 25] == list(rut_result)


