'''
Created on Sep 20, 2021

@author: Lukas Graf (D-USYS, ETHZ)
'''

import os
import requests
from datetime import date
from shapely.geometry import Polygon
from typing import Optional
import pandas as pd

from agrisatpy.config import get_settings
from agrisatpy.utils.constants.sentinel2 import ProcessingLevels


Settings = get_settings()
logger = Settings.logger

CREODIAS_FINDER_URL = 'https://finder.creodias.eu/resto/api/collections/Sentinel2/search.json?'
CHUNK_SIZE = 2096


def query_creodias(
        start_date: date,
        end_date: date,
        max_records: int,
        processing_level: ProcessingLevels,
        bounding_box: Polygon,
        cloud_cover_threshold: Optional[int]=100
    ) -> pd.DataFrame:
    """
    queries the CREODIAS Finder API to obtain available
    datasets for a given geographic region, date range and
    Sentinel-2 processing level (L1C or L2A).

    NO AUTHENTICATION is required for running this query.

    :param start_date:
        start date of the queried time period (inclusive)
    :param end_date:
        end date of the queried time period (inclusive)
    :param max_records:
        maximum number of items returned. NOTE that
        CREODIAS might limit this number!
    :param processing_level:
        queried Sentinel-2 processing level
    :param bounding_box:
        polygon in geographic coordinates (WGS84) denoting
        the queried region
    :param cloud_cover_threshold:
        cloudy pixel percentage threshold (0-100%) for filtering
        scenes too cloudy for processing. All scenes with a cloud
        cover lower than the threshold specified will be downloaded.
        Per default all scenes are downloaded.
    :return:
        results of the CREODIAS query (no downloaded data!)
        as pandas DataFrame
    """

    # convert dates to strings in the required format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # convert polygon to required format
    coords = bounding_box.exterior.coords.xy
    coord_str = ''
    n_points = len(coords[0])
    for n_point in range(n_points):
        x = coords[0][n_point]
        y = coords[1][n_point]
        coord_str += f'{x}+{y}%2C'

    # get rid of the last %2C
    coord_str = coord_str[:-3]
    # construct the REST query
    query = CREODIAS_FINDER_URL + f'maxRecords={max_records}&'
    query += f'startDate={start_date_str}T00%3A00%3A00Z&completionDate={end_date_str}T23%3A59%3A59Z&'
    query += f'cloudCover=%5B0%2C{cloud_cover_threshold}%5D&'
    query += f'processingLevel={processing_level.value}&'
    query += f'geometry=Polygon(({coord_str}))&'
    query += 'sortParam=startDate&sortOrder=descending&status=all&dataset=ESA-DATASET'

    # GET to CREODIAS Finder API
    res = requests.get(query)
    res.raise_for_status()
    res_json = res.json()

    # extract features (=available datasets)
    features = res_json['features']
    datasets = pd.DataFrame(features)

    # get *.SAFE dataset names
    datasets['dataset_name'] = datasets.properties.apply(
        lambda x: x['productIdentifier'].split('/')[-1]
    )

    return datasets


def get_keycloak() -> str:
    """
    Returns the CREODIAS keycloak token for a valid
    (i.e., registered) CREODIAS user. Takes the username
    and password from either config/settings.py, a .env
    file or environment variables.

    The token is required for downloading data from
    CREODIAS.

    Function taken from:
    https://creodias.eu/-/how-to-generate-keycloak-token-using-web-browser-console-
    (2021-09-23)
    """

    data = {
        "client_id": "CLOUDFERRO_PUBLIC",
        "username": Settings.CREODIAS_USER,
        "password": Settings.CREODIAS_PASSWORD,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://auth.creodias.eu/auth/realms/DIAS/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]


def download_datasets(
        datasets: pd.DataFrame,
        download_dir: str
    ) -> None:
    """
    Function for actual dataset download from CREODIAS.
    Requires valid CREODIAS username and password (to be
    specified in the BaseSettings)

    :param datasets:
        dataframe with results of CREODIAS Finder API request
        made by `query_creodias` function
    :param download_dir:
        directory where to store the downloaded files
    """

    # get API token from CREODIAS
    keycloak_token = get_keycloak()

    # change into download directory
    os.chdir(download_dir)

    # loop over datasets to download them sequentially
    for idx, dataset in datasets.iterrows():
        try:
            dataset_url = dataset.properties['services']['download']['url']
            response = requests.get(
                dataset_url,
                headers={'Authorization': f'Bearer {keycloak_token}'},
                stream=True
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f'Could not download {dataset.product_uri}: {e}')
            continue

        # download the data using the iter_content method (writes chunks to disk)
        fname = dataset.dataset_name.replace('SAFE', 'zip')
        logger.info(f'Starting downloading {fname} ({idx+1}/{datasets.shape[0]})')
        with open(fname, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                fd.write(chunk)
        logger.info(f'Finished downloading {fname} ({idx+1}/{datasets.shape[0]})')
