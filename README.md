# AgriSatPy - Agricultural Satellite Data in Python

AgriSatPy is a lightweight package to explore, organize and process satellite remote sensing data. The development was motivated by the frustration that there are no end-to-end workflows that manage **satellite data** and convert it into formats with which the image data can be used for further **data-driven analysis** with (geo)pandas in Python. 

Our focus lies on Sentinel-2 (S2) MSI data (https://sentinel.esa.int/web/sentinel/missions/sentinel-2) for which we offer special support. Other satellite platforms could be added in the future as well.

Because of our background in agronomy and geography, we developed the tools to efficiently answer our research questions. However, we are convinced that the present package can also be useful to other researchers who need to work with satellite data or want to get familiar with the world of remote sensing.

## Capabilities

Currently, the following list of tasks are implemented:

- **downloading** of Sentinel-2 scenes from [Copernicus](https://scihub.copernicus.eu/) (using [sentinelsat](https://sentinelsat.readthedocs.io/en/stable/)) or [CREODIAS](https://creodias.eu/). CREODIAS is recommended especially when downloading data has been moved to the long-term archive on Copernicus.

- extraction of **scene metadata** from S2 data in L1C and L2A (Sen2Cor) processing level and compilation of the extracted metadata into **a relational PostgreSQL database**. This database is used to make file handling and data filtering much easier.

- **spatial resampling of S2 data at L1C and L2A processing level** to 10, 20 or 60 meters and creation of a **stacked geoTiff file** containing all selected spectral bands (with band-names). The user can chose from a **list of resampling options provided by rasterio** (rasterio.readthedocs.io/) or **use our own resampling approach**. This also includes **resampling of the scene-classification layer (SCL)** that comes with the L2A product (either manually created by using Sen2Cor or direct L2A download).

- creation of **an archive structure** on the file system for storing the resampled and stacked geoTiff files including **file-system search capabilities (in addition to the database)**.

- **extraction of pixel values** from multiple S2 scenes for **selected areas of interest (AOIs)** by supplying Shapefiles of the AOIs: Either directly using the orignal .SAFE files (L1C and/or L2A) obtained from from ESA/Copernicus (where each spectral band is stored separately), or the spatially resampled and stacked geoTiffs generated before (see above). Returns extracted pixel values in a **pandas dataframe** that could be inserted into a relational database or stored on the file system. This is particularly helpful for **carrying out data science and machine learning tasks without having to read entire images and handle very large image data arrays**.

- **merging of split S2 scenes due to ending of S2 datastrip** using data from one scene to fill the blackfill (e.g. no-data values) of the second one.


- **conversion of extracted pixel data (as pandas dataframes) back to geoTiff image files** using rasterio. Thus, the results of a machine learning model can easily be visualized as georeferenced raster images.


- the **main processing steps for S2 data (L1C or L2A) have been compiled into a processing pipeline** that can be executed in **parallel** including the following steps:

		1. filtering of available ESA/Copernicus derived L1C/L2A data using date range, S2 tile and cloud cover thresholds (metadata database must have been created and populated beforehand)
		2. spatial resampling of the selected S2 scenes to a user-defined target resolution (e.g., 10 meters) and storage of the data in an easily-accessible archive structure
		3. merging of split scenes (i.e., filling of blackfill in regions where a S2 datastrip ended)
		4. extraction of pixel values into pandas dataframes (understood by most if not all machine-learning libraries)
		5. conversion of these dataframes into geo-referenced images again (including, e.g., the output of a machine learning model)

To come in the **future**:

- improvement of the archive maintainability (updates, usage of other sensors)
- more tests (yes, they are important :) )

## Code Documentation

We use [sphinx](https://www.sphinx-doc.org/en/master/) and the [autodoc-extension](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) to generate a Read-The-Docs like documentation of our modules, classes and functions.

## Contributing
Yes please :)

Before you start, please:
- familiarize yourself working with `git` if you haven't used it before. Always work in **your own fork** and **never in the master branch**.
- use meaningfull commit messages and try to commit your changes with a certain granularity (e.g, module-wise).
- All changes must be submitted using `merge requests`. The merge request should also have a small message text where you explain what this merge will change.
- make sure to follow the [coding style conventions](./CODE_STYLE.md).
- provide test (pytest) cases whenever possible.

## Versioning Rules

We use [setuptools-scm](https://pypi.org/project/setuptools-scm/) with a post-release scheme for dynamic versioning based on git commit IDs and git tags.

A **release** is a stable version of the package and should be used for operational purposes. Each release should have a version. The versioning system used is `vMajor.Minor (e.g., v1.1)`.
In addition to releases we have `development versions`. This also receive version numbers based on a `post-release` versioning logic. This consists of the latest stable version (i.e., the base for the development), plus the number of the commit since the last release. An example for such a development version is:

```bash
v1.1.post44+gaaa8b16
```

In this example, `v1.1` indicates that v1.1 was the last stable release on which the current development version is based. `post` means that the current version was created *after* this release. `44` is the number of commits since the last release and `gaaa8b16` a shortened version of commit UID.

To create a new release, commit and push all your changes into your development branch and merge them into the main project's master.

Next, checkout the master branch and pull the merged changes locally. Then add a git tag (by increasing the version number):

```bash
git checkout master
git pull upstream master
git tag -a v<major>.<minor> -m "v<major>.<minor>"
git push upstream master
```
Then publish the package to the PyPI package index by running `publish_pypi.sh` or `publish_pypi.bat` depending on your OS. Then checkout your development branch again.

**IMPORTANT**:

- New releases should always be discussed with the other developers.
- Only the package administrator should be able to publish new releases.
- Always add the most important changes to changelog so that it will be possible to track the development between the releases

