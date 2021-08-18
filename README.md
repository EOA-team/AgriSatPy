# AgriSatPy - Agricultural Satellite Data in Python

AgriSatPy is a lightweight package to organize, explore and process satellite remote sensing data. The development was motivated by the frustration that there are no end-to-end workflows that manage (Sentinel-2) **satellite data** and convert it into formats with which the image data can be used for further **data-driven analysis** with (geo)pandas.

Our focus lies on Sentinel-2 MSI for which we offer special support but other satellite platforms in the optical domain could be added in the future as well.

Because of our background in agronomy and geography, we developed the tools to efficiently answer our research questions. However, we are convinced that the present package can also be useful to other researchers who need to work with satellite data or want to get familiar with the world of remote sensing.

## Capabilities

Currently, we support the following list of tasks:

- extraction of **scene metadata** from Sentinel-2 data in L1C and L2A (Sen2Cor) processing level and compilation of the extracted metadata into a **CSV file that could be used to, for instance, populate a table in a relational database**.


- **spatial resampling of Sentinel-2 data in L1C and L2A processing level** to 10, 20 or 60 meters and creation of a **stacked geoTiff file** containing all selected spectral bands (with band-names). The user can chose from a **list of resampling options provided by rasterio or use our own resampling approach**. This also includes **resampling of the scene-classification layer (SCL)** that comes with the Sen2Cor derived L2A product.


- creation of **an archive structure on the file system for storing the resampled, stacked tiff files + search capabilities** in the created archive.


- **extraction of pixel values from multiple Sentinel-2 scenes for selected subsets of the images** using either the original ESA/Copernicus L1C or L2A data (each spectral band stored separately in the .SAFE folder structure) or the spatially resampled, stacked geoTiffs (see previous bullet item). Returns **pandas dataframes** that could be inserted into a relational database or stored on the file system. This is particularly helpful for **carrying out data science and machine learning tasks without having the need to read entire images and handle very large image data arrays**.


- **merging of scenes split because of data-strip end/beginning issue** using data from one scene to fill the blackfill (i.e., no-data values) of the second one.


- **conversion of extracted pixel data (as pandas dataframes) back to geoTiff image files** using rasterio. Thus, the results of, e.g., a machine learning modeling activity can be visualized as georeferenced raster images.


- the **main processing steps for Sentinel-2 data (L1C or L2A) have been compiled into a processing pipeline** that can be executed in **parallel** including the following steps:

		1. filtering of available ESA/Copernicus derived L1C/L2A data using date range, Sentinel-2 tile and cloud cover thresholds (metadata file must have been created beforehand)
		2. spatial resampling of the selected Sentinel-2 scenes to a user-defined target resolution (e.g., 10 meters) and storage of the data in an easily-accessible archive structure
		3. stacking of split scenes (i.e., filling of blackfill in regions of end/begin of Sentinel-2 data strips)

To come in the **future**:

- implementation of a download client (for Sentinel-2)
- improvement of the archive maintainability (updates, usage of other sensors)
- direct support for PostgreSQL as DBMS for storing and maintaining the extracted scene metadata
- ...



