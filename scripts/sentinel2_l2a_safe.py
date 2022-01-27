
from pathlib import Path
from shapely.geometry import Polygon
import geopandas as gpd
from agrisatpy.io.sentinel2 import Sentinel2Handler

# file-path to the .SAFE dataset
dot_safe_dir = Path('../data/S2A_MSIL2A_20190524T101031_N0212_R022_T32UPU_20190524T130304.SAFE')

# construct a bounding box for reading a spatial subset of the scene (geographic coordinates)
ymin, ymax = 47.949, 48.027
xmin, xmax = 11.295, 11.385
bbox = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

# AgriSatPy expects a vector file or a GeoDataFrame for spatial sub-setting
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=4326)

# read data from .SAFE using Sentinel2Handler (reads all 10 and 20m bands + scene classification layer)
handler = Sentinel2Handler()
handler.read_from_safe(
    in_dir=dot_safe_dir,
    polygon_features=bbox_gdf
)

# check the bands read. AgriSatPy converts band names (B02, etc.) to color names
handler.bandnames
# >>> ['blue', 'green', 'red', 'red_edge_1', 'red_edge_2', 'red_edge_3', 'nir_1', 'nir_2', 'swir_1', 'swir_2', 'scl']

# plot false-color infrared preview
fig_nir = handler.plot_false_color_infrared()
fig_nir.savefig('../img/AgriSatPy_Sentinel-2_NIR.png', dpi=150, bbox_inches='tight')

# plot scene classification layer
fig_scl = handler.plot_scl()
fig_scl.savefig('../img/AgriSatPy_Sentinel-2_SCL.png', dpi=150, bbox_inches='tight')

# calculate the NDVI using 10m bands (no spatial resampling required)
handler.calc_si('NDVI')
fig_ndvi = handler.plot_band('NDVI', colormap='summer')
fig_ndvi.savefig('../img/AgriSatPy_Sentinel-2_NDVI.png', dpi=150, bbox_inches='tight')

# mask the water (SCL class 6); requires resampling to 10m spatial resolution
handler.resample(target_resolution=10)
handler.mask(
    name_mask_band='SCL',
    mask_values=[6],
    bands_to_mask=['NDVI']
)
fig_ndvi = handler.plot_band('NDVI', colormap='summer')
fig_ndvi.savefig('../img/AgriSatPy_Sentinel-2_NDVI_masked.png', dpi=150, bbox_inches='tight')
