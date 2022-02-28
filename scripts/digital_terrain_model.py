
from agrisatpy.core.band import Band

# link to cloud-optimized geoTiffs at Swisstopp
dem_file = 'https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2019_2618-1092/swissalti3d_2019_2618-1092_2_2056_5728.tif'

# load resource into a Band instance and name it "Elevation"
dem = Band.from_rasterio(fpath_raster=dem_file, band_name_dst='Elevation')
print(dem.band_name)

# fast visualization
fig = dem.plot(
    colormap='terrain',
    colorbar_label=f'Elevation above Mean Sea Level [m]'
)

fig.savefig('../img/AgriSatPy_SwissALTI3D_sample.png', dpi=150, bbox_inches='tight')