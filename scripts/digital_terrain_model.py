
from agrisatpy.io import SatDataHandler

# link to cloud-optimized geoTiffs at Swisstopp
dem_file = 'https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2019_2618-1092/swissalti3d_2019_2618-1092_2_2056_5728.tif'

# get SatDataHandler instance
handler = SatDataHandler()

# read the data into the current SatDataHandler instance
handler.read_from_bandstack(
    fname_bandstack=dem_file
)

# we can get the band names (usually starting with B1, B2, etc.) from the handler
band_names = handler.get_bandnames()

# we can rename the default band names by overwriting them
# in this case we rename the single band the file has to "Elevation"
handler.reset_bandnames(['Elevation'])

# we can check the physical unit of the data (meters above mean sea level)
band_unit = handler.get_attrs('Elevation')['units'][0]

# the terrain model has only one band, which we can plot
fig = handler.plot_band(
    band_name='Elevation',
    colormap='terrain',
    colorbar_label=f'Elevation above Mean Sea Level [{band_unit}]'
)

fig.savefig('../img/AgriSatPy_SwissALTI3D_sample.png', dpi=150, bbox_inches='tight')