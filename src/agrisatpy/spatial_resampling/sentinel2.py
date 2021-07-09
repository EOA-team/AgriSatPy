'''
Created on Jul 9, 2021

@author:        Gregor Perich and Lukas Graf (D-USYS, ETHZ)

@purpose:       Spatial resampling of raster images to 10m resolution
                developed for Sentinel2.
                It is also possible to resample from a higher to
                a lower spatial resolution.

                The module can handle Sentinel-2 data in L1C and L2A
                processing level.
'''


import os
from typing import List
import glob
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio import Affine
import numpy as np
import itertools

from agrisatpy.config import Sentinel2
from agrisatpy.config import get_settings

Settings = get_settings()
logger = Settings.get_logger()


# global defintion of spectral bands and their spatial resolution
s2 = Sentinel2()


# %% helper funs
def get_S2_bandfiles(in_dir: str,
                     resolution_selection: List[float]=[10.0, 20.0],
                     search_str: str='*B*.jp2',
                     is_L2A: bool=True
                     ) -> pd.DataFrame:
    '''
    Returns a selection of native resolution Sentinel-2 bands (Def.: 10, 20 m).
    Works on MSIL2A data (sen2core derived) but also allows to work on Sentinel2
    L1C data. In the latter case, the spatial resolution of the single bands is
    hard-coded since the L1C data structure does not allow to extract the spatial
    resolution from the file or directory name.

    :param in_dir:
        Directory where the search_string is applied. Here - for Sentinel-2 - the
        it is the .SAFE directory of S2 rawdata files.
    :param resolution_selection:
        list of Sentinel-2 spatial resolutions to process (Def: [10, 20] m)
    :param search_str:
        search pattern for Sentinel-2 band files
    :param is_L2A:
        if False, assumes that the data is in Sentinel-2 L1C processing level. The
        spatial resolution is then hard-coded for each spectral band. Default is True.
    :returns: pandas dataframe of jp2 files
    '''
    # search for files in subdirectories in case of L2A data
    if is_L2A:
        band_list = [glob.glob(os.path.join(in_dir,f'GRANULE/*/IM*/*{int(x)}*/{search_str}')) \
                     for x in resolution_selection]
    else:
        band_list = []
        for spatial_resolution in s2.SPATIAL_RESOLUTIONS.keys():
            if spatial_resolution not in resolution_selection: continue
            tmp_list = []
            for band_name in s2.SPATIAL_RESOLUTIONS[spatial_resolution]:
                tmp_list.extend(glob.glob(
                    os.path.join(in_dir, f'GRANULE/*/IMG_DATA/*_{band_name}.jp2')))
            band_list.append(tmp_list)
                
    # convert list of list to dictionary using resolutions as keys
    band_dict = dict.fromkeys(resolution_selection)
    for idx, key in enumerate(band_dict.keys()):
        band_dict[key] = band_list[idx]

    # find the highest resolution
    highest_resolution = min(resolution_selection)
    # save the band numbers of those bands with the highest resolution
    if is_L2A:
        hires_bands = [x.split('_')[-2] for x in band_dict[highest_resolution]]
    else:
        hires_bands = [x.split('_')[-1].replace('jp2', '') \
                        for x in band_dict[highest_resolution]]
    
    # loop over the other resolutions and drop the downsampled high resolution
    # bands and keep only the native bands per resolution
    native_bands = band_dict[highest_resolution]
    resolution_per_band = [highest_resolution for x in hires_bands]
    for key in band_dict.keys():

        if key == highest_resolution:
            continue

        if is_L2A:
            lowres_bands = [(x.split('_')[-2], x) for x in band_dict[key]]
        else:
            lowres_bands = [(x.split('_')[-1].replace('.jp2', ''), x) \
                            for x in band_dict[key]]
        lowres_bands2keep = [x[1] for x in lowres_bands if x[0] not in hires_bands]
        native_bands.extend(lowres_bands2keep)
        
        lowres_resolution = [key for x in lowres_bands2keep]
        resolution_per_band.extend(lowres_resolution)

    if is_L2A:
        native_band_names = [x.split('_')[-2] for x in native_bands]
    else:
        native_band_names = [x.split('_')[-1].replace('.jp2','') \
                             for x in native_bands]
    native_band_df = pd.DataFrame(native_band_names, columns=['band_name'])
    native_band_df["band_path"] = native_bands
    native_band_df["band_resolution"] = resolution_per_band
    
    native_band_df = native_band_df.sort_values(by='band_name')

    # Sentinel-2 Band 8A needs "special" treatment in terms of band ordering
    if 'B8A' in native_band_df['band_name'].values:

        tmp_bandnums = [int(x.replace('B','')) for x in list(native_band_df['band_name'].values[0:-1])]
        indices2shift = [i for i,v in enumerate(tmp_bandnums) if v > 8]
        index_b8a = indices2shift[0]
        final_band_df = native_band_df.iloc[0:index_b8a]
        final_band_df = final_band_df.append(native_band_df[native_band_df['band_name']=='B8A'])
        final_band_df = final_band_df.append(native_band_df.iloc[indices2shift])
        return final_band_df
    else:
        return native_band_df


def get_S2_sclfile(in_dir: str
                   ) -> str:
    '''
    returns the path to the S2 SCL (scene classification file) files in 20m resolution.
    Works for L2A processing level, only.

    :param in_dir:
        directory containing the SCL band files (jp2000 file).
    '''
    search_pattern = "/GRANULE/*/IM*/R20m/*_SCL_20m.jp2"
    return glob.glob(in_dir + search_pattern)[0]


def get_S2_tci(in_dir: str,
               is_L2A: bool=True,
               ) -> str:

    '''
    Returns path to S2 TCI (quicklook) img (10m resolution). Works for both
    Sentinel-2 processing levels ('L2A' and 'L1C').

    :param in_dir:
        .SAFE folder which contains Sentinel-2 data
    :param is_L2A:
        if False, it is assumed that the data is organized in L1C .SAFE folder
        structure. The default is True.
    '''
    file_tci = ''
    if is_L2A:
        file_tci = glob.glob(in_dir + '/GRANULE' + '/*/IM*/*10*/*TCI*')[0]
    else:
        file_tci = glob.glob(in_dir + '/GRANULE' + '/*/IM*/*TCI*')[0]
    return file_tci


def upsample_array(in_array: np.array,
                   scaling_factor: int,
                   ) -> np.array:
    """
    takes a 2-dimensional input array (i.e., image matrix) and splits every
    array cell (i.e., pixel) into X smaller ones having all the same value
    as the "super" cell they belong to, where X is the scaling factor (X>=1).
    This way the input image matrix gets a higher spatial resolution without
    changing any of the original pixel values.

    The value of the scaling_factor determines the spatial resolution of the output.
    If scaling_factor = 1 then the input and the output array are the same.
    If scaling_factor = 2 then the output array has a spatial resolution two times
    higher then the input (e.g. from 20 to 10 m), and so on.

    :param array_in:
        2-d array (image matrix)
    :param scaling_factor:
        factor for increasing spatial resolution. Must be greater than/ equal to 1
    """
    # check inputs
    if scaling_factor < 1:
        raise ValueError('scaling_factor must be greater/equal 1')

    # define output image matrix array bounds
    shape_out = (in_array.shape[0]*scaling_factor,
                 in_array.shape[1]*scaling_factor)
    out_array = np.zeros(shape_out, dtype = in_array.dtype)

    # increase resolution using itertools by repeating pixel values
    # scaling_factor times
    counter = 0
    for row in range(in_array.shape[0]):
        column = in_array[row, :]
        out_array[counter:counter+scaling_factor,:] = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, scaling_factor) for x in column))
        counter += scaling_factor
    return out_array


# %% main fun
def resample_and_stack_S2(in_dir: str,
                          out_dir: str,
                          target_resolution: float=10.0,
                          interpolation: int=Resampling.cubic,
                          masking: bool=False,
                          pixel_division: bool=False,
                          is_L2A: bool=True,
                          **kwargs
                         ) -> str:
    '''
    Function to resample S2 scenes and write them to a single stacked .tiff.
    Creates also a RGB preview png-file of each scene. These files are stored
    in a sub-directory ('rgb_previews') within the specified out_dir.
    This function requires the "get_S2_bandfiles" and "get_S2_sclfile" functions!

    For each input 'in_dir', a S2 image stack is written as a .tiff file.
    Returns the full filepath of the written image stack

    :param in_dir:
        path of the .SAFE directory where the S2 data resides.
    :param out_dir:
        path to save the resampled & stacked .tiffs to.
    :param target_resolution:
        target resolution you want to resample to. The default is 10.
    :param interpolation:
        The interpolation algorithm you want to use for upsampling. 
        The defauls is "Resampling.cubic"
        Available options are all of "rasterio.Resample":
        https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling
    :param masking:
        Should the S2 scene be masked or not? The default is False.
        If set to true, an additional arg with a path to an ESRI shapefile or any other vector
        file format supported by geopandas is expected
    :param pixel_division:
        if set to True then pixel values will be divided into n*n subpixels 
        (only even numbers) depending on the target resolution. Takes the 
        current band resolution (for example 20m) and checks against the desired
        target_resolution and applies a scaling_factor. 
        This works, however, only if the spatial resolution is increased, e.g.
        from 20 to 10m. The interpolation argument is ignored then.
        Default value is False
    :param is_L2A:
        boolean flag indicating if the data is in L2A processing level (Default) or L1C.
        Depending on the processing level the struture of the .SAFE products containing the
        satellite data looks slightly different.
    :param kwargs:
        in_file_aoi: file containing the AOI to mask by
        resolution_selection: list of spatial resolution to process (10, 20, 60m)
    '''
    # read in S2 band and SCL file
    resolution_selection = kwargs.get('resolution_selection', [10., 20.])
    s2bands = get_S2_bandfiles(in_dir=in_dir,
                               is_L2A=is_L2A,
                               resolution_selection=resolution_selection)

    # get 10m TCI file
    tci_file = get_S2_tci(in_dir=in_dir,
                          is_L2A=is_L2A)

    # If masking is chosen
    if masking:
        in_file_aoi = kwargs.get("in_file_aoi", str)
        mask = gpd.read_file(in_file_aoi)
        # check CRS of mask
        crs_s2 = rio.open(s2bands["band_path"].iloc[0]).crs
        crs_mask = mask.crs
        # re-project data if needed    
        if crs_s2 != crs_mask:
            logger.warning("CRS mismatch between satellite image and mask layer!")
            logger.warning("Reprojecting mask CRS to S2 CRS.")
            mask = mask.to_crs(crs_s2)
    
    # get metadata of for a file with the highest resolution (here 10m)
    hires_bands = s2bands[s2bands["band_resolution"] == target_resolution]

    # check interpolation method, in case of pixel_division the spatial resolution must
    # become higher (e.g. from 20 to 10m)
    if pixel_division:
        if (target_resolution > s2bands["band_resolution"]).any():
            raise ValueError(
                'Could not decrease spatial resolution when using pixel_division!')
        interpolation = 'Resampling.pixel_division'
    
    # copy metadata of a hires file
    with rio.open(hires_bands["band_path"].iloc[0]) as src:
        meta = src.meta
        
        # in case of masking (clipping), update the meta file to get correct
        # extent of clipped layer. Important for the dataset writer
        if masking:
            out_band, out_transform = rio.mask.mask(src, 
                                                    mask["geometry"], 
                                                    crop = True, 
                                                    all_touched = True
                                                    )
            # check if the masked subset contains blackfill, only
            # check for blackfill
            if out_band.sum() == 0:
                logger.info('Masked area contains blackfill only. Skip scene')
                return ''
            
            meta.update({"driver": "GTiff",
                         "height": out_band.shape[1],
                         "width": out_band.shape[2], 
                         "transform": out_transform
                         })
     
        rows = meta['height']
        cols = meta['width']

    # update the out_meta with the desired number of S2 bands
    meta.update(count = s2bands.shape[0])
    meta.update(driver = "GTiff")
  
    # get S2 UID
    s2_uid = os.path.basename(p=in_dir)
    # keep only Date, Tile, ProcessingLevel & Sensor for the filename_out
    out_date = s2_uid.split("_")[2].split("T")[0] # date
    out_tile = s2_uid.split("_")[-2] # USMGRS tile
    out_level = s2_uid.split("_")[1] # processing level (L1C / L2A)
    out_sensor = s2_uid.split("_")[0] # sensor (S2A / S2B)
    out_method = str(interpolation).split(".")[-1] # resampling method
    out_resolution = str(int(target_resolution)) + "m" # resolution
    out_file = out_date + "_" + out_tile + "_" + out_level + "_" + \
        out_sensor + "_" + out_method + "_" + out_resolution + ".tiff"

    # create sub-folder for RGB preview images
    rgb_subdir = os.path.join(out_dir, "rgb_previews")
    if not os.path.isdir(rgb_subdir):
        os.mkdir(rgb_subdir)
    out_file_rgb = out_date + "_" + out_tile + "_" + out_level + "_" + \
        out_sensor + "_" + out_method + "_" + out_resolution + ".png"

    # case 1: resampling shall be carried on a masked area (e.g. AOI)
    if masking:
        with rio.open(os.path.join(out_dir, out_file), "w+", **meta) as dst:
            
            for idx in range(s2bands.shape[0]):
    
                logger.info(f'Processing band {s2bands["band_name"].iloc[idx]} ({idx+1}/{s2bands.shape[0]})')
                    
                # exception for native bands not to be resampled
                if s2bands["band_resolution"].iloc[idx] == target_resolution:
                    
                    with rio.open(s2bands["band_path"].iloc[idx]) as src:
                        meta = src.meta
                        out_band, out_transform = rio.mask.mask(src,
                                        mask["geometry"],
                                        crop = True, 
                                        all_touched = True,
                                        )

                        meta.update({"height": out_band.shape[1],
                                     "width": out_band.shape[2], 
                                     "transform": out_transform, 
                                     "count": s2bands.shape[0]})

                        # save using write() since it's an array
                        dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
                        dst.write(out_band[0, :, :], idx+1) 

                # resample all other non-native bands
                else:
                    
                    with rio.open(s2bands["band_path"].iloc[idx]) as src:
                        prof = src.profile
                        out_band, out_transform = rio.mask.mask(src,
                                        mask["geometry"],
                                        crop = True, 
                                        all_touched = True,
                                        )
                        prof.update({"driver": "GTiff",
                                     "height": out_band.shape[1],
                                     "width": out_band.shape[2], 
                                     "transform": out_transform})
                        
                        # use rasterio resampling method
                        if not pixel_division:
                            # store masked (clipped) raster in a temporary 'memfile' (memory file)
                            # https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array
                            with MemoryFile() as memfile:
                                with memfile.open(**prof) as dataset:
                                    dataset.write(out_band[0, :, :], 1) # due to height/width mess-up
                                    del out_band

                                # open with rio DatasetReader to perform resampling
                                with memfile.open() as dataset:
                                    sr= dataset.read(out_shape = (rows, cols), 
                                                     resampling = interpolation)
                                    dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
                                    dst.write(sr[0,:,:], idx+1)
                        # or apply pixel division
                        else:
                            scaling_factor = int(out_transform.a / target_resolution)
                            try:
                                out_array = upsample_array(in_array=out_band[0,:,:],
                                                           scaling_factor=scaling_factor)
                            except Exception as e:
                                logger.error(e)
                                return ''
                            dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
                            dst.write(out_array, idx+1)
                        logger.info("interpolation complete")

        # clip TCI by AOI and write to 8 bit RGB .png
        # read in TCI, clip to AOI and write out as RGB
        with rio.open(tci_file) as src:
            meta = src.meta
            out_img, out_transform = rio.mask.mask(src,
                                                   mask["geometry"],
                                                   crop=True,
                                                   all_touched=True,  # IMPORTANT!
                                                   )
            meta.update({"driver": "PNG",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         "dtype": np.uint8}
                        )
            # save using write() since it's an array
            # dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
            with rio.open(os.path.join(rgb_subdir, out_file_rgb), "w+", **meta) as dst:
                for i in range(out_img.shape[0]):
                    dst.write(out_img[i, :, :].astype(np.uint8), i+1)

    # case 2: if no masking is needed, just resample the whole S2 tile
    else: 
        
        with rio.open(os.path.join(out_dir, out_file), "w+", **meta) as dst:
            for idx in range(s2bands.shape[0]):
        
                logger.info(
                    f'Processing band {s2bands["band_name"].iloc[idx]} ({idx+1}/{s2bands.shape[0]})')
                
                if s2bands["band_resolution"].iloc[idx] == target_resolution:
                    
                    with rio.open(s2bands["band_path"].iloc[idx]) as src:
                        # band indices begin with 1 in rasterio
                        dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
                        dst.write_band(idx+1, src.read(1))

                # resample all other non-native bands
                else:
                    with rio.open(s2bands["band_path"].iloc[idx]) as src:
                        # use rasterio resampling method
                        if not pixel_division:
                            sr = src.read(out_shape=(src.count, rows, cols), 
                                          resampling=interpolation)
                            dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
                            dst.write_band(idx+1, sr[0,:,:])
                        else:
                            transform = src.meta['transform']
                            out_band = src.read(1)
                            scaling_factor = int(transform.a / target_resolution)
                            out_array = upsample_array(in_array=out_band[:,:],
                                                       scaling_factor=scaling_factor)
                            dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
                            dst.write(out_array, idx+1)
                    logger.info("interpolation complete")

        # read in TCI, reduce to 8 bit and write out as RGB .png
        with rio.open(tci_file) as src:
            meta = src.meta
            out_img = src.read()
            meta.update({"driver": "PNG",
                         "dtype": np.uint8}
                        )
            # save using write() since it's an array
            # dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
            with rio.open(os.path.join(rgb_subdir, out_file_rgb), "w+", **meta) as dst:
                for i in range(out_img.shape[0]):
                    dst.write(out_img[i, :, :].astype(np.uint8), i+1)

    # delete .aux.xml files in rgb_previews folder
    xml_files = glob.glob(os.path.join(rgb_subdir, '*.xml'))
    for xml_file in xml_files:
        os.remove(xml_file)

    logger.info(f"file {out_file} written successfully!")
    return os.path.join(out_dir, out_file)


def scl_10m_resampling(in_dir: str,
                       out_dir: str,
                       masking: bool = False,
                       **kwargs):
    '''
    Resamples the scene classification layer (SCL) available for L2A Sentinel-2 data.
    Since the SCL is provided in 20 m spatial resolution, this function allows to
    re-sample the data to 10m resolution using  "quadrupling" of the
    pixel values to not mess-up any of the SCL values which are discrete.

    We recognized that the the "JP2OpenJPEG" driver in rasterio writes artifacts!
    Therefore, the "GTiff" driver has to be used.

    Returns the full path to the resampled SCL file
    
    :param in_dir:
        path to .SAFE directory containing the original SCL file.
    :param out_dir:
        Path to the directory you want the SCL_subfolder created. Should be 
        the same as the out_dir of the "resample_and_stack_S2" function.
    :param masking
        SHould masking be applied. The default is False.
    :param kwargs:
        in_file_aoi = filepath to the mask (AOI file).
    '''    
    # scl file is resampled after band resampling (see below)
    scl_file = get_S2_sclfile(in_dir)

    # create subfolder for SCL scenes
    scl_subdir = os.path.join(out_dir, Settings.SUBDIR_SCL_FILES)
    if not os.path.isdir(scl_subdir):
        os.mkdir(scl_subdir)
        
    # get unique ID of S2 tile
    s2_uid = os.path.basename(p=in_dir)
    # Write out_file
    scl_out_file = s2_uid.split("_")[2].split("T")[0] + "_" + s2_uid.split("_")[-2] + \
        "_" + s2_uid.split("_")[0] + "_SCL_10m.tiff"
    scl_out_path = os.path.join(scl_subdir, scl_out_file)
    
    # read in mask & check CRS
    if masking:
        in_file_aoi = kwargs.get("in_file_aoi", str)
        mask = gpd.read_file(in_file_aoi)
        # check CRS of mask
        crs_scl = rio.open(scl_file).crs
        crs_mask = mask.crs
        # reproject if needed    
        if crs_scl != crs_mask:
            logger.warning("CRS mismatch between SCL and mask layer!")
            logger.info("Reprojecting mask CRS to SCL CRS.")
            mask = mask.to_crs(crs_scl)

    with rio.open(scl_file) as src:
        meta = src.meta
        
        if masking:
            out_band, out_transform = rio.mask.mask(src,
                            mask["geometry"],
                            crop = True, 
                            all_touched = True
                            )
        else:
            out_band = src.read()
            out_transform = src.transform
            
        # define new Affine for img writeout
        t = Affine(10, out_transform.b, out_transform.c, out_transform.d, -10, out_transform.f)

        scaling_factor = 2
        meta.update({"driver": "Gtiff",
                     "height": out_band.shape[1]*scaling_factor,
                     "width": out_band.shape[2]*scaling_factor, 
                     "transform": t})

        # upsample the array spatial resolution by a factor of 2
        out_array = upsample_array(in_array=out_band[0,:,:],
                                   scaling_factor=scaling_factor)

        with rio.open(scl_out_path, 'w',  **meta) as dst:
            dst.write(out_array, 1)
  
    logger.info(f"file {scl_out_file} written successfully!")
    return scl_out_path


if __name__ == '__main__':
    
    in_dir = '/home/graflu/public/Evaluation/Projects/KP0022_DeepField/Sentinel-2/S2_L1C_data/CH/CH_2018/PRODUCT/S2A_MSIL1C_20180108T104421_N0206_R008_T31UGP_20180108T124506.SAFE'
    out_dir = '/mnt/ides/Lukas/03_Debug/Sentinel2/L1C/'
    is_L2A = False
    
    out_file = resample_and_stack_S2(in_dir=in_dir,
                                     out_dir=out_dir,
                                     is_L2A=is_L2A)
