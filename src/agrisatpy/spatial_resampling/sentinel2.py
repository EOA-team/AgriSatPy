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
import glob
import rasterio as rio
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio import Affine
import numpy as np
from typing import Optional
from pathlib import Path

from agrisatpy.utils.reprojection import check_aoi_geoms
from agrisatpy.spatial_resampling import upsample_array
from agrisatpy.utils import get_S2_sclfile
from agrisatpy.utils import get_S2_bandfiles_with_res
from agrisatpy.utils import get_S2_tci
from agrisatpy.config import Sentinel2
from agrisatpy.config import get_settings
from pickle import FALSE

Settings = get_settings()
logger = Settings.logger


# global definition of spectral bands and their spatial resolution
s2 = Sentinel2()


def resample_and_stack_S2(
        in_dir: Path,
        out_dir: Path,
        target_resolution: Optional[float]=10.0,
        interpolation: Optional[int]=Resampling.cubic,
        masking: Optional[bool]=False,
        pixel_division: Optional[bool]=False,
        is_L2A: Optional[bool]=True,
        **kwargs
    ) -> Path:
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
    :return:
        filepath to resampled, bandstacked geoTiff file
    '''
    # read in S2 band and SCL file
    resolution_selection = kwargs.get('resolution_selection', [10., 20.])
    s2bands = get_S2_bandfiles_with_res(
        in_dir=in_dir,
        is_L2A=is_L2A,
        resolution_selection=resolution_selection
    )

    # get 10m TCI file
    tci_file = get_S2_tci(
        in_dir=in_dir,
        is_L2A=is_L2A
    )

    # If masking is chosen
    if masking:
        in_file_aoi = kwargs.get("in_file_aoi", str)
        mask = check_aoi_geoms(
            in_file_aoi=in_file_aoi,
            fname_sat=s2bands["band_path"].iloc[0],
            full_bounding_box_only=False
        )

    # get metadata of for a file with the highest resolution (here 10m)
    hires_bands = s2bands[s2bands["band_resolution"] == target_resolution]

    # check interpolation method, in case of pixel_division the spatial resolution must
    # become higher (e.g. from 20 to 10m)
    if pixel_division:
        if (target_resolution > s2bands["band_resolution"]).any():
            raise ValueError(
                'Could not decrease spatial resolution when using pixel_division!'
            )
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
    meta.update(count=s2bands.shape[0])
    meta.update(driver="GTiff")
  
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
    
                logger.info(
                    f'Processing band {s2bands["band_name"].iloc[idx]} ({idx+1}/{s2bands.shape[0]}) from {in_dir}'
                )

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

                    logger.info(
                            f"Completed writing of {s2bands['band_name'].iloc[idx]} from {in_dir}"
                        ) 

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
                        logger.info(
                            f"Completed interpolation of {s2bands['band_name'].iloc[idx]} from {in_dir}"
                        )

        # clip TCI by AOI and write to 8 bit RGB .png
        # read in TCI, clip to AOI and write out as RGB
        logger.info(f'Creating RGB preview from {in_dir}')
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
        logger.info(f'Completed creating RGB preview from {in_dir}')

    # case 2: if no masking is needed, just resample the whole S2 tile
    else:
        with rio.open(os.path.join(out_dir, out_file), "w+", **meta) as dst:
            for idx in range(s2bands.shape[0]):
        
                logger.info(
                    f'Processing band {s2bands["band_name"].iloc[idx]} ({idx+1}/{s2bands.shape[0]}) from {in_dir}'
                )
                
                if s2bands["band_resolution"].iloc[idx] == target_resolution:

                    # nothing to interpolate
                    with rio.open(s2bands["band_path"].iloc[idx]) as src:
                        # band indices begin with 1 in rasterio
                        dst.set_band_description(idx+1, s2bands["band_name"].iloc[idx])
                        dst.write_band(idx+1, src.read(1))

                    logger.info(
                        f"Completed writing of {s2bands['band_name'].iloc[idx]} from {in_dir}"
                    ) 

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
                    logger.info(
                        f"Completed interpolation of {s2bands['band_name'].iloc[idx]} from {in_dir}"
                    ) 

        # read in TCI, reduce to 8 bit and write out as RGB .png
        logger.info(f'Creating RGB preview from {in_dir}')
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
        logger.info(f'Completed creating RGB preview from {in_dir}')

    # delete .aux.xml files in rgb_previews folder
    xml_files = glob.glob(os.path.join(rgb_subdir, '*.xml'))
    for xml_file in xml_files:
        try:
            os.remove(xml_file)
        except Exception as e:
            logger.warning(f'Could not remove {xml_file}: {e}')
            continue

    logger.info(f"file {out_file} written successfully from {in_dir}!")
    return Path(out_dir).joinpath(out_file)


def scl_10m_resampling(
        in_dir: Path,
        out_dir: Path,
        masking: Optional[bool]=False,
        **kwargs
    ) -> Path:
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
        Should masking be applied. The default is False.
    :param kwargs:
        in_file_aoi = filepath to the mask (AOI file).
    :return scl_out_path:
        path to the resampled SCL file
    '''    
    # scl file is resampled after band resampling (see below)
    scl_file = get_S2_sclfile(in_dir)

    # create subfolder for SCL scenes
    scl_subdir = out_dir.joinpath(Settings.SUBDIR_SCL_FILES)
    if not scl_subdir.exists():
        os.mkdir(scl_subdir)
        
    # get unique ID of S2 tile
    s2_uid = os.path.basename(p=str(in_dir))
    # Write out_file
    scl_out_file = s2_uid.split("_")[2].split("T")[0] + "_" + s2_uid.split("_")[-2] + \
        "_" + s2_uid.split("_")[0] + "_SCL_10m.tiff"
    scl_out_path = os.path.join(scl_subdir, scl_out_file)

    logger.info(f'Starting interpolation of SCL file from {in_dir}')
    # read in mask & check CRS
    if masking:
        in_file_aoi = kwargs.get("in_file_aoi", Path)
        mask = check_aoi_geoms(
            in_file_aoi=in_file_aoi,
            fname_sat=scl_file,
            full_bounding_box_only=FALSE
        )

    with rio.open(scl_file) as src:
        meta = src.meta
        
        if masking:
            out_band, out_transform = rio.mask.mask(
                src,
                mask["geometry"],
                crop = True, 
                all_touched = True
            )
        else:
            out_band = src.read()
            out_transform = src.transform
            
        # define new Affine transformation for image writing
        t = Affine(
            10,
            out_transform.b,
            out_transform.c,
            out_transform.d,
            -10,
            out_transform.f
        )

        scaling_factor = 2
        meta.update({"driver": "Gtiff",
                     "height": out_band.shape[1]*scaling_factor,
                     "width": out_band.shape[2]*scaling_factor, 
                     "transform": t})

        # upsample the array spatial resolution by a factor of x
        out_array = upsample_array(
            in_array=out_band[0,:,:],
            scaling_factor=scaling_factor
        )

        with rio.open(scl_out_path, 'w',  **meta) as dst:
            dst.write(out_array, 1)
  
    logger.info(f"file {scl_out_file} written successfully from {in_dir}!")
    return Path(scl_out_path)


# if __name__ == '__main__':
#
#     in_dir = '/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Rawdata/L2A/CH/2018/S2A_MSIL2A_20180816T104021_N0208_R008_T32TLT_20180816T190612'
#     out_dir = '/mnt/ides/Lukas/03_Debug/Sentinel2/L1C/'
#     is_L2A = True
#
#     out_file = resample_and_stack_S2(
#         in_dir=Path(in_dir),
#         out_dir=Path(out_dir),
#         is_L2A=is_L2A
#     )
#
#     if is_L2A:
#         out_file_scl = scl_10m_resampling(
#             in_dir=Path(in_dir),
#             out_dir=Path(out_dir)
#         )
#
#     print(out_file)
