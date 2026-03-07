import os
import sys
import json
import time
import argparse
import warnings
import xml.etree.ElementTree as ET
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

import daisy
import dask
from dask.array import coarsen, mean
from dask.diagnostics import ProgressBar
import zarr
from funlib.persistence import Array, prepare_ds, open_ds
from funlib.geometry import Roi, Coordinate
import tifffile
from tqdm import tqdm
from skimage.measure import label, regionprops
from scipy import ndimage

import rtree
from shapely.geometry import Polygon, box

# for xml > zarr
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from skimage.draw import polygon as draw_polygon

cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))

# Import OpenSlide
OPENSLIDE_PATH = os.path.join(grandparent_dir, 'openslide-bin-4.0.0.11-windows-x64\\bin')
print(OPENSLIDE_PATH)
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def opensvs(svs_path, pyramid_level): # from rachel
    """
    open .svs as a dask array and retreive metadata (dimensions + resolution)
    
    INPUTS: 
    - svs_path (str): path to .svs file
    - pyramid_level (int): which pyramid level to open (0 = 40x, 1 = 20x, 2 = 10x, 3 = 5x)
    
    OUTPUTS:
    - dask_array: dask array of the image data for the specified pyramid level
    - x_res: resolution in nm/px in the x dimension
    - y_res: resolution in nm/px in the y dimension
    - units: tuple of units for x and y resolution (should be ("nm", "nm"))
    """
    slide = openslide.OpenSlide(svs_path)
    # grab resolution at micrometeres / px and convert to nm / px
    x_res = float(slide.properties["openslide.mpp-x"]) * 1000
    y_res = float(slide.properties["openslide.mpp-y"]) * 1000
    units = ("nm", "nm")
    store = tifffile.imread(svs_path, aszarr=True)
    dask_array = dask.array.from_zarr(store, pyramid_level)
    store.close()
    return dask_array, x_res, y_res, units

def svs_to_zarr(svs_path, zarr_path, offset, axis_names): # from rachel
    """
    convert H&E from .svs to .zarr file

    INPUTS:
    - svs_path (str): path to .svs file 
    - zarr_path (str): path to save .zarr file
    - offset (tuple): offset for the image data (e.g. (0, 0) if no offset)
    - axis_names (tuple): names of the axes (e.g. ("x", "y", "c"))

    OUTPUTS: 
    - saves .zarr file with the image data for each pyramid level (s0 = 40x, s1 = 20x, s2 = 10x, s3 = 5x) and metadata (voxel size, axis names, units)
    """
    # open highest pyramid level (40x)
    dask_array0, x_res, y_res, units = opensvs(svs_path, 0)
    s0_shape = dask_array0.shape
    # units are natively ("nm", "nm") so no need to convert to get voxel size
    
    # convert to integer and calculate for each pyramid level
    voxel_size0 = Coordinate(int(x_res), int(y_res))
    
    # format data as funlib dataset
    raw = prepare_ds(
        zarr_path / "raw" / "s0",
        dask_array0.shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )
    
    # storage info
    store_rgb = zarr.open(zarr_path / "raw" / "s0") # s0 = full resolution image at 40x magnification
    dask_array = dask_array0.rechunk(raw.data.chunksize)

    with ProgressBar():
        dask.array.store(dask_array, store_rgb)

    for i in range(1, 4): # iterate over each pyramid level: s1 (20x), s2 (10x), s3 (5x)
        # open the image file with openslide for info and tifffile as zarr
        try:
            dask_array, x_res, y_res, _ = opensvs(svs_path, i)
            # units are natively ("nm", "nm") so no need to convert to get voxel size
            
            # convert to integer and calculate for each pyramid level
            voxel_size0 = Coordinate(int(x_res), int(y_res))
            expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i, 3)) # downsampled shape
            print(f"expected shape: {expected_shape}")
            print(f"actual shape: {dask_array.shape}")

            # check shape is expected shape
            if dask_array.shape == expected_shape:
                print("correct shape")
                
                # format data as funlib dataset
                raw = prepare_ds(
                    zarr_path / "raw" / f"s{i}",
                    dask_array.shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                )
                
                # storage info
                store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
                dask_array = dask_array.rechunk(raw.data.chunksize)

                with ProgressBar():
                    dask.array.store(dask_array, store_rgb)

            else:
                voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
                # format data as funlib dataset
                raw = prepare_ds(
                    zarr_path / "raw" / f"s{i}",
                    expected_shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                )
                # storage info
                store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
                prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
                print(f"chunk shape: {prev_layer.chunk_shape}")

                # mean downsampling
                factors = {0: 2, 1: 2}
                try:
                    dask_array = coarsen(mean, prev_layer.data, factors)
                except ValueError as e:
                    new_shape = tuple(
                        (
                            (prev_layer.data.shape[i] // factors[i]) * factors[i]
                            if i in factors
                            else prev_layer.data.shape[i]
                        )
                        for i in range(prev_layer.data.ndim)
                    )
                    dask_array_cropped = prev_layer.data[:new_shape[0], :new_shape[1], :new_shape[2]]
                    dask_array = coarsen(mean, dask_array_cropped, factors)
                # save to zarr
                with ProgressBar():
                    dask.array.store(dask_array, store_rgb)
        
        except TypeError as e: # if it finds an empty pyramid level, it fills it in
            print(f"for layer {i}: {e}")
            print("Generating layer")
            prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
            voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
            # format data as funlib dataset
            raw = prepare_ds(
                zarr_path / "raw" / f"s{i}",
                expected_shape,
                offset,
                voxel_size,
                axis_names,
                units,
                mode="w",
                dtype=np.uint8,
            )
            # storage info
            store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
            prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
            print(f"chunk shape: {prev_layer.chunk_shape}")
            # mean downsampling
            factors = {0: 2, 1: 2}
            try:
                dask_array = coarsen(mean, prev_layer.data, factors)
            except ValueError as e:
                new_shape = tuple(
                    (
                        (prev_layer.data.shape[i] // factors[i]) * factors[i]
                        if i in factors
                        else prev_layer.data.shape[i]
                    )
                    for i in range(prev_layer.data.ndim)
                )
                dask_array_cropped = prev_layer.data[:new_shape[0], :new_shape[1], :new_shape[2]]
                dask_array = coarsen(mean, dask_array_cropped, factors)
            # save to zarr
            with ProgressBar():
                dask.array.store(dask_array, store_rgb)
    return print("svs conversion complete")

def xml_to_semantic_zarr(
    xml_path,
    he_zarr_path,
    axis_names=("y", "x"),
    num_pyramid_levels=4,
    chunk_size=4096,
    class_priority=None,  # optional list like [3,1,2]; earlier = higher priority
):
    """
    Convert Aperio ImageScope XML (with NegativeROA exclusions) to a semantic label Zarr pyramid
    aligned to the H&E Zarr ("raw/s0") spatial metadata.

    INPUTS:
    - xml_path (str): path to the Aperio ImageScope XML file containing annotations
    - he_zarr_path (str): path to the H&E Zarr file (should contain "raw/s0" with correct spatial metadata)
    - axis_names (tuple): names of the spatial axes (default ("y", "x"))
    - num_pyramid_levels (int): number of pyramid levels present (default 4 for s0-s3)
    - chunk_size (int): size of chunks to process at a time when rasterizing (default 4096)
    - class_priority (list, optional): list of class IDs in the order they should be drawn (earlier = higher priority). 

    OUTPUTS: 
    - saves semantic label Zarr pyramid under "labels/s{level}" in the same directory as the H&E Zarr, with labels rasterized according to the XML annotations and aligned to the H&E spatial metadata.
    """

    # read slide metadata from the h&e image zarr
    img_ds0 = open_ds(he_zarr_path / "raw" / "s0")
    voxel_size0 = img_ds0.voxel_size
    offset0 = img_ds0.roi.get_begin()
    shape_yx = (img_ds0.roi.get_shape() / voxel_size0)   # Coordinate
    height, width = int(shape_yx[0]), int(shape_yx[1])
    units = getattr(img_ds0, "units", None)

    # create s0 (40x magnification)
    label_store0 = prepare_ds(
        he_zarr_path / "labels" / "s0",
        (height, width),
        offset0,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    # parse XML into per-class include/exclude polys 
    tree = ET.parse(xml_path)
    root = tree.getroot()

    includes_by_class = {}
    excludes_by_class = {}

    for annotation in root.iter("Annotation"):
        class_id = int(annotation.attrib["Id"])

        for region in annotation.iter("Region"):

            # include or exclude annotation?
            neg = region.attrib.get("NegativeROA", "0") == "1"

            vertices = [(float(v.attrib["X"]), float(v.attrib["Y"])) for v in region.iter("Vertex")]
            
            # make sure polygon is valid/exists
            if len(vertices) < 3:
                continue
            poly = Polygon(vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            
            # append polygon based on exclusion/inclusion by class ID
            if neg:
                excludes_by_class.setdefault(class_id, []).append(poly) 
            else:
                includes_by_class.setdefault(class_id, []).append(poly)

    # final geometry = include - exclude (per class)
    final_geom_by_class = {}
    all_class_ids = set(includes_by_class) | set(excludes_by_class)

    for cid in all_class_ids:
        inc = unary_union(includes_by_class.get(cid, []))
        if inc.is_empty:
            continue
        exc = unary_union(excludes_by_class.get(cid, []))
        geom = inc.difference(exc) if not exc.is_empty else inc
        if not geom.is_empty:
            final_geom_by_class[cid] = geom

    # draw order / priority 
    if class_priority is None:
        draw_order = sorted(final_geom_by_class.keys())
    else:
        # keep only those present; then append any missing classes at end
        present = set(final_geom_by_class.keys())
        draw_order = [c for c in class_priority if c in present] + sorted(present - set(class_priority))

    def iter_polys(geom):
        """Code can only handle Polygons, but union/difference produces MultiPolygons"""
        if geom.is_empty:
            return
        gt = geom.geom_type
        if gt == "Polygon":
            yield geom
        elif gt == "MultiPolygon":
            yield from geom.geoms

    def paint_polygon_with_holes(tile_mask, poly, class_id, y0, x0):
        """Paint polygons without overwriting other classes + correctly clear holes"""
        # paint exterior (background-only so we don't overwrite other classes)
        ext = np.asarray(poly.exterior.coords)
        rr, cc = draw_polygon(ext[:, 1] - y0, ext[:, 0] - x0, tile_mask.shape)
        bg = (tile_mask[rr, cc] == 0)
        tile_mask[rr[bg], cc[bg]] = class_id

        # clear holes, but only where we just painted this class
        for ring in poly.interiors:
            hole = np.asarray(ring.coords)
            rrh, cch = draw_polygon(hole[:, 1] - y0, hole[:, 0] - x0, tile_mask.shape)
            here = (tile_mask[rrh, cch] == class_id)
            tile_mask[rrh[here], cch[here]] = 0

    # rasterize s0 in chunks
    for y0 in range(0, height, chunk_size):
        for x0 in range(0, width, chunk_size):
            y1 = min(y0 + chunk_size, height)
            x1 = min(x0 + chunk_size, width)

            tile_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            tile_box = box(x0, y0, x1, y1)

            # loop through each class
            for cid in draw_order:
                geom = final_geom_by_class[cid]
                if not geom.intersects(tile_box):
                    continue

                clipped = geom.intersection(tile_box)
                if clipped.is_empty:
                    continue

                for poly in iter_polys(clipped):
                    paint_polygon_with_holes(tile_mask, poly, cid, y0, x0)

            # --- old write to mask w/o re-mapping (s0)
            # label_store0[y0:y1, x0:x1] = tile_mask

            # write to mask (s0) with remapped labels
            IGNORE = 255  # background / unannotated
            remapped = np.full(tile_mask.shape, IGNORE, dtype=np.uint8)  # default ignore
            fg = tile_mask > 0 # get non-background pixels
            remapped[fg] = tile_mask[fg] - 1 # remap per-tile: 0->255, 1->0, 2->1, 3->2
            label_store0[y0:y1, x0:x1] = remapped

    print("labels/s0 written")

    # pyramid generation (max-pool downsample)
    prev = dask.array.from_zarr(he_zarr_path / "labels" / "s0")

    for level in range(1, num_pyramid_levels):
        voxel_size = voxel_size0 * (2**level)

        # downsampling by maxpooling annotated pixels
        IGNORE = np.uint8(255)

        def reduce_block(x, axis=None):
            valid = x != IGNORE # boolean array of valid vs. invalid pixels

            # temporarily replace invalid pixels 255-->0 to avoid dominating the max
            x0 = np.where(valid, x, 0).astype(np.uint8)

            # if a block had no valid pixels, set it to IGNORE
            m = x0.max(axis=axis)               # get max label among valid pixels
            any_valid = valid.any(axis=axis)    # get all pixels NOT ignored
            out = np.where(any_valid, m, IGNORE).astype(np.uint8)
            return out

        down = dask.array.coarsen(
            reduce_block,
            prev,
            {0: 2, 1: 2}, # halve the height (0) and width (1)
            trim_excess=True,
        )

        # sanity check
        print(prev.shape, down.shape)
        print(down.dtype)

        level_store = prepare_ds(
            he_zarr_path / "labels" / f"s{level}",
            down.shape,
            offset0,          # mirror image offset (not the 'offset' argument)
            voxel_size,
            axis_names,
            units,
            mode="w",
            dtype=np.uint8,
        )

        with ProgressBar():
            dask.array.store(down, level_store)

        prev = down
        print(f"labels/s{level} written")

    print("Semantic XML → Zarr conversion complete")

def save_npz_daisy(
    img_10x: Array,
    anno_mask_10x: Array,
    slidename: str,
    output_dir: str
):
    """
    Uses daisy to save 224x224 pixel blocks as a .npz from annotated H&E WSIs (.zarr)

    INPUTS:
    - img_10x (Array): funlib Array for the 10x magnification layer of the H&E image
    - anno_mask_10x (Array): funlib Array for the 10x magnification layer of the annotation mask (should be aligned with img_10x)
    - slidename (str): H&E slide name (used for naming output .npz files)
    - output_dir (str): directory to save output .npz files

    OUTPUTS: 
    - saves .npz files containing 'image' and 'label' arrays for each 224x224 block with annotations, named as {slidename}_tile_y{y}_x{x}.npz
        """
    # convert annotation mask 
    def save_npz_block(block: daisy.Block):
        # Clip the read ROI to the actual array bounds
        read_roi = block.read_roi.intersect(img_10x.roi)
        
        # If the intersection is empty, skip this block
        if read_roi.empty:
            # print(f"Skipping block at {block.read_roi} - outside array bounds")
            return
     
        # Get the corresponding y, x coordinates from the original block for naming
        begin = block.read_roi.get_begin()
        y, x = int(begin[-2]), int(begin[-1])

        # Reading image data + annotation mask
        img_data = img_10x[read_roi] 
        anno_mask_data = anno_mask_10x[read_roi] 

        # skip unannotated blocks
        IGNORE = 255
        if not np.any(anno_mask_data != IGNORE):
            return None
        
        # exclude patches where annotated pixels < 1% of the patch
        valid = (anno_mask_data != IGNORE)
        valid_frac = valid.mean()
        if valid_frac < 0.01:
            return None

        # save + format both the img and anno_mask into a .npz using naming convention
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{slidename}_tile_y{y}_x{x}.npz')

        np.savez(
                filename,
            image=img_data.astype(np.float32),
            label=anno_mask_data.astype(np.uint8)
        )

    # convert pixels to nm (world units)
    block_roi = Roi((0, 0), (224, 224)) * img_10x.voxel_size
    
    # configure daisy task
    save_npz_task = daisy.Task(
        "blockwise_save",
        total_roi=img_10x.roi, # dictates the total size for both annotations + img
        read_roi=block_roi,
        write_roi=block_roi,
        read_write_conflict=False,
        num_workers=2,
        process_function=save_npz_block,
        fit='overhang' # 'valid': skips border tiles, 'overhang': creates blocks that hang over the edge
    )

    print("IMG voxel_size:", img_10x.voxel_size)
    print("IMG roi      :", img_10x.roi)

    print("MASK voxel_size:", anno_mask_10x.voxel_size)
    print("MASK roi      :", anno_mask_10x.roi)
    
    # running the daisy task
    daisy.run_blockwise(tasks=[save_npz_task], multiprocessing=False)
    return

def load_zarr_level(zarr_path, level, roi_type='raw'):
    """
    Load a specific pyramid level from your SVS Zarr file.
    
    INPUTS:
        zarr_path: Path to the zarr file/group
        level: Pyramid level (0=40x, 1=20x, 2=10x, 3=5x)
    
    OUTPUTS:
        funlib.persistence.Array
    """

    return open_ds(zarr_path / roi_type / f"s{level}")

def extracting_annotations(slide_dir, training_patches_dir, level):
    ''' 
    General Function executing all tasks above to convert H&E image with annotations and save to .npz tiles

    INPUTS:
    - slide_dir(str): path to H&E images
    - training_patches_dir(str): path to destination of annotated patches
    - level: pyramid level (s[0,1,2,3] corresponding to 40x, 20x, 10x, and 5x)
    '''
    for slide in slide_dir.glob("*.svs"):
        slidename = slide.stem
        svs_path = slide
        zarr_path = slide_dir / f"{slidename}.zarr"
        xml_path = slide_dir / f"{slidename}.xml"
        output_dir = training_patches_dir / slidename
        output_dir.mkdir(parents=True,exist_ok=True)
        print(f'working on {slidename}!')
    
    # --- convert .svs to .zarr
    print('converting .svs to .zarr')
    offset = Coordinate(0,0)
    axisnames = ['y','x','c^']
    svs_to_zarr(svs_path,
                zarr_path,
                offset,
                axisnames)
    
    # --- convert .xml to zarr
    print('converting .xml to .zarr')
    xml_to_semantic_zarr(xml_path,
                         zarr_path)
    
    # --- extracting training annotations as .npz files
    print('extracting training annotations')
    raw_arr = load_zarr_level(zarr_path, level, roi_type='raw')
    labels_array = load_zarr_level(zarr_path, level, roi_type='labels')
    save_npz_daisy(raw_arr,
                    labels_array,
                    slidename,
                    output_dir)
    return print('successfully extracted and saved all training annotations!')