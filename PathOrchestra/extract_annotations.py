import os
import sys
import json
import time
import argparse
import warnings
import xml.etree.ElementTree as ET
import logging
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
OPENSLIDE_PATH = os.path.join(parent_dir, 'openslide-bin-4.0.0.11-windows-x64\\bin')
print(OPENSLIDE_PATH)
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def opensvs(svs_path, pyramid_level):
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
    x_res = float(slide.properties["openslide.mpp-x"]) * 1000
    y_res = float(slide.properties["openslide.mpp-y"]) * 1000
    units = ("nm", "nm")
    store = tifffile.imread(svs_path, aszarr=True)
    dask_array = dask.array.from_zarr(store, pyramid_level)
    store.close()
    return dask_array, x_res, y_res, units


def svs_to_zarr(svs_path, zarr_path, offset, axis_names):
    """
    Convert H&E from .svs to .zarr file (all string paths)

    INPUTS:
    - svs_path (str): path to .svs file 
    - zarr_path (str): path to save .zarr file
    - offset (tuple): offset for the image data (e.g. (0, 0) if no offset)
    - axis_names (tuple): names of the axes (e.g. ("y", "x"))

    OUTPUTS: 
    - saves .zarr file with the image data for each pyramid level (s0 = 40x, s1 = 20x, s2 = 10x, s3 = 5x)
    """
    # Ensure string paths
    svs_path = str(svs_path)
    zarr_path = str(zarr_path)
    
    dask_array0, x_res, y_res, units = opensvs(svs_path, 0)
    s0_shape = dask_array0.shape
    voxel_size0 = Coordinate(int(x_res), int(y_res))
    
    # format data as funlib dataset (use f-string paths)
    raw = prepare_ds(
        f"{zarr_path}/raw/s0",
        dask_array0.shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )
    
    store_rgb = zarr.open(f"{zarr_path}/raw/s0")
    dask_array = dask_array0.rechunk(raw.data.chunksize)

    with ProgressBar():
        dask.array.store(dask_array, store_rgb, num_workers=8)

    for i in range(1, 4):
        try:
            dask_array, x_res, y_res, _ = opensvs(svs_path, i)
            voxel_size0 = Coordinate(int(x_res), int(y_res))
            expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i, 3))
            print(f"expected shape: {expected_shape}")
            print(f"actual shape: {dask_array.shape}")

            if dask_array.shape == expected_shape:
                print("correct shape")
                
                raw = prepare_ds(
                    f"{zarr_path}/raw/s{i}",
                    dask_array.shape,
                    offset,
                    voxel_size0,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                )
                
                store_rgb = zarr.open(f"{zarr_path}/raw/s{i}")
                dask_array = dask_array.rechunk(raw.data.chunksize)

                with ProgressBar():
                    dask.array.store(dask_array, store_rgb, num_workers=8)

            else:
                voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
                raw = prepare_ds(
                    f"{zarr_path}/raw/s{i}",
                    expected_shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                )
                
                store_rgb = zarr.open(f"{zarr_path}/raw/s{i}")
                prev_layer = open_ds(f"{zarr_path}/raw/s{i-1}")
                print(f"chunk shape: {prev_layer.chunk_shape}")

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
                
                with ProgressBar():
                    dask.array.store(dask_array, store_rgb, num_workers=8)

        except TypeError as e:
            print(f"for layer {i}: {e}")
            print("Generating layer")
            prev_layer = open_ds(f"{zarr_path}/raw/s{i-1}")
            voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
            expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i, 3))
            
            raw = prepare_ds(
                f"{zarr_path}/raw/s{i}",
                expected_shape,
                offset,
                voxel_size,
                axis_names,
                units,
                mode="w",
                dtype=np.uint8,
            )
            
            store_rgb = zarr.open(f"{zarr_path}/raw/s{i}")
            prev_layer = open_ds(f"{zarr_path}/raw/s{i-1}")
            print(f"chunk shape: {prev_layer.chunk_shape}")

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
            
            with ProgressBar():
                dask.array.store(dask_array, store_rgb, num_workers=8)
    
    return print("svs conversion complete")


def xml_to_semantic_zarr(
    xml_path,
    he_zarr_path,
    axis_names=("y", "x"),
    num_pyramid_levels=4,
    chunk_size=4096,
    class_priority=None,
    class_map=None,
):
    """
    Convert Aperio ImageScope XML to semantic label Zarr pyramid (all string paths).
    """
    # Ensure string paths
    xml_path = str(xml_path)
    he_zarr_path = str(he_zarr_path)
    
    if class_map is None:
        class_map = {}

    img_ds0 = open_ds(f"{he_zarr_path}/raw/s0")
    voxel_size0 = img_ds0.voxel_size
    offset0 = img_ds0.roi.get_begin()
    shape_yx = (img_ds0.roi.get_shape() / voxel_size0)
    height, width = int(shape_yx[0]), int(shape_yx[1])
    units = getattr(img_ds0, "units", None)

    label_store0 = prepare_ds(
        f"{he_zarr_path}/labels/s0",
        (height, width),
        offset0,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    includes_by_class = {}
    excludes_by_class = {}

    for annotation in root.iter("Annotation"):
        class_id = int(annotation.attrib["Id"])

        for region in annotation.iter("Region"):
            neg = region.attrib.get("NegativeROA", "0") == "1"
            vertices = [(float(v.attrib["X"]), float(v.attrib["Y"])) for v in region.iter("Vertex")]
            
            if len(vertices) < 3:
                continue
            poly = Polygon(vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            
            if neg:
                excludes_by_class.setdefault(class_id, []).append(poly) 
            else:
                includes_by_class.setdefault(class_id, []).append(poly)

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

    # Apply class mapping
    if class_map:
        mapped_geom = {}
        for orig_cid, geom in final_geom_by_class.items():
            if orig_cid not in class_map:
                continue
            new_cid = class_map[orig_cid]
            if new_cid not in mapped_geom:
                mapped_geom[new_cid] = []
            mapped_geom[new_cid].append(geom)
        
        final_geom_by_class = {
            cid: unary_union(geoms) 
            for cid, geoms in mapped_geom.items()
        }

    if class_priority is None:
        draw_order = sorted(final_geom_by_class.keys())
    else:
        present = set(final_geom_by_class.keys())
        draw_order = [c for c in class_priority if c in present] + sorted(present - set(class_priority))

    def iter_polys(geom):
        if geom.is_empty:
            return
        gt = geom.geom_type
        if gt == "Polygon":
            yield geom
        elif gt == "MultiPolygon":
            yield from geom.geoms

    def paint_polygon_with_holes(tile_mask, poly, class_id, y0, x0):
        """Paint polygons without overwriting other classes + correctly clear holes"""
        # paint exterior
        ext = np.asarray(poly.exterior.coords)
        rr, cc = draw_polygon(ext[:, 1] - y0, ext[:, 0] - x0, tile_mask.shape)
        bg = (tile_mask[rr, cc] == 0)
        tile_mask[rr[bg], cc[bg]] = class_id + 1  # ← ADD 1 HERE (paint 1-indexed)

        # clear holes
        for ring in poly.interiors:
            hole = np.asarray(ring.coords)
            rrh, cch = draw_polygon(hole[:, 1] - y0, hole[:, 0] - x0, tile_mask.shape)
            here = (tile_mask[rrh, cch] == class_id + 1)  # ← AND HERE
            tile_mask[rrh[here], cch[here]] = 0

    for y0 in range(0, height, chunk_size):
        for x0 in range(0, width, chunk_size):
            y1 = min(y0 + chunk_size, height)
            x1 = min(x0 + chunk_size, width)

            tile_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            tile_box = box(x0, y0, x1, y1)

            for cid in draw_order:
                geom = final_geom_by_class[cid]
                if not geom.intersects(tile_box):
                    continue

                clipped = geom.intersection(tile_box)
                if clipped.is_empty:
                    continue

                for poly in iter_polys(clipped):
                    paint_polygon_with_holes(tile_mask, poly, cid, y0, x0)

            IGNORE = 255
            remapped = np.full(tile_mask.shape, IGNORE, dtype=np.uint8)
            fg = tile_mask > 0
            remapped[fg] = tile_mask[fg] - 1
            label_store0[y0:y1, x0:x1] = remapped

    print("labels/s0 written")

    prev = label_store0.data

    for level in range(1, num_pyramid_levels):
        voxel_size = voxel_size0 * (2**level)
        IGNORE = np.uint8(255)

        def reduce_block(x, axis=None):
            valid = x != IGNORE
            x0 = np.where(valid, x, 0).astype(np.uint8)
            m = x0.max(axis=axis)
            any_valid = valid.any(axis=axis)
            out = np.where(any_valid, m, IGNORE).astype(np.uint8)
            return out

        down = dask.array.coarsen(
            reduce_block,
            prev,
            {0: 2, 1: 2},
            trim_excess=True,
        )

        print(prev.shape, down.shape)
        print(down.dtype)

        level_store = prepare_ds(
            f"{he_zarr_path}/labels/s{level}",
            down.shape,
            offset0,
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


class Rectangle:
    """Define rectangle coordinates & overlapping conditions."""
    def __init__(self, x1, y1, x2, y2):
        self.left = x1
        self.top = y1
        self.right = x2
        self.bottom = y2

    def overlaps(self, other):
        """Return TRUE if overlaps, FALSE if not overlaps"""
        return not (self.right < other.left or
                    self.left > other.right or
                    self.bottom < other.top or
                    self.top > other.bottom)


def parse_xml_regions_no_union(xml_path):
    """
    Parse XML and return ALL individual regions (not grouped by annotation ID, not unioned).
    Only includes positive regions (excludes NegativeROA polygons).
    
    OUTPUTS:
    - regions (list): List of tuples (class_id, Polygon)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    regions = []
    
    for annotation in root.iter("Annotation"):
        class_id = int(annotation.attrib["Id"])
        
        for region in annotation.iter("Region"):
            # Skip exclusion regions (NegativeROA)
            neg = region.attrib.get("NegativeROA", "0") == "1"
            if neg:
                continue
            
            # Parse vertices
            vertices = [(float(v.attrib["X"]), float(v.attrib["Y"])) for v in region.iter("Vertex")]
            
            # Validate polygon
            if len(vertices) < 3:
                continue
            
            poly = Polygon(vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            
            regions.append((class_id, poly))
    
    return regions


def get_bounding_boxes_grouped_by_coords(regions):
    """
    Extract bounding boxes from regions and group by bbox coordinates.
    
    INPUTS:
    - regions (list): List of tuples (class_id, Polygon)
    
    OUTPUTS:
    - bbox_dict (dict): {(minx, miny, maxx, maxy): [(class_id, Polygon), ...]}
    - rectangles (list): List of Rectangle objects with bbox coords
    """
    bbox_dict = {}  # {bbox_coords: [(class_id, poly), ...]}
    rectangles = []
    
    for class_id, poly in regions:
        minx, miny, maxx, maxy = poly.bounds
        bbox_coords = (int(minx), int(miny), int(maxx), int(maxy))
        
        # Group by coordinates
        if bbox_coords not in bbox_dict:
            bbox_dict[bbox_coords] = []
        bbox_dict[bbox_coords].append((class_id, poly))
        
        # Also create Rectangle for overlap detection
        rect = Rectangle(int(minx), int(miny), int(maxx), int(maxy))
        rectangles.append((bbox_coords, rect))
    
    return bbox_dict, rectangles


def get_all_bboxes_from_xml(xml_path):
    """
    Complete pipeline: parse XML, extract all individual region bboxes, group by coordinates.
    
    OUTPUTS:
    - bbox_dict (dict): {(minx, miny, maxx, maxy): [(class_id, Polygon), ...]}
    - bbox_list (list): Sorted list of unique bbox coordinates
    """
    regions = parse_xml_regions_no_union(xml_path)
    bbox_dict, _ = get_bounding_boxes_grouped_by_coords(regions)
    bbox_list = sorted(list(bbox_dict.keys()))
    
    return bbox_dict, bbox_list


def group_overlapping_rectangles(rectangles):
    """
    Group overlapping rectangles using DFS.
    
    INPUTS:
    - rectangles (list): List of Rectangle objects
    
    OUTPUTS:
    - groups (list): List of groups, each group is a list of rectangle indices
    """
    n = len(rectangles)
    adjacency_list = {i: [] for i in range(n)}
    
    # Build adjacency list
    for i in range(n):
        for j in range(i + 1, n):
            if rectangles[i].overlaps(rectangles[j]):
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
    
    # DFS to find connected components
    visited = [False] * n
    groups = []
    
    def dfs(i, group):
        visited[i] = True
        group.append(i)
        for neighbor in adjacency_list[i]:
            if not visited[neighbor]:
                dfs(neighbor, group)
    
    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)
    
    return groups


def save_npz_daisy(
    img_arr,
    anno_mask_arr,
    slidename: str,
    output_dir: str,
    xml_path: str,
    patch_size: int = 224,
    overlap: int = 0,
    level: int = 0,
):
    """
    Uses daisy to save patches as .npz from annotated regions only.
    Extracts patches per unique bounding box (grouped by coordinates, not annotation ID).

    INPUTS:
    - img_arr (Array): funlib Array for H&E image at the desired level
    - anno_mask_arr (Array): funlib Array for the annotation mask at the desired level
    - slidename (str): H&E slide name (used for naming output .npz files)
    - output_dir (str): directory to save output .npz files
    - xml_path (str): path to XML annotation file
    - patch_size (int): size of square patches in pixels (default 224)
    - overlap (int): overlap between patches in pixels (default 0)
    - level (int): pyramid level (0=40x, 1=20x, 2=10x, etc.). ADD THIS

    OUTPUTS: 
    - saves .npz files for annotated regions only
    """
    root_logger = logging.getLogger()
    handlers_to_remove = [
        h for h in root_logger.handlers 
        if isinstance(h, logging.StreamHandler)
    ]
    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)
    
    logging.getLogger('daisy').setLevel(logging.WARNING)

    print("Parsing XML bounding boxes...")
    bbox_dict, bbox_list = get_all_bboxes_from_xml(xml_path)
    
    if not bbox_dict:
        print("No annotations found in XML!")
        return
    
    print(f"Found {len(bbox_dict)} unique bounding box(es)")
    
    # SCALE COORDINATES BY PYRAMID LEVEL
    scale_factor = 2 ** level
    print(f"Scaling coordinates by 2^{level} = {scale_factor} (level {level})")
    
    rectangles = []
    for bbox_coords in bbox_list:
        minx, miny, maxx, maxy = bbox_coords
        # Divide by scale factor to convert from level 0 to level N
        minx, miny, maxx, maxy = minx // scale_factor, miny // scale_factor, maxx // scale_factor, maxy // scale_factor
        rect = Rectangle(int(minx), int(miny), int(maxx), int(maxy))
        rectangles.append(rect)
    
    groups = group_overlapping_rectangles(rectangles)
    print(f"Grouped into {len(groups)} region(s)")
    
    def save_npz_block(block: daisy.Block):
        read_roi = block.read_roi.intersect(img_arr.roi)
        if read_roi.empty:
            return
        
        begin = block.read_roi.get_begin()
        y, x = int(begin[-2]), int(begin[-1])

        img_data = img_arr[read_roi]
        anno_mask_data = anno_mask_arr[read_roi]

        IGNORE = 255
        if not np.any(anno_mask_data != IGNORE):
            return None
        
        valid = (anno_mask_data != IGNORE)
        valid_frac = valid.mean()
        if valid_frac < 0.01:
            return None

        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{slidename}_tile_y{y}_x{x}.npz')
        np.savez(
            filename,
            image=img_data.astype(np.float32),
            label=anno_mask_data.astype(np.uint8)
        )
    
    stride = patch_size - overlap
    write_roi = Roi((0, 0), (stride, stride)) * anno_mask_arr.voxel_size
    read_roi = Roi((0, 0), (patch_size, patch_size)) * anno_mask_arr.voxel_size
    
    for group_idx, group in enumerate(groups):
        print(f"\nProcessing group {group_idx + 1}/{len(groups)}...")
        
        group_rects = [rectangles[i] for i in group]
        y_min = min(r.top for r in group_rects)
        y_max = max(r.bottom for r in group_rects)
        x_min = min(r.left for r in group_rects)
        x_max = max(r.right for r in group_rects)
        
        padding = patch_size
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(anno_mask_arr.roi.get_shape()[0], y_max + padding)
        x_max = min(anno_mask_arr.roi.get_shape()[1], x_max + padding)
        
        print(f"  Bounding box: y=[{y_min}, {y_max}], x=[{x_min}, {x_max}] ({y_max-y_min} × {x_max-x_min} px)")
        
        y_min_nm = int(y_min * anno_mask_arr.voxel_size[0])
        x_min_nm = int(x_min * anno_mask_arr.voxel_size[1])
        height_nm = int((y_max - y_min) * anno_mask_arr.voxel_size[0])
        width_nm = int((x_max - x_min) * anno_mask_arr.voxel_size[1])
        
        roi_begin = Coordinate(y_min_nm, x_min_nm)
        roi_shape = Coordinate(height_nm, width_nm)
        group_roi = Roi(roi_begin, roi_shape)
        
        save_npz_task = daisy.Task(
            f"blockwise_save_group{group_idx}",
            total_roi=group_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            read_write_conflict=False,
            num_workers=8,
            process_function=save_npz_block,
            fit='overhang'
        )
        
        print(f"  Extracting {patch_size}x{patch_size} patches (stride={stride}px)")
        daisy.run_blockwise(tasks=[save_npz_task], multiprocessing=False)
    
    print(f"\n✓ Extraction complete!")
    return


def load_zarr_level(zarr_path, level, roi_type='raw'):
    """Load a specific pyramid level from your SVS Zarr file."""
    zarr_path = str(zarr_path)
    return open_ds(f"{zarr_path}/{roi_type}/s{level}")


def extracting_annotations(
        slide_dir, 
        training_patches_dir, 
        level, 
        patch_size=224, 
        overlap=0, 
        class_map=None):
    ''' 
    General Function executing all tasks to convert H&E image with annotations and save to .npz tiles
    '''
    # Convert all paths to strings ONCE at start
    slide_dir = str(slide_dir)
    training_patches_dir = str(training_patches_dir)
    
    for slide in Path(slide_dir).glob("*.svs"):
        slidename = slide.stem
        svs_path = str(slide)
        zarr_path = f"{slide_dir}/{slidename}.zarr"
        xml_path = f"{slide_dir}/{slidename}.xml"
        output_dir = f"{training_patches_dir}/{slidename}"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f'Working on {slidename}!')
        
        # Check if XML exists
        if not Path(xml_path).exists():
            print(f"Skipping {slidename} - no .xml found at {xml_path}")
            continue

        # SKIP SVS→Zarr if raw/s0 already exists
        if not Path(f"{zarr_path}/raw/s0").exists():
            print('Converting .svs to .zarr')
            offset = Coordinate(0, 0)
            axisnames = ['y', 'x', 'c^']
            svs_to_zarr(svs_path, zarr_path, offset, axisnames)
        else:
            print(f'✓ Skipping .svs→.zarr (already exists)')
        
        # SKIP XML→Zarr if labels/s0 already exists
        if not Path(f"{zarr_path}/labels/s0").exists():
            print('Converting .xml to .zarr')
            xml_to_semantic_zarr(xml_path, zarr_path, class_map=class_map)
        else:
            print(f'✓ Skipping .xml→.zarr (already exists)')

        # SKIP if training patches exist
        npz_files = list(Path(output_dir).glob("*.npz"))
        if len(npz_files) == 0:
            # --- extracting training annotations as .npz files
            print(f'Extracting training annotations at level {level}')
            raw_arr = load_zarr_level(zarr_path, level, roi_type='raw')
            labels_array = load_zarr_level(zarr_path, level, roi_type='labels')
            save_npz_daisy(
                raw_arr,
                labels_array,
                slidename,
                output_dir,
                xml_path=xml_path,
                patch_size=patch_size,
                overlap=overlap,
                level=level,
            )
            print(f'✓ Successfully processed {slidename}!\n')
        else:
             print(f'✓ Skipping annotation extraction (already exists)')
    
    return print('All slides processed successfully!')


def find_slide_directories(root_dir):
    """Recursively find all directories containing .svs files."""
    slide_dirs = set()
    for svs_file in Path(root_dir).rglob("*.svs"):
        slide_dirs.add(svs_file.parent)
    return sorted(list(slide_dirs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract annotated patches from H&E slides"
    )
    parser.add_argument("--root_dir", type=Path, required=True,
        help="Root directory to search for .svs files (searches recursively)")
    parser.add_argument("--output_dir", type=Path, required=True, 
        help="Root output directory (will create subdirs matching input structure)")
    parser.add_argument("--level", type=int, default=2,
        help="Pyramid level (0=40x, 1=20x, 2=10x, 3=5x). Default: 2 (10x)")
    parser.add_argument("--patch_size", type=int, default=224,
        help="Size of square patches. Default: 224")
    parser.add_argument("--overlap", type=int, default=0,
        help="Overlap between patches in pixels. Default: 0 (no overlap)")
    parser.add_argument("--use_class_map", action="store_true",
        help="Apply 9→5 class mapping. Default: False (no mapping)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {args.root_dir}")
    
    if args.patch_size <= 0:
        raise ValueError("patch_size must be positive")
    
    if args.overlap < 0 or args.overlap >= args.patch_size:
        raise ValueError(f"overlap must be in range [0, {args.patch_size-1}]")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    slide_dirs = find_slide_directories(args.root_dir)
    
    if len(slide_dirs) == 0:
        raise ValueError(f"No .svs files found in {args.root_dir} or subdirectories")
    
    print(f"Found {len(slide_dirs)} directories with .svs files")
    print(f"=" * 60)

    CLASS_MAP = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,  
        6: 1,  
        # 7: 2,  
        8: 3,        
        9: 4, 
    }
    class_map = CLASS_MAP if args.use_class_map else None

    for i, slide_dir in enumerate(slide_dirs, 1):
        rel_path = slide_dir.relative_to(args.root_dir)
        output_subdir = args.output_dir / rel_path
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{i}/{len(slide_dirs)}] Processing: {rel_path}")
        print(f"Input:  {slide_dir}")
        print(f"Output: {output_subdir}")
        print(f"-" * 60)
        
        try:
            extracting_annotations(
                slide_dir=slide_dir,
                training_patches_dir=output_subdir,
                level=args.level,
                patch_size=args.patch_size,
                overlap=args.overlap,
                class_map=class_map,
            )
            print(f"✓ Successfully processed {slide_dir.name}")
        except Exception as e:
            print(f"✗ Error processing {slide_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n" + "=" * 60)
    print(f"All directories processed!")
    print(f"Output saved to: {args.output_dir.resolve()}")
