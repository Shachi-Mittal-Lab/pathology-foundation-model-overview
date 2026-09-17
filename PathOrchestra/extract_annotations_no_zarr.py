import os
import sys
import json
import time
import argparse
import warnings
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
from PIL import Image

from funlib.geometry import Roi, Coordinate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from skimage.draw import polygon as draw_polygon
from tqdm import tqdm

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


class SVSPatchExtractor:
    """
    Extract annotated patches directly from SVS files without full Zarr conversion.
    Reads image patches on-demand and rasterizes labels per-patch.
    """
    
    def __init__(self, svs_path: str, xml_path: str, pyramid_level: int = 2):
        """
        INPUTS:
        - svs_path: path to .svs file
        - xml_path: path to .xml annotation file
        - pyramid_level: 0=40x, 1=20x, 2=10x, 3=5x
        """
        self.svs_path = svs_path
        self.xml_path = xml_path
        self.pyramid_level = pyramid_level
        
        # Open slide
        self.slide = openslide.OpenSlide(svs_path)
        self._get_metadata()
        self._parse_annotations()
    
    def _get_metadata(self):
        """Extract resolution and dimensions at pyramid level 0."""
        # Get resolution from properties
        self.mpp_x = float(self.slide.properties.get("openslide.mpp-x", 0.25))
        self.mpp_y = float(self.slide.properties.get("openslide.mpp-y", 0.25))
        
        # Get dimensions at level 0
        w0, h0 = self.slide.level_dimensions[0]
        
        # Get dimensions at target level
        self.width = self.slide.level_dimensions[self.pyramid_level][0]
        self.height = self.slide.level_dimensions[self.pyramid_level][1]
        
        # Downsampling factor from level 0 to target level
        self.downsample = self.slide.level_downsamples[self.pyramid_level]
        
        print(f"SVS loaded: {w0}×{h0} @ level 0")
        print(f"Level {self.pyramid_level}: {self.width}×{self.height}, downsample={self.downsample:.1f}x")
        print(f"MPP: {self.mpp_x:.4f} (x), {self.mpp_y:.4f} (y)")
    
    def _parse_annotations(self):
        """Parse XML and extract geometries grouped by class."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        includes_by_class = {}
        excludes_by_class = {}
        
        for annotation in root.iter("Annotation"):
            class_id = int(annotation.attrib["Id"])
            
            for region in annotation.iter("Region"):
                neg = region.attrib.get("NegativeROA", "0") == "1"
                vertices = [(float(v.attrib["X"]), float(v.attrib["Y"])) 
                           for v in region.iter("Vertex")]
                
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
        
        # Merge and apply exclusions
        self.geoms_by_class = {}
        all_class_ids = set(includes_by_class) | set(excludes_by_class)
        
        for cid in all_class_ids:
            inc = unary_union(includes_by_class.get(cid, []))
            if inc.is_empty:
                continue
            exc = unary_union(excludes_by_class.get(cid, []))
            geom = inc.difference(exc) if not exc.is_empty else inc
            if not geom.is_empty:
                self.geoms_by_class[cid] = geom
        
        print(f"Parsed {len(self.geoms_by_class)} classes from XML")
    
    def read_image_patch(self, x: int, y: int, patch_size: int) -> np.ndarray:
        """
        Read image patch at given (x, y) coordinates at target pyramid level.
        
        INPUTS:
        - x, y: top-left corner in pixels at target pyramid level
        - patch_size: patch size in pixels at target pyramid level
        
        OUTPUTS:
        - patch: (patch_size, patch_size, 3) uint8 RGB array
        """
        # Convert to level 0 coordinates
        x0 = int(x * self.downsample)
        y0 = int(y * self.downsample)
        w0 = int(patch_size * self.downsample)
        h0 = int(patch_size * self.downsample)
        
        # Clamp to slide boundaries
        w0 = min(w0, self.slide.level_dimensions[0][0] - x0)
        h0 = min(h0, self.slide.level_dimensions[0][1] - y0)
        
        if w0 <= 0 or h0 <= 0:
            return None
        
        # Read from slide at level 0, then resize to target level
        img = self.slide.read_region((x0, y0), 0, (w0, h0))
        img = img.convert('RGB')
        
        # Resize to patch_size if needed
        if img.size != (patch_size, patch_size):
            img = img.resize((patch_size, patch_size), Image.BILINEAR)
        
        return np.array(img, dtype=np.uint8)
    
    def rasterize_labels_patch(self, x: int, y: int, patch_size: int, 
                               class_map: Optional[Dict] = None) -> np.ndarray:
        """
        Rasterize annotation labels for a patch.
        
        INPUTS:
        - x, y: top-left corner in pixels at target pyramid level
        - patch_size: patch size in pixels
        - class_map: optional mapping of class IDs
        
        OUTPUTS:
        - mask: (patch_size, patch_size) uint8 with class IDs (or 255 for IGNORE)
        """
        IGNORE = 255
        mask = np.full((patch_size, patch_size), IGNORE, dtype=np.uint8)
        
        # Create box for this patch
        patch_box = box(x, y, x + patch_size, y + patch_size)
        
        # Rasterize each class
        for cid in sorted(self.geoms_by_class.keys()):
            geom = self.geoms_by_class[cid]
            
            # Skip if no intersection
            if not geom.intersects(patch_box):
                continue
            
            clipped = geom.intersection(patch_box)
            if clipped.is_empty:
                continue
            
            # Rasterize polygons
            for poly in self._iter_polys(clipped):
                self._paint_polygon(mask, poly, cid, x, y, IGNORE)
        
        # Apply class mapping if provided
        if class_map:
            new_mask = np.full_like(mask, IGNORE)
            for old_cid, new_cid in class_map.items():
                new_mask[mask == old_cid] = new_cid
            mask = new_mask
        
        return mask
    
    def _iter_polys(self, geom):
        """Iterate over polygons in geometry (handles MultiPolygon)."""
        gt = geom.geom_type
        if gt == "Polygon":
            yield geom
        elif gt == "MultiPolygon":
            yield from geom.geoms
    
    def _paint_polygon(self, mask: np.ndarray, poly: Polygon, class_id: int, 
                      x0: int, y0: int, ignore_val: int):
        """Paint polygon interior + clear holes in mask."""
        # Paint exterior
        ext = np.asarray(poly.exterior.coords)
        rr, cc = draw_polygon(ext[:, 1] - y0, ext[:, 0] - x0, mask.shape)
        rr = np.clip(rr, 0, mask.shape[0] - 1)
        cc = np.clip(cc, 0, mask.shape[1] - 1)
        bg = (mask[rr, cc] == ignore_val)
        mask[rr[bg], cc[bg]] = class_id
        
        # Clear holes
        for ring in poly.interiors:
            hole = np.asarray(ring.coords)
            rrh, cch = draw_polygon(hole[:, 1] - y0, hole[:, 0] - x0, mask.shape)
            rrh = np.clip(rrh, 0, mask.shape[0] - 1)
            cch = np.clip(cch, 0, mask.shape[1] - 1)
            here = (mask[rrh, cch] == class_id)
            mask[rrh[here], cch[here]] = ignore_val
    
    def extract_patches(self, output_dir: str, slidename: str, patch_size: int = 224,
                       overlap: int = 0, class_map: Optional[Dict] = None,
                       min_coverage: float = 0.01):
        """
        Extract and save all annotated patches as .npz files.
        
        INPUTS:
        - output_dir: where to save .npz files
        - slidename: prefix for output filenames
        - patch_size: patch size in pixels
        - overlap: overlap between patches
        - class_map: optional class ID remapping
        - min_coverage: minimum fraction of labeled pixels to save patch
        """
        os.makedirs(output_dir, exist_ok=True)
        stride = patch_size - overlap
        
        # Get bounding box of all annotations
        all_geoms = list(self.geoms_by_class.values())
        if not all_geoms:
            print("No annotations found!")
            return 0
        
        union_geom = unary_union(all_geoms)
        bbox = union_geom.bounds
        x_min, y_min, x_max, y_max = bbox
        
        # Add padding
        padding = patch_size
        x_min = max(0, int(x_min) - padding)
        y_min = max(0, int(y_min) - padding)
        x_max = min(self.width, int(x_max) + padding)
        y_max = min(self.height, int(y_max) + padding)
        
        print(f"Annotation bbox (level {self.pyramid_level}): "
              f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        print(f"Extracting {patch_size}×{patch_size} patches (stride={stride}px)")
        
        # Iterate over patches
        saved_count = 0
        total_patches = 0
        
        for y in tqdm(range(y_min, y_max - patch_size + 1, stride), 
                     desc="Extracting patches"):
            for x in range(x_min, x_max - patch_size + 1, stride):
                total_patches += 1
                
                # Read image
                img_patch = self.read_image_patch(x, y, patch_size)
                if img_patch is None:
                    continue
                
                # Rasterize labels
                label_patch = self.rasterize_labels_patch(x, y, patch_size, class_map)
                
                # Check coverage
                IGNORE = 255
                labeled_pixels = (label_patch != IGNORE).sum()
                coverage = labeled_pixels / (patch_size * patch_size)
                
                if coverage < min_coverage:
                    continue
                
                # Save
                filename = os.path.join(output_dir, 
                                       f'{slidename}_tile_y{y}_x{x}.npz')
                np.savez(
                    filename,
                    image=img_patch.astype(np.float32),
                    label=label_patch.astype(np.uint8)
                )
                saved_count += 1
        
        print(f"✓ Extracted {saved_count}/{total_patches} patches")
        return saved_count


def process_slides_fast(
    slide_dir: str,
    output_dir: str,
    level: int = 2,
    patch_size: int = 224,
    overlap: int = 0,
    class_map: Optional[Dict] = None
):
    """
    Process all .svs files in a directory without full Zarr conversion.
    
    INPUTS:
    - slide_dir: directory containing .svs and .xml pairs
    - output_dir: where to save .npz patches
    - level: pyramid level (0=40x, 1=20x, 2=10x, 3=5x)
    - patch_size: patch size in pixels
    - overlap: overlap between patches
    - class_map: optional class ID remapping
    """
    slide_dir = Path(slide_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    svs_files = list(slide_dir.glob("*.svs"))
    print(f"Found {len(svs_files)} .svs files")
    
    for svs_path in svs_files:
        slidename = svs_path.stem
        xml_path = slide_dir / f"{slidename}.xml"
        patch_output_dir = output_dir / slidename
        
        print(f"\n{'='*60}")
        print(f"Processing: {slidename}")
        
        if not xml_path.exists():
            print(f"⚠ No .xml found for {slidename}, skipping")
            continue
        
        # Check if already extracted
        npz_files = list(patch_output_dir.glob("*.npz"))
        if len(npz_files) > 0:
            print(f"✓ Already extracted ({len(npz_files)} patches)")
            continue
        
        try:
            extractor = SVSPatchExtractor(str(svs_path), str(xml_path), 
                                         pyramid_level=level)
            extractor.extract_patches(
                str(patch_output_dir),
                slidename,
                patch_size=patch_size,
                overlap=overlap,
                class_map=class_map
            )
            print(f"✓ Completed {slidename}")
        
        except Exception as e:
            print(f"✗ Error processing {slidename}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast extraction of annotated patches from SVS (no Zarr conversion)"
    )
    parser.add_argument("--slide_dir", type=str, required=True,
                       help="Directory containing .svs and .xml files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for .npz patches")
    parser.add_argument("--level", type=int, default=2,
                       help="Pyramid level (0=40x, 1=20x, 2=10x, 3=5x). Default: 2")
    parser.add_argument("--patch_size", type=int, default=224,
                       help="Patch size in pixels. Default: 224")
    parser.add_argument("--overlap", type=int, default=0,
                       help="Overlap between patches. Default: 0")
    parser.add_argument("--use_class_map", action="store_true",
                       help="Apply class mapping (9→5 classes)")
    
    args = parser.parse_args()
    
    CLASS_MAP = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
        6: 1,
        7: 2,
        8: 3,
        9: 4,
    } if args.use_class_map else None
    
    process_slides_fast(
        slide_dir=args.slide_dir,
        output_dir=args.output_dir,
        level=args.level,
        patch_size=args.patch_size,
        overlap=args.overlap,
        class_map=CLASS_MAP
    )