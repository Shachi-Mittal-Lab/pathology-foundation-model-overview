import json
from pathlib import Path
from typing import List, Dict, Tuple, Union
from abc import ABC, abstractmethod

try:
    from shapely.geometry import shape
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


class AnnotationLoader(ABC):
    """Base class for annotation loaders from different file formats."""
    
    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> List[Dict]:
        """
        Load annotations and return list of features/geometries.
        
        Returns:
            List of geometry dicts with 'type' and 'coordinates' keys
        """
        pass


class GeoJSONLoader(AnnotationLoader):
    """Load annotations from GeoJSON files (QuPath format)."""
    
    def load(self, filepath: Union[str, Path]) -> List[Dict]:
        """
        Load GeoJSON file and extract features.
        
        Args:
            filepath: Path to .geojson file
            
        Returns:
            List of geometry dicts
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Annotation file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            geojson_data = json.load(f)
        
        # Handle both FeatureCollection and raw geometry
        if geojson_data.get('type') == 'FeatureCollection':
            features = geojson_data.get('features', [])
            geometries = [feat['geometry'] for feat in features if 'geometry' in feat]
        else:
            geometries = [geojson_data]
        
        return geometries


class AnnotationCoordinateExtractor:
    """Extract and process coordinates from loaded annotations."""
    
    def __init__(self, loader: AnnotationLoader):
        """
        Args:
            loader: AnnotationLoader instance (GeoJSONLoader, etc.)
        """
        self.loader = loader
    
    def get_polygons(self, filepath: Union[str, Path]):
        """
        Load annotations and return as list of shapely Polygon objects.
        
        Args:
            filepath: Path to annotation file
            
        Returns:
            List of shapely Polygon objects (requires shapely)
        """
        if not HAS_SHAPELY:
            raise ImportError("shapely is required. Install with: pip install shapely")
        
        geometries = self.loader.load(filepath)
        polygons = []
        
        for geom in geometries:
            try:
                poly = shape(geom)
                if poly.is_valid:
                    polygons.append(poly)
            except Exception as e:
                print(f"Warning: Could not parse geometry: {e}")
        
        return polygons
    
    def get_bounding_boxes(self, filepath: Union[str, Path]) -> List[Tuple[float, float, float, float]]:
        """
        Get bounding boxes for each annotation (minx, miny, maxx, maxy).
        
        Args:
            filepath: Path to annotation file
            
        Returns:
            List of (minx, miny, maxx, maxy) tuples
        """
        if not HAS_SHAPELY:
            raise ImportError("shapely is required. Install with: pip install shapely")
        
        polygons = self.get_polygons(filepath)
        bboxes = [poly.bounds for poly in polygons]
        return bboxes
    
    def get_coordinates_list(self, filepath: Union[str, Path]) -> List[List[Tuple[float, float]]]:
        """
        Get raw coordinate lists for each polygon.
        
        Args:
            filepath: Path to annotation file
            
        Returns:
            List of coordinate lists (exterior rings only)
        """
        geometries = self.loader.load(filepath)
        coord_lists = []
        
        for geom in geometries:
            if geom['type'] in ['Polygon', 'MultiPolygon']:
                if geom['type'] == 'Polygon':
                    # Get exterior ring (first coordinate array)
                    coords = geom['coordinates'][0]
                    coord_lists.append(coords)
                else:  # MultiPolygon
                    for poly_coords in geom['coordinates']:
                        coord_lists.append(poly_coords[0])
        
        return coord_lists
    
    def get_union_polygon(self, filepath: Union[str, Path]):
        """
        Get union of all annotation polygons (single merged geometry).
        Useful for excluding background.
        
        Args:
            filepath: Path to annotation file
            
        Returns:
            Single shapely Polygon (union of all annotations)
        """
        if not HAS_SHAPELY:
            raise ImportError("shapely is required. Install with: pip install shapely")
        
        polygons = self.get_polygons(filepath)
        
        if not polygons:
            raise ValueError("No valid polygons found in annotation file")
        
        if len(polygons) == 1:
            return polygons[0]
        
        return unary_union(polygons)


def load_annotations(
    filepath: Union[str, Path],
    format: str = "geojson",
    return_type: str = "polygons"
):
    """
    Convenience function to load annotations with one call.
    
    Args:
        filepath: Path to annotation file
        format: File format ("geojson", "shapefile", etc.)
        return_type: Type of output ("polygons", "bboxes", "coords", "union")
        
    Returns:
        Loaded annotations in requested format
    """
    # Map format strings to loaders
    loaders = {
        "geojson": GeoJSONLoader,
        # Add more formats here: "shapefile": ShapefileLoader, etc.
    }
    
    if format.lower() not in loaders:
        raise ValueError(f"Unsupported format: {format}. Available: {list(loaders.keys())}")
    
    loader = loaders[format.lower()]()
    extractor = AnnotationCoordinateExtractor(loader)
    
    if return_type == "polygons":
        return extractor.get_polygons(filepath)
    elif return_type == "bboxes":
        return extractor.get_bounding_boxes(filepath)
    elif return_type == "coords":
        return extractor.get_coordinates_list(filepath)
    elif return_type == "union":
        return extractor.get_union_polygon(filepath)
    else:
        raise ValueError(f"Unknown return_type: {return_type}")
