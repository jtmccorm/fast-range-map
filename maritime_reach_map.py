#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = REPO_ROOT / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

MPL_CONFIG_DIR = REPO_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import shapefile
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, MultipleLocator
from PIL import Image, ImageDraw
from shapely import STRtree
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

NM_TO_KM = 1.852
EARTH_RADIUS_KM = 6371.0088
DEFAULT_BBOX = (70.0, 170.0, -20.0, 40.0)
DEFAULT_LAND_SHP = REPO_ROOT / "data" / "ne_10m_land" / "ne_10m_land.shp"
CACHE_DIR = REPO_ROOT / "cache"
NEIGHBOR_DELTAS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)


@dataclass(frozen=True)
class HubLocation:
    lat: float
    lon: float
    vessels: tuple["VesselSpec", ...] = ()

    @property
    def transport_strength(self) -> float:
        return float(sum(vessel.transport_strength for vessel in self.vessels))

    @property
    def max_vessel_range_nm(self) -> float:
        return max((vessel.range_nm for vessel in self.vessels), default=0.0)


@dataclass(frozen=True)
class VesselSpec:
    payload_tons: float
    speed_knots: float
    range_nm: float

    @property
    def transport_strength(self) -> float:
        return self.payload_tons * self.speed_knots * 12.0


@dataclass(frozen=True)
class CachedDistanceField:
    distances: np.ndarray
    bounds: tuple[int, int, int, int]
    grid_shape: tuple[int, int]

    @property
    def row_slice(self) -> slice:
        return slice(self.bounds[0], self.bounds[1])

    @property
    def col_slice(self) -> slice:
        return slice(self.bounds[2], self.bounds[3])


@dataclass(frozen=True)
class RoutedHub:
    index: int
    original: HubLocation
    trace_origin: HubLocation
    start_cell: tuple[int, int]
    distance_field: CachedDistanceField

    @property
    def label(self) -> str:
        return f"Hub {self.index}"


@dataclass(frozen=True)
class TracedHub(RoutedHub):
    round_trip_polygon: BaseGeometry
    one_way_polygon: BaseGeometry


@dataclass(frozen=True)
class NavigationGrid:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    lon_step_deg: float
    lat_step_deg: float
    lon_centers: np.ndarray
    lat_centers: np.ndarray
    land_mask: np.ndarray
    water_mask: np.ndarray
    ns_cost_km: float
    ew_cost_km: np.ndarray
    diag_up_cost_km: np.ndarray
    diag_down_cost_km: np.ndarray
    min_edge_cost_km: float

    @property
    def rows(self) -> int:
        return int(self.lat_centers.size)

    @property
    def cols(self) -> int:
        return int(self.lon_centers.size)

    def cell_center(self, row: int, col: int) -> HubLocation:
        return HubLocation(lat=float(self.lat_centers[row]), lon=float(self.lon_centers[col]))

    def coord_to_cell(self, lat: float, lon: float) -> tuple[int, int]:
        row = int(math.floor((lat - self.min_lat) / self.lat_step_deg))
        col = int(math.floor((lon - self.min_lon) / self.lon_step_deg))
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))
        return row, col


class LandDetector:
    def __init__(self, polygons: Sequence[Polygon]) -> None:
        self.polygons = list(polygons)
        self.tree = STRtree(self.polygons)
        self.union = unary_union(self.polygons)

    def is_land(self, lon: float, lat: float) -> bool:
        return len(self.tree.query(Point(lon, lat), predicate="covered_by")) > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static PNG map of maritime reach constrained by land."
    )
    parser.add_argument(
        "--output-mode",
        choices=("range", "throughput"),
        default="range",
        help="Visualization mode. Use 'range' for the existing reach map or 'throughput' for tons/day capacity.",
    )
    parser.add_argument(
        "--hub",
        action="append",
        nargs=2,
        metavar=("LAT", "LON"),
        type=float,
        help="Hub latitude and longitude. Repeat for multiple hubs.",
    )
    parser.add_argument(
        "--hub-vessel",
        action="append",
        nargs=4,
        metavar=("HUB_INDEX", "PAYLOAD_TONS", "SPEED_KNOTS", "RANGE_NM"),
        type=float,
        help=(
            "Assign a vessel type to a hub for throughput mode. Repeat to add multiple vessel types or vessels. "
            "Hub indices are 1-based and correspond to the order of --hub arguments."
        ),
    )
    parser.add_argument(
        "--range-nm",
        type=float,
        default=2000.0,
        help="Platform range in nautical miles. Default: 2000.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        metavar=("MIN_LON", "MAX_LON", "MIN_LAT", "MAX_LAT"),
        type=float,
        default=DEFAULT_BBOX,
        help="Map bounding box in degrees. Default: 70 170 -20 40.",
    )
    parser.add_argument(
        "--rays",
        type=int,
        default=360,
        help="Deprecated and ignored. Retained only for backward compatibility.",
    )
    parser.add_argument(
        "--step-km",
        type=float,
        default=8.0,
        help="Routing grid resolution in kilometers. Default: 8.",
    )
    parser.add_argument(
        "--land-shapefile",
        type=Path,
        default=DEFAULT_LAND_SHP,
        help=f"Path to the land shapefile. Default: {DEFAULT_LAND_SHP}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output/maritime_reach_map.png",
        help="Output PNG path. Default: maritime_reach_map.png.",
    )
    parser.add_argument(
        "--throughput-contours",
        nargs="+",
        metavar="TONS_PER_DAY",
        type=float,
        default=[50.0, 100.0, 250.0, 500.0],
        help="Contour levels for throughput mode in tons/day. Default: 50 100 250 500.",
    )
    parser.add_argument(
        "--min-cycle-days",
        type=float,
        default=1.0,
        help=(
            "Minimum delivery cycle time per vessel in days for throughput mode. "
            "Caps each vessel at payload_tons / min_cycle_days tons/day. Default: 1.0."
        ),
    )
    return parser.parse_args()


def normalize_longitude(lon: float) -> float:
    return (lon + 540.0) % 360.0 - 180.0


def destination_point(
    start_lat: float, start_lon: float, bearing_deg: float, distance_km: float
) -> tuple[float, float]:
    lat1 = math.radians(start_lat)
    lon1 = math.radians(start_lon)
    bearing = math.radians(bearing_deg)
    angular_distance = distance_km / EARTH_RADIUS_KM

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_dist = math.sin(angular_distance)
    cos_dist = math.cos(angular_distance)

    lat2 = math.asin(sin_lat1 * cos_dist + cos_lat1 * sin_dist * math.cos(bearing))
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * sin_dist * cos_lat1,
        cos_dist - sin_lat1 * math.sin(lat2),
    )
    return math.degrees(lat2), normalize_longitude(math.degrees(lon2))


def great_circle_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = math.radians(lon2 - lon1)
    sin_half_lat = math.sin(delta_lat / 2.0)
    sin_half_lon = math.sin(delta_lon / 2.0)
    a = (
        sin_half_lat * sin_half_lat
        + math.cos(lat1_rad) * math.cos(lat2_rad) * sin_half_lon * sin_half_lon
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def expand_bbox(
    bbox: tuple[float, float, float, float], range_nm: float
) -> tuple[float, float, float, float]:
    min_lon, max_lon, min_lat, max_lat = bbox
    margin_deg = max(6.0, range_nm / 60.0 + 2.0)
    return (
        max(-179.9, min_lon - margin_deg),
        min(179.9, max_lon + margin_deg),
        max(-89.9, min_lat - margin_deg),
        min(89.9, max_lat + margin_deg),
    )


def load_land_polygons(
    shapefile_path: Path, search_bbox: tuple[float, float, float, float]
) -> LandDetector:
    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Missing land dataset: {shapefile_path}. "
            "Expected Natural Earth land polygons in data/ne_10m_land/."
        )

    search_region = box(search_bbox[0], search_bbox[2], search_bbox[1], search_bbox[3])
    polygons: list[Polygon] = []
    reader = shapefile.Reader(str(shapefile_path))
    for shp in reader.iterShapes():
        geom = shape(shp.__geo_interface__)
        if not geom.intersects(search_region):
            continue
        clipped = geom.intersection(search_region)
        for polygon in iter_polygons(clipped):
            if not polygon.is_empty:
                polygons.append(polygon)

    if not polygons:
        raise RuntimeError("No land polygons intersect the requested region.")
    return LandDetector(polygons)


def build_navigation_grid(
    detector: LandDetector,
    routing_bbox: tuple[float, float, float, float],
    step_km: float,
) -> NavigationGrid:
    min_lon, max_lon, min_lat, max_lat = routing_bbox
    mid_lat = (min_lat + max_lat) / 2.0
    lat_step_deg = step_km / 111.32
    lon_scale = max(0.2, math.cos(math.radians(mid_lat)))
    lon_step_deg = step_km / (111.32 * lon_scale)

    rows = max(2, int(math.ceil((max_lat - min_lat) / lat_step_deg)))
    cols = max(2, int(math.ceil((max_lon - min_lon) / lon_step_deg)))
    grid_max_lat = min_lat + rows * lat_step_deg
    grid_max_lon = min_lon + cols * lon_step_deg

    lat_centers = min_lat + (np.arange(rows, dtype=float) + 0.5) * lat_step_deg
    lon_centers = min_lon + (np.arange(cols, dtype=float) + 0.5) * lon_step_deg

    image = Image.new("L", (cols, rows), 0)
    draw = ImageDraw.Draw(image)

    def lon_to_x(lon: float) -> float:
        return (lon - min_lon) / lon_step_deg

    def lat_to_y(lat: float) -> float:
        return (grid_max_lat - lat) / lat_step_deg

    for polygon in detector.polygons:
        exterior = [(lon_to_x(lon), lat_to_y(lat)) for lon, lat in polygon.exterior.coords]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=255)
        for interior in polygon.interiors:
            hole = [(lon_to_x(lon), lat_to_y(lat)) for lon, lat in interior.coords]
            if len(hole) >= 3:
                draw.polygon(hole, fill=0)

    land_mask = np.flipud(np.array(image, dtype=np.uint8) > 0)
    water_mask = ~land_mask

    ns_cost_km = great_circle_distance_km(lat_centers[0], lon_centers[0], lat_centers[1], lon_centers[0])
    ew_cost_km = np.array(
        [
            great_circle_distance_km(lat, lon_centers[0], lat, lon_centers[0] + lon_step_deg)
            for lat in lat_centers
        ],
        dtype=float,
    )
    diag_up_cost_km = np.full(rows, np.inf, dtype=float)
    diag_down_cost_km = np.full(rows, np.inf, dtype=float)
    for row in range(1, rows):
        diag_up_cost_km[row] = great_circle_distance_km(
            lat_centers[row], lon_centers[0], lat_centers[row - 1], lon_centers[0] + lon_step_deg
        )
    for row in range(rows - 1):
        diag_down_cost_km[row] = great_circle_distance_km(
            lat_centers[row], lon_centers[0], lat_centers[row + 1], lon_centers[0] + lon_step_deg
        )

    min_edge_cost_km = min(
        float(ns_cost_km),
        float(ew_cost_km.min()),
        float(diag_up_cost_km[1:].min()),
        float(diag_down_cost_km[:-1].min()),
    )

    return NavigationGrid(
        min_lon=min_lon,
        max_lon=grid_max_lon,
        min_lat=min_lat,
        max_lat=grid_max_lat,
        lon_step_deg=lon_step_deg,
        lat_step_deg=lat_step_deg,
        lon_centers=lon_centers,
        lat_centers=lat_centers,
        land_mask=land_mask,
        water_mask=water_mask,
        ns_cost_km=float(ns_cost_km),
        ew_cost_km=ew_cost_km,
        diag_up_cost_km=diag_up_cost_km,
        diag_down_cost_km=diag_down_cost_km,
        min_edge_cost_km=float(min_edge_cost_km),
    )


def snap_hub_to_water(
    hub: HubLocation, detector: LandDetector, grid: NavigationGrid
) -> tuple[HubLocation, tuple[int, int]]:
    row, col = grid.coord_to_cell(hub.lat, hub.lon)
    if grid.water_mask[row, col] and not detector.is_land(hub.lon, hub.lat):
        return hub, (row, col)

    snapped_row, snapped_col = find_nearest_water_cell(grid, row, col)
    return grid.cell_center(snapped_row, snapped_col), (snapped_row, snapped_col)


def find_nearest_water_cell(
    grid: NavigationGrid, start_row: int, start_col: int
) -> tuple[int, int]:
    if grid.water_mask[start_row, start_col]:
        return start_row, start_col

    for radius in range(1, max(grid.rows, grid.cols)):
        best_cell: tuple[int, int] | None = None
        best_distance = math.inf
        row_min = max(0, start_row - radius)
        row_max = min(grid.rows - 1, start_row + radius)
        col_min = max(0, start_col - radius)
        col_max = min(grid.cols - 1, start_col + radius)

        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                if max(abs(row - start_row), abs(col - start_col)) != radius:
                    continue
                if not grid.water_mask[row, col]:
                    continue
                candidate_distance = (row - start_row) ** 2 + (col - start_col) ** 2
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    best_cell = (row, col)

        if best_cell is not None:
            return best_cell

    raise RuntimeError("Unable to locate any water cell in the navigation grid.")


def compute_cost_distance(
    grid: NavigationGrid, start_cell: tuple[int, int], max_distance_km: float
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    start_row, start_col = start_cell
    radius_cells = int(math.ceil(max_distance_km / grid.min_edge_cost_km)) + 4
    row_start = max(0, start_row - radius_cells)
    row_end = min(grid.rows, start_row + radius_cells + 1)
    col_start = max(0, start_col - radius_cells)
    col_end = min(grid.cols, start_col + radius_cells + 1)

    water_mask = grid.water_mask[row_start:row_end, col_start:col_end]
    distances = np.full(water_mask.shape, np.inf, dtype=np.float32)
    visited = np.zeros(water_mask.shape, dtype=bool)

    local_start = (start_row - row_start, start_col - col_start)
    if not water_mask[local_start]:
        raise RuntimeError("Routing origin is not on a water cell.")

    heap: list[tuple[float, int, int]] = [(0.0, local_start[0], local_start[1])]
    distances[local_start] = 0.0

    while heap:
        current_distance, row, col = heapq.heappop(heap)
        if current_distance > max_distance_km:
            break
        if visited[row, col]:
            continue
        visited[row, col] = True
        for next_row, next_col, step_cost in iter_navigable_neighbors(
            grid, water_mask, row, col, row_start, col_start
        ):
            proposed_distance = current_distance + step_cost
            if proposed_distance >= distances[next_row, next_col] or proposed_distance > max_distance_km:
                continue

            distances[next_row, next_col] = proposed_distance
            heapq.heappush(heap, (proposed_distance, next_row, next_col))

    return distances, (row_start, row_end, col_start, col_end)


def cache_distance_field(
    cache_path: Path, distances: np.ndarray, bounds: tuple[int, int, int, int]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        cache_path,
        {"distances": np.asarray(distances, dtype=np.float32), "bounds": np.asarray(bounds, dtype=np.int32)},
        allow_pickle=True,
    )


def load_cached_distance_field(
    cache_path: Path,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    if not cache_path.exists():
        return None

    try:
        payload = np.load(cache_path, allow_pickle=True).item()
        distances = np.asarray(payload["distances"], dtype=np.float32)
        bounds_array = np.asarray(payload["bounds"], dtype=np.int32)
    except (OSError, ValueError, KeyError, TypeError):
        return None

    if distances.ndim != 2 or bounds_array.shape != (4,):
        return None

    row_start, row_end, col_start, col_end = (int(value) for value in bounds_array)
    if row_start < 0 or col_start < 0 or row_end <= row_start or col_end <= col_start:
        return None
    if distances.shape != (row_end - row_start, col_end - col_start):
        return None

    return distances, (row_start, row_end, col_start, col_end)


def distance_cache_path(
    hub_index: int,
    hub: HubLocation,
    trace_origin: HubLocation,
    start_cell: tuple[int, int],
    grid: NavigationGrid,
    step_km: float,
    max_distance_km: float,
) -> Path:
    cache_key = {
        "hub_index": hub_index,
        "hub_lat": round(hub.lat, 6),
        "hub_lon": round(hub.lon, 6),
        "trace_lat": round(trace_origin.lat, 6),
        "trace_lon": round(trace_origin.lon, 6),
        "start_cell": list(start_cell),
        "step_km": round(step_km, 6),
        "max_distance_km": round(max_distance_km, 6),
        "grid": {
            "rows": grid.rows,
            "cols": grid.cols,
            "min_lon": round(grid.min_lon, 6),
            "max_lon": round(grid.max_lon, 6),
            "min_lat": round(grid.min_lat, 6),
            "max_lat": round(grid.max_lat, 6),
        },
    }
    digest = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode("ascii")).hexdigest()[:12]
    filename = (
        f"hub{hub_index:02d}_"
        f"step{step_km:.3f}_"
        f"lat{hub.lat:.4f}_"
        f"lon{hub.lon:.4f}_"
        f"{digest}_distance.npy"
    ).replace("-", "m")
    return CACHE_DIR / filename


def movement_cost_km(
    grid: NavigationGrid, row: int, delta_row: int, delta_col: int
) -> float:
    if delta_row == 0:
        return float(grid.ew_cost_km[row])
    if delta_col == 0:
        return grid.ns_cost_km
    if delta_row < 0:
        return float(grid.diag_up_cost_km[row])
    return float(grid.diag_down_cost_km[row])


def iter_navigable_neighbors(
    grid: NavigationGrid,
    water_mask: np.ndarray,
    row: int,
    col: int,
    row_offset: int,
    col_offset: int,
) -> Iterator[tuple[int, int, float]]:
    global_row = row_offset + row
    global_col = col_offset + col

    for delta_row, delta_col in NEIGHBOR_DELTAS:
        next_row = row + delta_row
        next_col = col + delta_col
        if (
            next_row < 0
            or next_row >= water_mask.shape[0]
            or next_col < 0
            or next_col >= water_mask.shape[1]
            or not water_mask[next_row, next_col]
        ):
            continue
        if not move_is_clear(water_mask, row, col, delta_row, delta_col):
            continue
        yield next_row, next_col, edge_cost_km(grid, global_row, global_col, delta_row, delta_col)


def move_is_clear(
    water_mask: np.ndarray, row: int, col: int, delta_row: int, delta_col: int
) -> bool:
    abs_row = abs(delta_row)
    abs_col = abs(delta_col)

    if abs_row <= 1 and abs_col <= 1:
        if abs_row == 1 and abs_col == 1:
            return bool(water_mask[row + delta_row, col] and water_mask[row, col + delta_col])
        return True

    if sorted((abs_row, abs_col)) != [1, 2]:
        return False

    return any(
        all(water_mask[row + offset_row, col + offset_col] for offset_row, offset_col in path_offsets)
        for path_offsets in knight_path_offsets(delta_row, delta_col)
    )


def knight_path_offsets(delta_row: int, delta_col: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    row_step = 1 if delta_row > 0 else -1
    col_step = 1 if delta_col > 0 else -1

    if abs(delta_row) == 2:
        return (
            ((row_step, 0), (2 * row_step, 0)),
            ((row_step, 0), (row_step, col_step)),
            ((0, col_step), (row_step, col_step)),
        )

    return (
        ((0, col_step), (0, 2 * col_step)),
        ((0, col_step), (row_step, col_step)),
        ((row_step, 0), (row_step, col_step)),
    )


def edge_cost_km(grid: NavigationGrid, row: int, col: int, delta_row: int, delta_col: int) -> float:
    if max(abs(delta_row), abs(delta_col)) == 1:
        return movement_cost_km(grid, row, delta_row, delta_col)

    return great_circle_distance_km(
        grid.lat_centers[row],
        grid.lon_centers[col],
        grid.lat_centers[row + delta_row],
        grid.lon_centers[col + delta_col],
    )


def build_reach_polygon(
    distances: np.ndarray,
    threshold_km: float,
    grid: NavigationGrid,
    bounds: tuple[int, int, int, int],
    land_union: BaseGeometry,
    trace_origin: HubLocation,
    bbox: tuple[float, float, float, float],
) -> BaseGeometry:
    if not np.isfinite(distances).any():
        return GeometryCollection()

    row_start, row_end, col_start, col_end = bounds
    lon_centers = grid.lon_centers[col_start:col_end]
    lat_centers = grid.lat_centers[row_start:row_end]

    raw_geometry = distance_field_to_geometry(
        distances, threshold_km, lon_centers, lat_centers, grid.lon_step_deg, grid.lat_step_deg
    )
    if raw_geometry.is_empty:
        return raw_geometry
    if not raw_geometry.is_valid:
        raw_geometry = raw_geometry.buffer(0)

    cleaned = raw_geometry.difference(land_union)
    if not cleaned.is_valid:
        cleaned = cleaned.buffer(0)

    simplify_tolerance = max(grid.lon_step_deg, grid.lat_step_deg) * 0.45
    if simplify_tolerance > 0.0 and not cleaned.is_empty:
        cleaned = cleaned.simplify(simplify_tolerance, preserve_topology=True)
        cleaned = cleaned.difference(land_union)
        if not cleaned.is_valid:
            cleaned = cleaned.buffer(0)

    connected = keep_component_for_anchor(cleaned, Point(trace_origin.lon, trace_origin.lat))
    clipped = connected.intersection(box(bbox[0], bbox[2], bbox[1], bbox[3]))
    if not clipped.is_valid:
        clipped = clipped.buffer(0)
    return clipped


def distance_field_to_geometry(
    distances: np.ndarray,
    threshold_km: float,
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
    lon_step_deg: float,
    lat_step_deg: float,
) -> BaseGeometry:
    clipped_distances = np.where(np.isfinite(distances), distances, np.nan)
    padded_distances = np.pad(clipped_distances, 1, constant_values=np.nan)
    padded_lon = np.concatenate(
        ([lon_centers[0] - lon_step_deg], lon_centers, [lon_centers[-1] + lon_step_deg])
    )
    padded_lat = np.concatenate(
        ([lat_centers[0] - lat_step_deg], lat_centers, [lat_centers[-1] + lat_step_deg])
    )

    figure, axis = plt.subplots()
    contour = axis.contourf(padded_lon, padded_lat, padded_distances, levels=[0.0, threshold_km])
    polygons: list[BaseGeometry] = []
    for path in contour.get_paths():
        for ring in path.to_polygons():
            polygon = Polygon(ring)
            if not polygon.is_empty and polygon.area > 0.0:
                polygons.append(polygon)
    plt.close(figure)

    if not polygons:
        return GeometryCollection()
    geometry = unary_union(polygons)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def keep_component_for_anchor(geometry: BaseGeometry, anchor: Point) -> BaseGeometry:
    if geometry.is_empty:
        return geometry
    if isinstance(geometry, Polygon):
        return geometry

    polygons = list(iter_polygons(geometry))
    if not polygons:
        return GeometryCollection()

    buffered_anchor = anchor.buffer(1e-6)
    for polygon in polygons:
        if polygon.covers(anchor) or polygon.intersects(buffered_anchor):
            return polygon
    return min(polygons, key=lambda polygon: polygon.distance(anchor))


def compute_overlap(polygons: Sequence[BaseGeometry]) -> BaseGeometry:
    overlaps: list[BaseGeometry] = []
    for left, right in itertools.combinations(polygons, 2):
        if left.is_empty or right.is_empty:
            continue
        intersection = left.intersection(right)
        if not intersection.is_empty:
            overlaps.append(intersection)
    if not overlaps:
        return GeometryCollection()
    merged = unary_union(overlaps)
    if not merged.is_valid:
        merged = merged.buffer(0)
    return merged


def iter_polygons(geometry: BaseGeometry) -> Iterator[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            if not polygon.is_empty:
                yield polygon
        return
    if isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from iter_polygons(child)


def polygon_to_path(polygon: Polygon) -> MplPath:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []

    def add_ring(coords: Iterable[tuple[float, float]]) -> None:
        ring = list(coords)
        if len(ring) < 4:
            return
        vertices.extend((float(lon), float(lat)) for lon, lat in ring)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])

    add_ring(polygon.exterior.coords)
    for interior in polygon.interiors:
        add_ring(interior.coords)

    return MplPath(vertices, codes)


def add_geometry(
    ax: plt.Axes,
    geometry: BaseGeometry,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    paths = [polygon_to_path(polygon) for polygon in iter_polygons(geometry)]
    if not paths:
        return
    compound_path = MplPath.make_compound_path(*paths)
    patch = PathPatch(
        compound_path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)


def format_lat(value: float, _position: float) -> str:
    suffix = "N" if value >= 0 else "S"
    return f"{abs(value):.0f}°{suffix}"


def format_lon(value: float, _position: float) -> str:
    suffix = "E" if value >= 0 else "W"
    return f"{abs(value):.0f}°{suffix}"


def add_land_layer(ax: plt.Axes, land_union: BaseGeometry, bbox: tuple[float, float, float, float], *, zorder: int) -> None:
    map_region = box(bbox[0], bbox[2], bbox[1], bbox[3])
    land_in_view = land_union.intersection(map_region)
    add_geometry(
        ax,
        land_in_view,
        facecolor="#efe7d8",
        edgecolor="#49423f",
        linewidth=0.55,
        alpha=1.0,
        zorder=zorder,
    )


def add_hub_markers(ax: plt.Axes, hubs: Sequence[RoutedHub], bbox: tuple[float, float, float, float]) -> None:
    min_lon, max_lon, min_lat, max_lat = bbox
    for hub in hubs:
        ax.scatter(
            hub.original.lon,
            hub.original.lat,
            s=72,
            facecolor="#c1121f",
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
        )
        offset_lon = 1.1 if hub.original.lon <= (min_lon + max_lon) / 2.0 else -1.1
        offset_lat = 0.9 if hub.original.lat <= (min_lat + max_lat) / 2.0 else -0.9
        horizontal_alignment = "left" if offset_lon > 0 else "right"
        vertical_alignment = "bottom" if offset_lat > 0 else "top"
        label = f"{hub.label}\n({hub.original.lat:.2f}, {hub.original.lon:.2f})"
        text = ax.text(
            hub.original.lon + offset_lon,
            hub.original.lat + offset_lat,
            label,
            fontsize=9,
            color="#1f2933",
            ha=horizontal_alignment,
            va=vertical_alignment,
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.4},
        )
        text.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground="white")])


def style_map_axes(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    title: str,
    subtitle: str,
) -> None:
    min_lon, max_lon, min_lat, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.grid(color="white", linewidth=0.8, linestyle="--", alpha=0.85)
    ax.set_aspect(1.0 / math.cos(math.radians(mid_lat)))
    ax.set_title(f"{title}\n{subtitle}", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)


def compute_throughput_field(
    distance_fields: Sequence[CachedDistanceField],
    hubs: Sequence[HubLocation],
    d_min_nm: float | None = None,
    min_cycle_days: float = 1.0,
) -> np.ndarray:
    if len(distance_fields) != len(hubs):
        raise ValueError("Distance fields and hubs must have matching lengths.")
    if not distance_fields:
        raise ValueError("At least one cached distance field is required.")
    if min_cycle_days <= 0.0:
        raise ValueError("min_cycle_days must be positive.")

    grid_shape = distance_fields[0].grid_shape
    if any(field.grid_shape != grid_shape for field in distance_fields):
        raise ValueError("All cached distance fields must use the same grid shape.")

    effective_distance_floor_nm = max(float(d_min_nm or 1.0), 1e-6)
    throughput = np.zeros(grid_shape, dtype=np.float32)

    for distance_field, hub in zip(distance_fields, hubs):
        if not hub.vessels:
            continue

        distance_nm = distance_field.distances.astype(np.float32, copy=False) / NM_TO_KM
        finite_mask = np.isfinite(distance_nm)
        if not finite_mask.any():
            continue

        stabilized_distance_nm = np.maximum(distance_nm, effective_distance_floor_nm)
        throughput_window = throughput[distance_field.row_slice, distance_field.col_slice]

        for vessel in hub.vessels:
            reachable_mask = finite_mask & (distance_nm <= (vessel.range_nm / 2.0))
            if not reachable_mask.any():
                continue
            raw_throughput = vessel.transport_strength / stabilized_distance_nm[reachable_mask]
            capped_throughput = np.minimum(raw_throughput, vessel.payload_tons / min_cycle_days)
            throughput_window[reachable_mask] += capped_throughput.astype(np.float32, copy=False)

    return throughput


def plot_throughput_heatmap(
    ax: plt.Axes,
    throughput_field: np.ndarray,
    grid: NavigationGrid,
    *,
    vmax: float | None = None,
):
    positive_mask = np.isfinite(throughput_field) & grid.water_mask & (throughput_field > 0.0)
    display_field = np.ma.masked_where(~positive_mask, throughput_field)
    maximum_value = float(vmax) if vmax is not None else float(display_field.max()) if display_field.count() else 1.0
    heatmap_cmap = plt.get_cmap("YlOrRd").copy()
    heatmap_cmap.set_bad(alpha=0.0)
    return ax.imshow(
        display_field,
        origin="lower",
        extent=(grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat),
        cmap=heatmap_cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=max(maximum_value, 1.0),
        alpha=0.86,
        zorder=2,
    )


def plot_throughput_contours(
    ax: plt.Axes,
    throughput_field: np.ndarray,
    grid: NavigationGrid,
    contour_levels: Sequence[float] = (50.0, 100.0, 250.0, 500.0),
):
    contour_field = np.ma.masked_where(~grid.water_mask | (throughput_field <= 0.0), throughput_field)
    if contour_field.count() == 0:
        return None

    max_capacity = float(contour_field.max())
    levels = sorted({float(level) for level in contour_levels if 0.0 < float(level) <= max_capacity})
    if not levels:
        return None

    contours = ax.contour(
        grid.lon_centers,
        grid.lat_centers,
        contour_field,
        levels=levels,
        colors="#3d2d1f",
        linewidths=0.95,
        alpha=0.92,
        zorder=5,
    )
    for collection in getattr(contours, "collections", ()):
        collection.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])

    labels = ax.clabel(contours, fmt=lambda value: f"{value:,.0f}", inline=True, fontsize=8, colors="#2a1b12")
    for text in labels:
        text.set_path_effects([path_effects.withStroke(linewidth=1.3, foreground="white")])
    return contours


def render_map(
    traced_hubs: Sequence[TracedHub],
    land_union: BaseGeometry,
    bbox: tuple[float, float, float, float],
    range_nm: float,
    output_path: Path,
) -> None:
    round_trip_overlap = compute_overlap([hub.round_trip_polygon for hub in traced_hubs])
    one_way_overlap = compute_overlap([hub.one_way_polygon for hub in traced_hubs])
    combined_overlap = unary_union(
        [geometry for geometry in (round_trip_overlap, one_way_overlap) if not geometry.is_empty]
    )

    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#d8eef7")

    add_land_layer(ax, land_union, bbox, zorder=1)

    for hub in traced_hubs:
        add_geometry(
            ax,
            hub.one_way_polygon,
            facecolor="#f4a261",
            edgecolor="#cf7c1d",
            linewidth=0.6,
            alpha=0.30,
            zorder=2,
        )
    for hub in traced_hubs:
        add_geometry(
            ax,
            hub.round_trip_polygon,
            facecolor="#4f83cc",
            edgecolor="#2c5ea8",
            linewidth=0.7,
            alpha=0.36,
            zorder=3,
        )
    add_geometry(
        ax,
        combined_overlap,
        facecolor="#7b2cbf",
        edgecolor="#5a189a",
        linewidth=0.8,
        alpha=0.42,
        zorder=4,
    )

    add_hub_markers(ax, traced_hubs, bbox)
    style_map_axes(
        ax,
        bbox,
        title="Maritime Operational Reach in Southeast Asia",
        subtitle=f"Round trip: {range_nm / 2:.0f} nm | One way: {range_nm:.0f} nm",
    )

    legend_items = [
        Patch(facecolor="#4f83cc", edgecolor="#2c5ea8", alpha=0.36, label="Round Trip Range"),
        Patch(facecolor="#f4a261", edgecolor="#cf7c1d", alpha=0.30, label="One Way Range"),
        Patch(facecolor="#7b2cbf", edgecolor="#5a189a", alpha=0.42, label="Overlap Between Hubs"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#c1121f",
            markeredgecolor="black",
            markersize=9,
            label="Hub",
        ),
    ]
    legend = ax.legend(
        handles=legend_items,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#a0aec0",
        fontsize=10,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.text(
        0.013,
        0.012,
        "Land mask: Natural Earth 1:10m land polygons | Water-routed cost-distance model",
        fontsize=8.5,
        color="#4a5568",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_throughput_map(
    routed_hubs: Sequence[RoutedHub],
    throughput_field: np.ndarray,
    land_union: BaseGeometry,
    grid: NavigationGrid,
    bbox: tuple[float, float, float, float],
    contour_levels: Sequence[float],
    output_path: Path,
    d_min_nm: float,
    min_cycle_days: float,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#d8eef7")

    visible_positive = throughput_field[np.isfinite(throughput_field) & grid.water_mask & (throughput_field > 0.0)]
    heatmap = plot_throughput_heatmap(
        ax,
        throughput_field,
        grid,
        vmax=float(visible_positive.max()) if visible_positive.size else 1.0,
    )
    add_land_layer(ax, land_union, bbox, zorder=3)
    plot_throughput_contours(ax, throughput_field, grid, contour_levels)
    add_hub_markers(ax, routed_hubs, bbox)
    style_map_axes(
        ax,
        bbox,
        title="Maritime Throughput Capacity in Southeast Asia",
        subtitle="Sustainment capacity from vessel transport strength and navigable delivery distance",
    )

    colorbar = fig.colorbar(heatmap, ax=ax, pad=0.02, shrink=0.9)
    colorbar.set_label("Throughput Capacity (tons/day)", fontsize=11)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:,.0f}"))

    fig.text(
        0.013,
        0.012,
        (
            "Water-routed throughput model | "
            f"d_min = {d_min_nm:.1f} nm | "
            f"Min cycle = {min_cycle_days:.1f} day(s) | "
            "Vessel contribution cutoff at half of listed range"
        ),
        fontsize=8.5,
        color="#4a5568",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_routed_hubs(
    hubs: Sequence[HubLocation],
    routing_range_nm: float,
    step_km: float,
    detector: LandDetector,
    grid: NavigationGrid,
) -> list[RoutedHub]:
    routed_hubs: list[RoutedHub] = []
    max_distance_km = routing_range_nm * NM_TO_KM

    for index, hub in enumerate(hubs, start=1):
        trace_origin, start_cell = snap_hub_to_water(hub, detector, grid)
        cache_path = distance_cache_path(index, hub, trace_origin, start_cell, grid, step_km, max_distance_km)
        cached_distance_field = load_cached_distance_field(cache_path)
        if cached_distance_field is None:
            distances, bounds = compute_cost_distance(grid, start_cell, max_distance_km)
            cache_distance_field(cache_path, distances, bounds)
        else:
            distances, bounds = cached_distance_field

        routed_hubs.append(
            RoutedHub(
                index=index,
                original=hub,
                trace_origin=trace_origin,
                start_cell=start_cell,
                distance_field=CachedDistanceField(
                    distances=distances,
                    bounds=bounds,
                    grid_shape=(grid.rows, grid.cols),
                ),
            )
        )
    return routed_hubs


def build_traced_hubs(
    routed_hubs: Sequence[RoutedHub],
    range_nm: float,
    bbox: tuple[float, float, float, float],
    detector: LandDetector,
    grid: NavigationGrid,
) -> list[TracedHub]:
    traced_hubs: list[TracedHub] = []
    round_trip_km = range_nm * NM_TO_KM / 2.0
    one_way_km = range_nm * NM_TO_KM

    for hub in routed_hubs:
        round_trip_polygon = build_reach_polygon(
            hub.distance_field.distances,
            round_trip_km,
            grid,
            hub.distance_field.bounds,
            detector.union,
            hub.trace_origin,
            bbox,
        )
        one_way_polygon = build_reach_polygon(
            hub.distance_field.distances,
            one_way_km,
            grid,
            hub.distance_field.bounds,
            detector.union,
            hub.trace_origin,
            bbox,
        )
        traced_hubs.append(
            TracedHub(
                index=hub.index,
                original=hub.original,
                trace_origin=hub.trace_origin,
                start_cell=hub.start_cell,
                distance_field=hub.distance_field,
                round_trip_polygon=round_trip_polygon,
                one_way_polygon=one_way_polygon,
            )
        )
    return traced_hubs


def parse_hub_vessels(
    raw_hub_vessels: Sequence[Sequence[float]] | None,
    hub_count: int,
) -> dict[int, list[VesselSpec]]:
    vessels_by_hub = {hub_index: [] for hub_index in range(1, hub_count + 1)}
    for raw_spec in raw_hub_vessels or []:
        if len(raw_spec) != 4:
            raise ValueError("Each --hub-vessel must provide HUB_INDEX PAYLOAD_TONS SPEED_KNOTS RANGE_NM.")

        raw_hub_index, payload_tons, speed_knots, range_nm = (float(value) for value in raw_spec)
        if not raw_hub_index.is_integer():
            raise ValueError(f"Hub index must be an integer: {raw_hub_index}.")
        hub_index = int(raw_hub_index)
        if hub_index < 1 or hub_index > hub_count:
            raise ValueError(f"Hub index {hub_index} is out of range for {hub_count} configured hubs.")
        if payload_tons <= 0.0 or speed_knots <= 0.0 or range_nm <= 0.0:
            raise ValueError("Payload, speed, and range must all be positive for --hub-vessel.")

        vessels_by_hub[hub_index].append(
            VesselSpec(payload_tons=payload_tons, speed_knots=speed_knots, range_nm=range_nm)
        )
    return vessels_by_hub


def parse_hubs(
    raw_hubs: Sequence[Sequence[float]] | None,
    raw_hub_vessels: Sequence[Sequence[float]] | None = None,
) -> list[HubLocation]:
    if not raw_hubs:
        raw_hubs = [(12.7, 121.0), (-12.4, 130.8)]
    base_hubs = [HubLocation(lat=float(lat), lon=float(lon)) for lat, lon in raw_hubs]
    if not base_hubs:
        raise ValueError("At least one hub is required.")

    vessels_by_hub = parse_hub_vessels(raw_hub_vessels, len(base_hubs))
    return [
        HubLocation(
            lat=hub.lat,
            lon=hub.lon,
            vessels=tuple(vessels_by_hub[index]),
        )
        for index, hub in enumerate(base_hubs, start=1)
    ]


def routing_range_nm_for_mode(
    hubs: Sequence[HubLocation],
    output_mode: str,
    fallback_range_nm: float,
) -> float:
    if output_mode == "range":
        return fallback_range_nm

    throughput_range_nm = max((hub.max_vessel_range_nm for hub in hubs), default=0.0)
    if throughput_range_nm <= 0.0:
        raise ValueError("Throughput mode requires at least one --hub-vessel specification.")
    return throughput_range_nm


def main() -> None:
    args = parse_args()
    if args.min_cycle_days <= 0.0:
        raise ValueError("--min-cycle-days must be positive.")
    hubs = parse_hubs(args.hub, args.hub_vessel)
    bbox = tuple(float(value) for value in args.bbox)
    routing_range_nm = routing_range_nm_for_mode(hubs, args.output_mode, args.range_nm)
    routing_bbox = expand_bbox(bbox, routing_range_nm)
    detector = load_land_polygons(args.land_shapefile, routing_bbox)
    grid = build_navigation_grid(detector, routing_bbox, args.step_km)
    routed_hubs = build_routed_hubs(
        hubs=hubs,
        routing_range_nm=routing_range_nm,
        step_km=args.step_km,
        detector=detector,
        grid=grid,
    )

    if args.output_mode == "range":
        traced_hubs = build_traced_hubs(
            routed_hubs=routed_hubs,
            range_nm=args.range_nm,
            bbox=bbox,
            detector=detector,
            grid=grid,
        )
        render_map(
            traced_hubs=traced_hubs,
            land_union=detector.union,
            bbox=bbox,
            range_nm=args.range_nm,
            output_path=args.output,
        )
        hubs_to_report: Sequence[RoutedHub] = traced_hubs
    else:
        d_min_nm = grid.min_edge_cost_km / NM_TO_KM
        throughput_field = compute_throughput_field(
            [hub.distance_field for hub in routed_hubs],
            hubs,
            d_min_nm=d_min_nm,
            min_cycle_days=args.min_cycle_days,
        )
        render_throughput_map(
            routed_hubs=routed_hubs,
            throughput_field=throughput_field,
            land_union=detector.union,
            grid=grid,
            bbox=bbox,
            contour_levels=args.throughput_contours,
            output_path=args.output,
            d_min_nm=d_min_nm,
            min_cycle_days=args.min_cycle_days,
        )
        hubs_to_report = routed_hubs

    print(f"Saved map to {args.output}")
    for hub in hubs_to_report:
        if hub.trace_origin != hub.original:
            print(
                f"{hub.label}: tracing origin adjusted from "
                f"({hub.original.lat:.3f}, {hub.original.lon:.3f}) to "
                f"({hub.trace_origin.lat:.3f}, {hub.trace_origin.lon:.3f})"
            )


if __name__ == "__main__":
    main()
