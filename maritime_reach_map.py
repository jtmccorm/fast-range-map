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
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from scenario_config import (
    BoundingBox,
    HubDefinition,
    MapConfig,
    ModelConfig,
    OperationalLegendConfig,
    OutputConfig,
    RoutingConfig,
    ScenarioConfig,
    ScenarioMetadata,
    VesselDefinition,
    VisualizationConfig,
    load_config,
)

try:
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
except ImportError:
    scipy_gaussian_filter = None

NM_TO_KM = 1.852
EARTH_RADIUS_KM = 6371.0088
DEFAULT_BBOX = (70.0, 170.0, -20.0, 40.0)
DEFAULT_LAND_SHP = REPO_ROOT / "data" / "ne_10m_land" / "ne_10m_land.shp"
CACHE_DIR = REPO_ROOT / "cache"
THROUGHPUT_COLORMAPS = ("viridis", "cividis", "plasma", "inferno")
DEFAULT_THROUGHPUT_COLORMAP = "viridis"
DEFAULT_HEATMAP_ALPHA = 0.65
DEFAULT_COLOR_PERCENTILE = 97.0
DEFAULT_HEATMAP_SIGMA = 1.0
LAND_COLOR = "#cbb89d"
COASTLINE_COLOR = "#3a3a3a"
OCEAN_COLOR = "#d8e3ea"
GRID_COLOR = "#ffffff"
SPINE_COLOR = "#718096"
TICK_COLOR = "#334155"
BASE_NEIGHBOR_DELTAS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
KNIGHT_MOVE_DELTAS = (
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
    id: str | None = None
    label: str | None = None
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
    name: str | None = None

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
        if self.original.label:
            return self.original.label
        if self.original.id:
            return self.original.id.replace("_", " ").title()
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

    @property
    def center_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2.0

    def align_lon(self, lon: float) -> float:
        return align_longitude(lon, self.center_lon)

    def cell_center(self, row: int, col: int) -> HubLocation:
        return HubLocation(lat=float(self.lat_centers[row]), lon=float(self.lon_centers[col]))

    def coord_to_cell(self, lat: float, lon: float) -> tuple[int, int]:
        row = int(math.floor((lat - self.min_lat) / self.lat_step_deg))
        aligned_lon = self.align_lon(lon)
        col = int(math.floor((aligned_lon - self.min_lon) / self.lon_step_deg))
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))
        return row, col


@dataclass(frozen=True)
class ThroughputDisplayTransform:
    enabled: bool
    display_mode: str
    scale: float
    unit_label: str
    unit_abbreviation: str
    consumption_rate_tons_per_day: float | None = None

    @classmethod
    def from_config(cls, config: OperationalLegendConfig) -> "ThroughputDisplayTransform":
        if not config.enabled:
            return cls(
                enabled=False,
                display_mode="raw",
                scale=1.0,
                unit_label="tons/day",
                unit_abbreviation="tons/day",
            )
        return cls(
            enabled=True,
            display_mode=config.display_mode,
            scale=1.0 / config.consumption_rate_tons_per_day,
            unit_label=config.unit_label,
            unit_abbreviation=config.unit_abbreviation,
            consumption_rate_tons_per_day=config.consumption_rate_tons_per_day,
        )

    @property
    def colorbar_label(self) -> str:
        if not self.enabled:
            return "Throughput Capacity (tons/day)"
        if self.display_mode == "dual" and self.consumption_rate_tons_per_day is not None:
            return (
                f"Operational Capacity ({self.unit_label}; "
                f"1 {self.unit_abbreviation} = {self.consumption_rate_tons_per_day:,.0f} tons/day)"
            )
        return f"Operational Capacity ({self.unit_label})"

    @property
    def legend_text(self) -> str | None:
        if not self.enabled or self.consumption_rate_tons_per_day is None:
            return None
        lines = [
            "Operational Translation",
            self.unit_label,
            f"1 {self.unit_abbreviation} = {self.consumption_rate_tons_per_day:,.0f} tons/day",
        ]
        if self.display_mode == "dual":
            lines.append("Contour labels: operational + t/day")
        return "\n".join(lines)

    @property
    def footer_note(self) -> str | None:
        if not self.enabled or self.consumption_rate_tons_per_day is None:
            return None
        note = f"Operational display = {self.unit_abbreviation} @ {self.consumption_rate_tons_per_day:,.0f} tons/day"
        if self.display_mode == "dual":
            note += " | Dual contour labels"
        return note

    def transform_field(self, values: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return values
        return (values * self.scale).astype(np.float32, copy=False)

    def transform_levels(self, values: Sequence[float]) -> tuple[float, ...]:
        if not self.enabled:
            return tuple(float(value) for value in values)
        return tuple(float(value) * self.scale for value in values)

    def format_value(self, value: float) -> str:
        if not np.isfinite(value):
            return ""
        if not self.enabled:
            return f"{value:,.0f}"

        absolute_value = abs(float(value))
        if absolute_value >= 100.0:
            precision = 0
        elif absolute_value >= 1.0:
            precision = 1
        else:
            precision = 2
        formatted = f"{value:,.{precision}f}"
        if precision > 0:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    def format_contour_value(self, value: float) -> str:
        formatted = self.format_value(value)
        if not self.enabled:
            return formatted
        if self.display_mode == "dual":
            raw_value = value / self.scale
            return f"{formatted} {self.unit_abbreviation} / {raw_value:,.0f} t/day"
        return f"{formatted} {self.unit_abbreviation}"


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
        "--config",
        type=Path,
        help="Path to a YAML scenario configuration file. When provided, the YAML scenario drives all outputs.",
    )
    parser.add_argument(
        "--defaults-config",
        type=Path,
        help="Optional override path for the YAML defaults file used with --config.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("range", "throughput"),
        default="range",
        help="Legacy CLI mode. Use 'range' for reach maps or 'throughput' for tons/day capacity.",
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
        help="Legacy CLI output PNG path. Default: maritime_reach_map.png.",
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
    parser.add_argument(
        "--colormap",
        choices=THROUGHPUT_COLORMAPS,
        default=DEFAULT_THROUGHPUT_COLORMAP,
        help="Sequential colormap for throughput mode. Default: viridis.",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=DEFAULT_HEATMAP_ALPHA,
        help="Opacity for the throughput heatmap overlay. Default: 0.65.",
    )
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=DEFAULT_COLOR_PERCENTILE,
        help="Percentile cap used for throughput heatmap color scaling. Default: 97.",
    )
    parser.add_argument(
        "--heatmap-sigma",
        type=float,
        default=DEFAULT_HEATMAP_SIGMA,
        help="Gaussian smoothing sigma for throughput visualization only. Set to 0 to disable. Default: 1.0.",
    )
    return parser.parse_args()


def normalize_longitude(lon: float) -> float:
    return (lon + 540.0) % 360.0 - 180.0


def align_longitude(lon: float, reference_longitude: float) -> float:
    normalized = normalize_longitude(lon)
    shift_turns = math.floor(((reference_longitude - normalized) / 360.0) + 0.5)
    return normalized + 360.0 * shift_turns


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
    expanded_min_lon = min_lon - margin_deg
    expanded_max_lon = max_lon + margin_deg
    if expanded_max_lon - expanded_min_lon >= 359.8:
        midpoint = (expanded_min_lon + expanded_max_lon) / 2.0
        expanded_min_lon = midpoint - 179.9
        expanded_max_lon = midpoint + 179.9
    return (
        expanded_min_lon,
        expanded_max_lon,
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
        for offset in (-360.0, 0.0, 360.0):
            shifted = geom if offset == 0.0 else translate(geom, xoff=offset)
            if not shifted.intersects(search_region):
                continue
            clipped = shifted.intersection(search_region)
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
    aligned_hub = HubLocation(
        lat=hub.lat,
        lon=grid.align_lon(hub.lon),
        id=hub.id,
        label=hub.label,
        vessels=hub.vessels,
    )
    row, col = grid.coord_to_cell(hub.lat, hub.lon)
    if grid.water_mask[row, col] and not detector.is_land(aligned_hub.lon, aligned_hub.lat):
        return aligned_hub, (row, col)

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
    grid: NavigationGrid,
    start_cell: tuple[int, int],
    max_distance_km: float,
    neighbor_deltas: Sequence[tuple[int, int]],
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
            grid,
            water_mask,
            row,
            col,
            row_start,
            col_start,
            neighbor_deltas,
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
    routing_algorithm: str,
    knight_moves: bool,
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
        "routing": {
            "algorithm": routing_algorithm,
            "knight_moves": knight_moves,
        },
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
    neighbor_deltas: Sequence[tuple[int, int]],
) -> Iterator[tuple[int, int, float]]:
    global_row = row_offset + row
    global_col = col_offset + col

    for delta_row, delta_col in neighbor_deltas:
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


def neighbor_deltas_for_knight_moves(knight_moves: bool) -> tuple[tuple[int, int], ...]:
    if knight_moves:
        return BASE_NEIGHBOR_DELTAS + KNIGHT_MOVE_DELTAS
    return BASE_NEIGHBOR_DELTAS


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
    normalized = normalize_longitude(value)
    suffix = "E" if normalized >= 0 else "W"
    return f"{abs(normalized):.0f}°{suffix}"


def add_land_layer(
    ax: plt.Axes,
    land_union: BaseGeometry,
    bbox: tuple[float, float, float, float],
    visualization: VisualizationConfig,
    *,
    zorder: int,
) -> None:
    map_region = box(bbox[0], bbox[2], bbox[1], bbox[3])
    land_in_view = land_union.intersection(map_region)
    add_geometry(
        ax,
        land_in_view,
        facecolor=visualization.land_color,
        edgecolor=visualization.coastline_color,
        linewidth=1.0,
        alpha=1.0,
        zorder=zorder,
    )


def add_hub_markers(
    ax: plt.Axes,
    hubs: Sequence[RoutedHub],
    bbox: tuple[float, float, float, float],
    visualization: VisualizationConfig,
) -> None:
    min_lon, max_lon, min_lat, max_lat = bbox
    center_lon = (min_lon + max_lon) / 2.0
    for hub in hubs:
        display_lon = align_longitude(hub.original.lon, center_lon)
        ax.scatter(
            display_lon,
            hub.original.lat,
            s=72,
            facecolor=visualization.hub_marker_color,
            edgecolor=visualization.hub_edge_color,
            linewidth=1.0,
            zorder=6,
        )
        offset_lon = 1.1 if display_lon <= center_lon else -1.1
        offset_lat = 0.9 if hub.original.lat <= (min_lat + max_lat) / 2.0 else -0.9
        horizontal_alignment = "left" if offset_lon > 0 else "right"
        vertical_alignment = "bottom" if offset_lat > 0 else "top"
        label = hub.label
        if visualization.show_hub_coordinates:
            label = f"{label}\n({hub.original.lat:.2f}, {hub.original.lon:.2f})"
        text = ax.text(
            display_lon + offset_lon,
            hub.original.lat + offset_lat,
            label,
            fontsize=9,
            color=visualization.hub_label_color,
            ha=horizontal_alignment,
            va=vertical_alignment,
            zorder=7,
            bbox={
                "facecolor": visualization.hub_label_background_color,
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 0.4,
            },
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=1.2, foreground=visualization.hub_label_background_color)]
        )


def apply_projection_to_axes(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    projection: str,
) -> None:
    _min_lon, _max_lon, min_lat, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    if projection == "mercator":
        ax.set_aspect(1.0 / max(math.cos(math.radians(mid_lat)), 0.2))
        return
    if projection == "plate_carree":
        ax.set_aspect("auto")
        return
    raise ValueError(f"Unsupported projection: {projection}")


def style_map_axes(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    title: str,
    subtitle: str,
    projection: str,
    visualization: VisualizationConfig,
) -> None:
    min_lon, max_lon, min_lat, max_lat = bbox

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.set_axisbelow(True)
    ax.grid(color=visualization.grid_color, linewidth=0.7, linestyle="--", alpha=0.42)
    apply_projection_to_axes(ax, bbox, projection)
    title_text = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(title_text, fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.tick_params(colors=visualization.tick_color)
    for spine in ax.spines.values():
        spine.set_color(visualization.spine_color)
        spine.set_linewidth(0.9)


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


def gaussian_filter_array(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return values
    if scipy_gaussian_filter is not None:
        return scipy_gaussian_filter(values, sigma=sigma, mode="nearest")

    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    def convolve_along_axis(array: np.ndarray, axis: int) -> np.ndarray:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (radius, radius)
        padded = np.pad(array, pad_width, mode="edge")
        return np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), axis, padded)

    return convolve_along_axis(convolve_along_axis(values, 0), 1)


def build_throughput_visualization_field(
    throughput_field: np.ndarray,
    grid: NavigationGrid,
    *,
    sigma: float,
) -> np.ndarray:
    masked_throughput = np.where(grid.water_mask, throughput_field, np.nan).astype(np.float32, copy=False)
    if sigma <= 0.0:
        return masked_throughput

    filled_values = np.nan_to_num(masked_throughput, nan=0.0).astype(np.float32, copy=False)
    water_weights = grid.water_mask.astype(np.float32, copy=False)
    smoothed_values = gaussian_filter_array(filled_values, sigma)
    smoothed_weights = gaussian_filter_array(water_weights, sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = smoothed_values / smoothed_weights

    smoothed[smoothed_weights <= 1e-6] = np.nan
    smoothed[~grid.water_mask] = np.nan
    np.clip(smoothed, 0.0, None, out=smoothed)
    return smoothed.astype(np.float32, copy=False)


def compute_heatmap_vmax(
    throughput_field: np.ndarray,
    grid: NavigationGrid,
    percentile: float,
) -> float:
    visible_positive = throughput_field[
        np.isfinite(throughput_field) & grid.water_mask & (throughput_field > 0.0)
    ]
    if visible_positive.size == 0:
        return 1.0
    percentile_value = float(np.percentile(visible_positive, percentile))
    minimum_positive = float(visible_positive.min())
    return max(percentile_value, minimum_positive, float(np.finfo(np.float32).eps))


def plot_throughput_heatmap(
    ax: plt.Axes,
    throughput_field: np.ndarray,
    grid: NavigationGrid,
    *,
    cmap_name: str,
    alpha: float,
    vmax: float | None = None,
):
    positive_mask = np.isfinite(throughput_field) & grid.water_mask & (throughput_field > 0.0)
    display_field = np.ma.masked_where(~positive_mask, throughput_field)
    maximum_value = float(vmax) if vmax is not None else float(display_field.max()) if display_field.count() else 1.0
    heatmap_cmap = plt.get_cmap(cmap_name).copy()
    heatmap_cmap.set_bad(alpha=0.0)
    return ax.imshow(
        display_field,
        origin="lower",
        extent=(grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat),
        cmap=heatmap_cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=max(maximum_value, float(np.finfo(np.float32).eps)),
        alpha=alpha,
        zorder=2,
    )


def plot_throughput_contours(
    ax: plt.Axes,
    throughput_field: np.ndarray,
    grid: NavigationGrid,
    contour_levels: Sequence[float] = (50.0, 100.0, 250.0, 500.0),
    *,
    display_transform: ThroughputDisplayTransform,
    contour_color: str,
    contour_linewidth: float,
):
    contour_field = np.ma.masked_where(
        ~grid.water_mask | ~np.isfinite(throughput_field) | (throughput_field <= 0.0),
        throughput_field,
    )
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
        colors=contour_color,
        linewidths=contour_linewidth,
        alpha=0.7,
        zorder=5,
    )
    labels = ax.clabel(
        contours,
        fmt=lambda value: display_transform.format_contour_value(value),
        inline=True,
        fontsize=8,
    )
    for text in labels:
        text.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground="white")])
    return contours


def add_operational_translation_legend(
    ax: plt.Axes,
    display_transform: ThroughputDisplayTransform,
    visualization: VisualizationConfig,
) -> None:
    legend_text = display_transform.legend_text
    if legend_text is None:
        return
    ax.text(
        0.016,
        0.984,
        legend_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=visualization.tick_color,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": visualization.spine_color,
            "alpha": 0.94,
        },
        zorder=7,
    )


def render_map(
    traced_hubs: Sequence[TracedHub],
    land_union: BaseGeometry,
    bbox: tuple[float, float, float, float],
    range_nm: float,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    projection: str,
    visualization: VisualizationConfig,
    show_hubs: bool,
) -> None:
    round_trip_overlap = compute_overlap([hub.round_trip_polygon for hub in traced_hubs])
    one_way_overlap = compute_overlap([hub.one_way_polygon for hub in traced_hubs])
    combined_overlap = unary_union(
        [geometry for geometry in (round_trip_overlap, one_way_overlap) if not geometry.is_empty]
    )

    with matplotlib.rc_context({"font.family": visualization.font_family}):
        fig, ax = plt.subplots(
            figsize=(visualization.figure_width, visualization.figure_height),
            dpi=visualization.dpi,
        )
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.11)
        ax.set_facecolor(visualization.ocean_color)

        add_land_layer(ax, land_union, bbox, visualization, zorder=1)

        for hub in traced_hubs:
            add_geometry(
                ax,
                hub.one_way_polygon,
                facecolor=visualization.range_one_way_fill_color,
                edgecolor=visualization.range_one_way_edge_color,
                linewidth=0.6,
                alpha=visualization.range_one_way_alpha,
                zorder=2,
            )
        for hub in traced_hubs:
            add_geometry(
                ax,
                hub.round_trip_polygon,
                facecolor=visualization.range_round_trip_fill_color,
                edgecolor=visualization.range_round_trip_edge_color,
                linewidth=0.7,
                alpha=visualization.range_round_trip_alpha,
                zorder=3,
            )
        add_geometry(
            ax,
            combined_overlap,
            facecolor=visualization.overlap_fill_color,
            edgecolor=visualization.overlap_edge_color,
            linewidth=0.8,
            alpha=visualization.overlap_alpha,
            zorder=4,
        )

        if show_hubs:
            add_hub_markers(ax, traced_hubs, bbox, visualization)
        style_map_axes(
            ax,
            bbox,
            title=title,
            subtitle=subtitle,
            projection=projection,
            visualization=visualization,
        )

        legend_items = [
            Patch(
                facecolor=visualization.range_round_trip_fill_color,
                edgecolor=visualization.range_round_trip_edge_color,
                alpha=visualization.range_round_trip_alpha,
                label="Round Trip Range",
            ),
            Patch(
                facecolor=visualization.range_one_way_fill_color,
                edgecolor=visualization.range_one_way_edge_color,
                alpha=visualization.range_one_way_alpha,
                label="One Way Range",
            ),
            Patch(
                facecolor=visualization.overlap_fill_color,
                edgecolor=visualization.overlap_edge_color,
                alpha=visualization.overlap_alpha,
                label="Overlap Between Hubs",
            ),
        ]
        if show_hubs:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=visualization.hub_marker_color,
                    markeredgecolor=visualization.hub_edge_color,
                    markersize=9,
                    label="Hub",
                )
            )
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
    colormap: str,
    heatmap_alpha: float,
    color_percentile: float,
    heatmap_sigma: float,
    *,
    title: str,
    subtitle: str,
    projection: str,
    operational_legend: OperationalLegendConfig,
    visualization: VisualizationConfig,
    show_hubs: bool,
) -> None:
    display_transform = ThroughputDisplayTransform.from_config(operational_legend)

    with matplotlib.rc_context({"font.family": visualization.font_family}):
        fig, ax = plt.subplots(
            figsize=(visualization.figure_width, visualization.figure_height),
            dpi=visualization.dpi,
        )
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.11)
        ax.set_facecolor(visualization.ocean_color)

        visualization_field = build_throughput_visualization_field(
            throughput_field,
            grid,
            sigma=heatmap_sigma,
        )
        display_field = display_transform.transform_field(visualization_field)
        heatmap = plot_throughput_heatmap(
            ax,
            display_field,
            grid,
            cmap_name=colormap,
            alpha=heatmap_alpha,
            vmax=compute_heatmap_vmax(display_field, grid, color_percentile),
        )
        add_land_layer(ax, land_union, bbox, visualization, zorder=3)
        plot_throughput_contours(
            ax,
            display_field,
            grid,
            display_transform.transform_levels(contour_levels),
            display_transform=display_transform,
            contour_color=visualization.throughput_contour_color,
            contour_linewidth=visualization.throughput_contour_linewidth,
        )
        if show_hubs:
            add_hub_markers(ax, routed_hubs, bbox, visualization)
        style_map_axes(
            ax,
            bbox,
            title=title,
            subtitle=subtitle,
            projection=projection,
            visualization=visualization,
        )
        add_operational_translation_legend(ax, display_transform, visualization)

        colorbar = fig.colorbar(heatmap, ax=ax, pad=0.02, shrink=0.86)
        colorbar.set_label(display_transform.colorbar_label, fontsize=11)
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: display_transform.format_value(value))
        )
        colorbar.ax.tick_params(labelsize=9, colors=visualization.tick_color)
        colorbar.outline.set_edgecolor(visualization.spine_color)
        colorbar.outline.set_linewidth(0.8)

        footer_note = display_transform.footer_note
        fig.text(
            0.013,
            0.012,
            (
                "Water-routed throughput model | "
                f"d_min = {d_min_nm:.1f} nm | "
                f"Min cycle = {min_cycle_days:.1f} day(s) | "
                f"Color cap = P{color_percentile:.0f} | "
                f"Sigma = {heatmap_sigma:.1f} | "
                "Vessel contribution cutoff at half of listed range"
                + (f" | {footer_note}" if footer_note else "")
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
    *,
    routing_algorithm: str,
    knight_moves: bool,
    use_distance_cache: bool,
) -> list[RoutedHub]:
    routed_hubs: list[RoutedHub] = []
    max_distance_km = routing_range_nm * NM_TO_KM
    neighbor_deltas = neighbor_deltas_for_knight_moves(knight_moves)

    for index, hub in enumerate(hubs, start=1):
        trace_origin, start_cell = snap_hub_to_water(hub, detector, grid)
        cache_path = distance_cache_path(
            index,
            hub,
            trace_origin,
            start_cell,
            grid,
            step_km,
            max_distance_km,
            routing_algorithm,
            knight_moves,
        )
        cached_distance_field = load_cached_distance_field(cache_path) if use_distance_cache else None
        if cached_distance_field is None:
            distances, bounds = compute_cost_distance(
                grid,
                start_cell,
                max_distance_km,
                neighbor_deltas,
            )
            if use_distance_cache:
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


def validate_legacy_args(args: argparse.Namespace) -> None:
    if args.min_cycle_days <= 0.0:
        raise ValueError("--min-cycle-days must be positive.")
    if not 0.0 <= args.heatmap_alpha <= 1.0:
        raise ValueError("--heatmap-alpha must be between 0 and 1.")
    if not 0.0 < args.color_percentile <= 100.0:
        raise ValueError("--color-percentile must be greater than 0 and at most 100.")
    if args.heatmap_sigma < 0.0:
        raise ValueError("--heatmap-sigma must be non-negative.")


def default_visualization_config(
    *,
    throughput_colormap: str = DEFAULT_THROUGHPUT_COLORMAP,
    throughput_heatmap_alpha: float = DEFAULT_HEATMAP_ALPHA,
    throughput_color_percentile: float = DEFAULT_COLOR_PERCENTILE,
    throughput_heatmap_sigma: float = DEFAULT_HEATMAP_SIGMA,
) -> VisualizationConfig:
    return VisualizationConfig(
        land_color=LAND_COLOR,
        coastline_color=COASTLINE_COLOR,
        ocean_color=OCEAN_COLOR,
        grid_color=GRID_COLOR,
        spine_color=SPINE_COLOR,
        tick_color=TICK_COLOR,
        hub_marker_color="#c1121f",
        hub_edge_color="#000000",
        hub_label_color="#1f2933",
        hub_label_background_color="#ffffff",
        font_family="DejaVu Sans",
        show_hub_coordinates=True,
        figure_width=16.0,
        figure_height=10.0,
        dpi=180,
        range_one_way_fill_color="#f4a261",
        range_one_way_edge_color="#cf7c1d",
        range_one_way_alpha=0.30,
        range_round_trip_fill_color="#4f83cc",
        range_round_trip_edge_color="#2c5ea8",
        range_round_trip_alpha=0.36,
        overlap_fill_color="#7b2cbf",
        overlap_edge_color="#5a189a",
        overlap_alpha=0.42,
        throughput_colormap=throughput_colormap,
        throughput_heatmap_alpha=throughput_heatmap_alpha,
        throughput_color_percentile=throughput_color_percentile,
        throughput_heatmap_sigma=throughput_heatmap_sigma,
        throughput_contour_color="#000000",
        throughput_contour_linewidth=0.8,
    )


def parse_legacy_hub_vessels(
    raw_hub_vessels: Sequence[Sequence[float]] | None,
    hub_count: int,
) -> tuple[dict[str, VesselDefinition], dict[int, dict[str, int]]]:
    vessel_definitions: dict[str, VesselDefinition] = {}
    vessels_by_hub = {hub_index: {} for hub_index in range(1, hub_count + 1)}
    vessel_sequence_by_hub = {hub_index: 0 for hub_index in range(1, hub_count + 1)}

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

        vessel_sequence_by_hub[hub_index] += 1
        vessel_id = f"hub{hub_index}_vessel{vessel_sequence_by_hub[hub_index]}"
        vessel_definitions[vessel_id] = VesselDefinition(
            id=vessel_id,
            payload_tons=payload_tons,
            speed_knots=speed_knots,
            range_nm=range_nm,
        )
        vessels_by_hub[hub_index][vessel_id] = 1

    return vessel_definitions, vessels_by_hub


def build_legacy_scenario_config(args: argparse.Namespace) -> ScenarioConfig:
    raw_hubs = args.hub or [(12.7, 121.0), (-12.4, 130.8)]
    if not raw_hubs:
        raise ValueError("At least one hub is required.")

    vessel_definitions, vessels_by_hub = parse_legacy_hub_vessels(args.hub_vessel, len(raw_hubs))
    bbox = BoundingBox(*[float(value) for value in args.bbox])
    output_type = "range_map" if args.output_mode == "range" else "throughput_field"

    if output_type == "range_map":
        output_title = "Maritime Operational Reach in Southeast Asia"
        output_subtitle = f"Round trip: {args.range_nm / 2:.0f} nm | One way: {args.range_nm:.0f} nm"
        contour_levels: tuple[float, ...] = ()
        color_scheme = None
    else:
        output_title = "Maritime Throughput Capacity in Southeast Asia"
        output_subtitle = "Sustainment capacity from vessel transport strength and navigable delivery distance"
        contour_levels = tuple(float(value) for value in args.throughput_contours)
        color_scheme = args.colormap

    hubs = tuple(
        HubDefinition(
            id=f"hub_{index}",
            label=f"Hub {index}",
            lat=float(lat),
            lon=float(lon),
            vessels=vessels_by_hub[index],
        )
        for index, (lat, lon) in enumerate(raw_hubs, start=1)
    )
    return ScenarioConfig(
        source_path=REPO_ROOT / "legacy_cli.yaml",
        defaults_path=None,
        scenario=ScenarioMetadata(
            name="legacy_cli",
            title=output_title,
            subtitle=output_subtitle,
        ),
        map=MapConfig(
            grid_km=float(args.step_km),
            projection="mercator",
            bounding_box=bbox,
            land_shapefile=args.land_shapefile,
        ),
        model=ModelConfig(
            range_nm=float(args.range_nm),
            distance_cache=True,
            min_cycle_days=float(args.min_cycle_days),
            routing=RoutingConfig(algorithm="dijkstra", knight_moves=True),
        ),
        vessels=vessel_definitions,
        hubs=hubs,
        visualization=default_visualization_config(
            throughput_colormap=args.colormap,
            throughput_heatmap_alpha=float(args.heatmap_alpha),
            throughput_color_percentile=float(args.color_percentile),
            throughput_heatmap_sigma=float(args.heatmap_sigma),
        ),
        outputs=(
            OutputConfig(
                id="legacy_output",
                type=output_type,
                title=output_title,
                subtitle=output_subtitle,
                bounding_box=bbox,
                filename=args.output,
                show_hubs=True,
                color_scheme=color_scheme,
                contour_levels=contour_levels,
            ),
        ),
    )


def runtime_hubs_from_config(config: ScenarioConfig) -> list[HubLocation]:
    hubs: list[HubLocation] = []
    for hub in config.hubs:
        vessels: list[VesselSpec] = []
        for vessel_id, count in hub.vessels.items():
            vessel_definition = config.vessels[vessel_id]
            for _ in range(count):
                vessels.append(
                    VesselSpec(
                        payload_tons=vessel_definition.payload_tons,
                        speed_knots=vessel_definition.speed_knots,
                        range_nm=vessel_definition.range_nm,
                        name=vessel_definition.id,
                    )
                )
        hubs.append(
            HubLocation(
                lat=hub.lat,
                lon=hub.lon,
                id=hub.id,
                label=hub.label,
                vessels=tuple(vessels),
            )
        )
    return hubs


def combine_output_bboxes(outputs: Sequence[OutputConfig]) -> tuple[float, float, float, float]:
    if not outputs:
        raise ValueError("At least one output is required.")

    west, east, south, north = outputs[0].bounding_box.as_unwrapped_tuple()
    for output in outputs[1:]:
        output_west, output_east, output_south, output_north = output.bounding_box.as_unwrapped_tuple(
            reference_longitude=(west + east) / 2.0
        )
        west = min(west, output_west)
        east = max(east, output_east)
        south = min(south, output_south)
        north = max(north, output_north)
    return (west, east, south, north)


def hubs_share_same_coordinates(
    left: HubLocation,
    right: HubLocation,
    *,
    tolerance: float = 1e-6,
) -> bool:
    return (
        abs(left.lat - right.lat) <= tolerance
        and abs(normalize_longitude(left.lon - right.lon)) <= tolerance
    )


def max_fleet_range_nm(hubs: Sequence[HubLocation]) -> float:
    return max((hub.max_vessel_range_nm for hub in hubs), default=0.0)


def range_nm_for_output(output: OutputConfig, config: ScenarioConfig) -> float:
    if output.vessel is not None:
        return config.vessels[output.vessel].range_nm
    return config.model.range_nm


def routing_range_nm_for_outputs(config: ScenarioConfig, hubs: Sequence[HubLocation]) -> float:
    routing_ranges: list[float] = []
    for output in config.outputs:
        if output.type == "range_map":
            routing_ranges.append(range_nm_for_output(output, config))
            continue
        throughput_range_nm = max_fleet_range_nm(hubs)
        if throughput_range_nm <= 0.0:
            raise ValueError("Throughput outputs require at least one configured vessel.")
        routing_ranges.append(throughput_range_nm)
    if not routing_ranges:
        raise ValueError("At least one output is required.")
    return max(routing_ranges)


def select_routed_hubs_for_output(
    routed_hubs: Sequence[RoutedHub],
    output: OutputConfig,
) -> list[RoutedHub]:
    if output.vessel is None:
        return list(routed_hubs)
    return [
        hub
        for hub in routed_hubs
        if any(vessel.name == output.vessel for vessel in hub.original.vessels)
    ]


def generate_outputs(config: ScenarioConfig) -> tuple[list[Path], list[RoutedHub]]:
    hubs = runtime_hubs_from_config(config)
    routing_range_nm = routing_range_nm_for_outputs(config, hubs)
    routing_bbox = expand_bbox(combine_output_bboxes(config.outputs), routing_range_nm)
    detector = load_land_polygons(config.map.land_shapefile, routing_bbox)
    grid = build_navigation_grid(detector, routing_bbox, config.map.grid_km)
    routed_hubs = build_routed_hubs(
        hubs=hubs,
        routing_range_nm=routing_range_nm,
        step_km=config.map.grid_km,
        detector=detector,
        grid=grid,
        routing_algorithm=config.model.routing.algorithm,
        knight_moves=config.model.routing.knight_moves,
        use_distance_cache=config.model.distance_cache,
    )

    throughput_field: np.ndarray | None = None
    d_min_nm = grid.min_edge_cost_km / NM_TO_KM
    if any(output.type == "throughput_field" for output in config.outputs):
        throughput_field = compute_throughput_field(
            [hub.distance_field for hub in routed_hubs],
            hubs,
            d_min_nm=d_min_nm,
            min_cycle_days=config.model.min_cycle_days,
        )

    output_paths: list[Path] = []
    routing_center_lon = (grid.min_lon + grid.max_lon) / 2.0
    for output in config.outputs:
        output_path = config.resolve_output_path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bbox = output.bounding_box.as_unwrapped_tuple(reference_longitude=routing_center_lon)

        if output.type == "range_map":
            selected_hubs = select_routed_hubs_for_output(routed_hubs, output)
            range_nm = range_nm_for_output(output, config)
            subtitle = output.subtitle or f"Round trip: {range_nm / 2:.0f} nm | One way: {range_nm:.0f} nm"
            traced_hubs = build_traced_hubs(
                routed_hubs=selected_hubs,
                range_nm=range_nm,
                bbox=bbox,
                detector=detector,
                grid=grid,
            )
            render_map(
                traced_hubs=traced_hubs,
                land_union=detector.union,
                bbox=bbox,
                range_nm=range_nm,
                output_path=output_path,
                title=output.title,
                subtitle=subtitle,
                projection=config.map.projection,
                visualization=config.visualization,
                show_hubs=output.show_hubs,
            )
        else:
            if throughput_field is None:
                raise RuntimeError("Throughput field was not computed for throughput output generation.")
            subtitle = (
                output.subtitle
                or "Sustainment capacity from vessel transport strength and navigable delivery distance"
            )
            render_throughput_map(
                routed_hubs=routed_hubs,
                throughput_field=throughput_field,
                land_union=detector.union,
                grid=grid,
                bbox=bbox,
                contour_levels=output.contour_levels,
                output_path=output_path,
                d_min_nm=d_min_nm,
                min_cycle_days=config.model.min_cycle_days,
                colormap=output.color_scheme or config.visualization.throughput_colormap,
                heatmap_alpha=config.visualization.throughput_heatmap_alpha,
                color_percentile=config.visualization.throughput_color_percentile,
                heatmap_sigma=config.visualization.throughput_heatmap_sigma,
                title=output.title,
                subtitle=subtitle,
                projection=config.map.projection,
                operational_legend=output.operational_legend,
                visualization=config.visualization,
                show_hubs=output.show_hubs,
            )
        output_paths.append(output_path)

    return output_paths, routed_hubs


def main() -> None:
    args = parse_args()
    if args.defaults_config is not None and args.config is None:
        raise ValueError("--defaults-config requires --config.")

    if args.config is not None:
        config = load_config(args.config, defaults_path=args.defaults_config)
    else:
        validate_legacy_args(args)
        config = build_legacy_scenario_config(args)

    output_paths, routed_hubs = generate_outputs(config)
    for output_path in output_paths:
        print(f"Saved map to {output_path}")
    for hub in routed_hubs:
        if not hubs_share_same_coordinates(hub.trace_origin, hub.original):
            print(
                f"{hub.label}: tracing origin adjusted from "
                f"({hub.original.lat:.3f}, {hub.original.lon:.3f}) to "
                f"({hub.trace_origin.lat:.3f}, {hub.trace_origin.lon:.3f})"
            )


if __name__ == "__main__":
    main()
