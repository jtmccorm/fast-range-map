#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import itertools
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
NEIGHBOR_DELTAS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


@dataclass(frozen=True)
class HubLocation:
    lat: float
    lon: float


@dataclass(frozen=True)
class TracedHub:
    index: int
    original: HubLocation
    trace_origin: HubLocation
    round_trip_polygon: BaseGeometry
    one_way_polygon: BaseGeometry

    @property
    def label(self) -> str:
        return f"Hub {self.index}"


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
        "--hub",
        action="append",
        nargs=2,
        metavar=("LAT", "LON"),
        type=float,
        help="Hub latitude and longitude. Repeat for multiple hubs.",
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
        global_row = row_start + row

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

            if delta_row != 0 and delta_col != 0:
                if not water_mask[row + delta_row, col] or not water_mask[row, col + delta_col]:
                    continue

            step_cost = movement_cost_km(grid, global_row, delta_row, delta_col)
            proposed_distance = current_distance + step_cost
            if proposed_distance >= distances[next_row, next_col] or proposed_distance > max_distance_km:
                continue

            distances[next_row, next_col] = proposed_distance
            heapq.heappush(heap, (proposed_distance, next_row, next_col))

    return distances, (row_start, row_end, col_start, col_end)


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


def render_map(
    traced_hubs: Sequence[TracedHub],
    land_union: BaseGeometry,
    bbox: tuple[float, float, float, float],
    range_nm: float,
    output_path: Path,
) -> None:
    min_lon, max_lon, min_lat, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0

    round_trip_overlap = compute_overlap([hub.round_trip_polygon for hub in traced_hubs])
    one_way_overlap = compute_overlap([hub.one_way_polygon for hub in traced_hubs])
    combined_overlap = unary_union(
        [geometry for geometry in (round_trip_overlap, one_way_overlap) if not geometry.is_empty]
    )

    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#d8eef7")

    map_region = box(min_lon, min_lat, max_lon, max_lat)
    land_in_view = land_union.intersection(map_region)
    add_geometry(
        ax,
        land_in_view,
        facecolor="#efe7d8",
        edgecolor="#49423f",
        linewidth=0.55,
        alpha=1.0,
        zorder=1,
    )

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

    for hub in traced_hubs:
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

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.grid(color="white", linewidth=0.8, linestyle="--", alpha=0.85)
    ax.set_aspect(1.0 / math.cos(math.radians(mid_lat)))

    ax.set_title(
        "Maritime Operational Reach in Southeast Asia\n"
        f"Round trip: {range_nm / 2:.0f} nm | One way: {range_nm:.0f} nm",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

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


def build_traced_hubs(
    hubs: Sequence[HubLocation],
    range_nm: float,
    step_km: float,
    bbox: tuple[float, float, float, float],
    detector: LandDetector,
    grid: NavigationGrid,
) -> list[TracedHub]:
    traced_hubs: list[TracedHub] = []
    round_trip_km = range_nm * NM_TO_KM / 2.0
    one_way_km = range_nm * NM_TO_KM

    for index, hub in enumerate(hubs, start=1):
        trace_origin, start_cell = snap_hub_to_water(hub, detector, grid)
        distances, bounds = compute_cost_distance(grid, start_cell, one_way_km)

        round_trip_polygon = build_reach_polygon(
            distances, round_trip_km, grid, bounds, detector.union, trace_origin, bbox
        )
        one_way_polygon = build_reach_polygon(
            distances, one_way_km, grid, bounds, detector.union, trace_origin, bbox
        )

        traced_hubs.append(
            TracedHub(
                index=index,
                original=hub,
                trace_origin=trace_origin,
                round_trip_polygon=round_trip_polygon,
                one_way_polygon=one_way_polygon,
            )
        )
    return traced_hubs


def parse_hubs(raw_hubs: Sequence[Sequence[float]] | None) -> list[HubLocation]:
    if not raw_hubs:
        raw_hubs = [(12.7, 121.0), (-12.4, 130.8)]
    hubs = [HubLocation(lat=float(lat), lon=float(lon)) for lat, lon in raw_hubs]
    if not hubs:
        raise ValueError("At least one hub is required.")
    return hubs


def main() -> None:
    args = parse_args()
    hubs = parse_hubs(args.hub)
    bbox = tuple(float(value) for value in args.bbox)
    routing_bbox = expand_bbox(bbox, args.range_nm)
    detector = load_land_polygons(args.land_shapefile, routing_bbox)
    grid = build_navigation_grid(detector, routing_bbox, args.step_km)
    traced_hubs = build_traced_hubs(
        hubs=hubs,
        range_nm=args.range_nm,
        step_km=args.step_km,
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

    print(f"Saved map to {args.output}")
    for hub in traced_hubs:
        if hub.trace_origin != hub.original:
            print(
                f"{hub.label}: tracing origin adjusted from "
                f"({hub.original.lat:.3f}, {hub.original.lon:.3f}) to "
                f"({hub.trace_origin.lat:.3f}, {hub.trace_origin.lon:.3f})"
            )


if __name__ == "__main__":
    main()
