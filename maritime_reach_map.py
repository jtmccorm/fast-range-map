#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from shapely import STRtree
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

NM_TO_KM = 1.852
EARTH_RADIUS_KM = 6371.0088
DEFAULT_BBOX = (70.0, 170.0, -20.0, 40.0)
DEFAULT_LAND_SHP = REPO_ROOT / "data" / "ne_10m_land" / "ne_10m_land.shp"


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
        help="Number of rays per hub. Default: 360.",
    )
    parser.add_argument(
        "--step-km",
        type=float,
        default=8.0,
        help="Ray marching step size in kilometers. Default: 8.",
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
        default=REPO_ROOT / "maritime_reach_map.png",
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


def load_land_polygons(
    shapefile_path: Path, bbox: tuple[float, float, float, float], range_nm: float
) -> LandDetector:
    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Missing land dataset: {shapefile_path}. "
            "Expected Natural Earth land polygons in data/ne_10m_land/."
        )

    min_lon, max_lon, min_lat, max_lat = bbox
    search_margin_deg = max(6.0, range_nm / 60.0 + 2.0)
    search_region = box(
        min_lon - search_margin_deg,
        min_lat - search_margin_deg,
        max_lon + search_margin_deg,
        max_lat + search_margin_deg,
    )

    polygons: list[Polygon] = []
    reader = shapefile.Reader(str(shapefile_path))
    for shp in reader.iterShapes():
        geom = shape(shp.__geo_interface__)
        if not geom.intersects(search_region):
            continue
        clipped = geom.intersection(search_region)
        for polygon in iter_polygons(clipped):
            if polygon.is_empty:
                continue
            polygons.append(polygon)

    if not polygons:
        raise RuntimeError("No land polygons intersect the requested region.")
    return LandDetector(polygons)


def snap_hub_to_water(
    hub: HubLocation,
    detector: LandDetector,
    max_search_km: float = 35.0,
    radius_step_km: float = 2.0,
    bearing_step_deg: float = 10.0,
    clearance_km: float = 20.0,
    clearance_bearing_step_deg: float = 15.0,
    preferred_open_bearings: int = 18,
) -> HubLocation:
    def open_bearing_score(candidate: HubLocation) -> int:
        score = 0
        for bearing in np.arange(0.0, 360.0, clearance_bearing_step_deg):
            lat, lon = destination_point(candidate.lat, candidate.lon, float(bearing), clearance_km)
            if not detector.is_land(lon, lat):
                score += 1
        return score

    best_candidate: HubLocation | None = None
    best_rank: tuple[int, float] | None = None

    if not detector.is_land(hub.lon, hub.lat):
        initial_score = open_bearing_score(hub)
        if initial_score >= preferred_open_bearings:
            return hub
        best_candidate = hub
        best_rank = (initial_score, 0.0)

    radii = np.arange(radius_step_km, max_search_km + radius_step_km, radius_step_km)
    for radius_km in radii:
        for bearing in np.arange(0.0, 360.0, bearing_step_deg):
            lat, lon = destination_point(hub.lat, hub.lon, float(bearing), float(radius_km))
            if detector.is_land(lon, lat):
                continue
            candidate = HubLocation(lat=lat, lon=lon)
            rank = (open_bearing_score(candidate), -float(radius_km))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(
            f"Unable to find nearby water for hub at ({hub.lat:.3f}, {hub.lon:.3f})."
        )
    return best_candidate


def trace_ray(
    origin: HubLocation,
    bearing_deg: float,
    max_distance_km: float,
    step_km: float,
    detector: LandDetector,
) -> tuple[float, float]:
    last_water = (origin.lon, origin.lat)
    for distance_km in np.arange(step_km, max_distance_km + step_km, step_km):
        lat, lon = destination_point(origin.lat, origin.lon, bearing_deg, float(distance_km))
        if detector.is_land(lon, lat):
            break
        last_water = (lon, lat)
    return last_water


def trace_reach(
    origin: HubLocation,
    range_km: float,
    rays: int,
    step_km: float,
    detector: LandDetector,
) -> list[tuple[float, float]]:
    bearings = np.linspace(0.0, 360.0, rays, endpoint=False)
    return [
        trace_ray(origin, float(bearing), range_km, step_km, detector)
        for bearing in bearings
    ]


def build_reach_polygon(
    ray_endpoints: Sequence[tuple[float, float]],
    land_union: BaseGeometry,
    trace_origin: HubLocation,
    bbox: tuple[float, float, float, float],
) -> BaseGeometry:
    anchor = (trace_origin.lon, trace_origin.lat)
    sectors: list[BaseGeometry] = []
    wrapped_endpoints = list(ray_endpoints)
    if len(wrapped_endpoints) < 2:
        return GeometryCollection()

    for left, right in zip(wrapped_endpoints, wrapped_endpoints[1:] + wrapped_endpoints[:1]):
        triangle = Polygon([anchor, left, right])
        if triangle.is_empty or triangle.area == 0.0:
            continue
        if not triangle.is_valid:
            triangle = triangle.buffer(0)
        if not triangle.is_empty:
            sectors.append(triangle)

    if not sectors:
        return GeometryCollection()

    raw_reach = unary_union(sectors)
    if not raw_reach.is_valid:
        raw_reach = raw_reach.buffer(0)

    water_only = raw_reach.difference(land_union)
    connected = keep_component_for_anchor(water_only, Point(trace_origin.lon, trace_origin.lat))
    clipped = connected.intersection(box(bbox[0], bbox[2], bbox[1], bbox[3]))
    if not clipped.is_valid:
        clipped = clipped.buffer(0)
    return clipped


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
        label = (
            f"{hub.label}\n"
            f"({hub.original.lat:.2f}, {hub.original.lon:.2f})"
        )
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
        "Land mask: Natural Earth 1:10m land polygons | Water-constrained ray tracing",
        fontsize=8.5,
        color="#4a5568",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_traced_hubs(
    hubs: Sequence[HubLocation],
    range_nm: float,
    rays: int,
    step_km: float,
    bbox: tuple[float, float, float, float],
    detector: LandDetector,
) -> list[TracedHub]:
    traced_hubs: list[TracedHub] = []
    round_trip_km = range_nm * NM_TO_KM / 2.0
    one_way_km = range_nm * NM_TO_KM

    for index, hub in enumerate(hubs, start=1):
        trace_origin = snap_hub_to_water(hub, detector)
        round_trip_points = trace_reach(trace_origin, round_trip_km, rays, step_km, detector)
        one_way_points = trace_reach(trace_origin, one_way_km, rays, step_km, detector)

        round_trip_polygon = build_reach_polygon(
            round_trip_points, detector.union, trace_origin, bbox
        )
        one_way_polygon = build_reach_polygon(one_way_points, detector.union, trace_origin, bbox)

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
    detector = load_land_polygons(args.land_shapefile, bbox, args.range_nm)
    traced_hubs = build_traced_hubs(
        hubs=hubs,
        range_nm=args.range_nm,
        rays=args.rays,
        step_km=args.step_km,
        bbox=bbox,
        detector=detector,
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
