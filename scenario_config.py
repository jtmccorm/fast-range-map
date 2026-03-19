from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime when YAML loading is requested
    yaml = None


SUPPORTED_OUTPUT_TYPES = frozenset({"range_map", "throughput_field"})
SUPPORTED_PROJECTIONS = frozenset({"mercator", "plate_carree"})
SUPPORTED_ROUTING_ALGORITHMS = frozenset({"dijkstra"})
SUPPORTED_OPERATIONAL_UNIT_TYPES = frozenset({"ibct", "abct", "battalion", "custom"})
SUPPORTED_OPERATIONAL_DISPLAY_MODES = frozenset({"translated", "dual"})

DEFAULT_LAND_COLOR = "#cbb89d"
DEFAULT_COASTLINE_COLOR = "#3a3a3a"
DEFAULT_OCEAN_COLOR = "#d8e3ea"
DEFAULT_GRID_COLOR = "#ffffff"
DEFAULT_SPINE_COLOR = "#718096"
DEFAULT_TICK_COLOR = "#334155"
DEFAULT_HUB_MARKER_COLOR = "#c1121f"
DEFAULT_HUB_EDGE_COLOR = "#000000"
DEFAULT_HUB_LABEL_COLOR = "#1f2933"
DEFAULT_HUB_LABEL_BACKGROUND_COLOR = "#ffffff"
DEFAULT_RANGE_ONE_WAY_FILL_COLOR = "#f4a261"
DEFAULT_RANGE_ONE_WAY_EDGE_COLOR = "#cf7c1d"
DEFAULT_RANGE_ROUND_TRIP_FILL_COLOR = "#4f83cc"
DEFAULT_RANGE_ROUND_TRIP_EDGE_COLOR = "#2c5ea8"
DEFAULT_OVERLAP_FILL_COLOR = "#7b2cbf"
DEFAULT_OVERLAP_EDGE_COLOR = "#5a189a"
DEFAULT_OPERATIONAL_UNIT_RATES_TPD = {
    "ibct": 300.0,
    "abct": 700.0,
    "battalion": 75.0,
    "custom": 100.0,
}
DEFAULT_OPERATIONAL_UNIT_LABELS = {
    "ibct": "IBCT days of sustainment",
    "abct": "ABCT days of sustainment",
    "battalion": "Battalion days of sustainment",
    "custom": "Custom sustainment equivalents",
}
DEFAULT_OPERATIONAL_UNIT_ABBREVIATIONS = {
    "ibct": "IBCT",
    "abct": "ABCT",
    "battalion": "BN",
    "custom": "EQ",
}


@dataclass(frozen=True)
class BoundingBox:
    west: float
    east: float
    south: float
    north: float

    def __post_init__(self) -> None:
        values = (self.west, self.east, self.south, self.north)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding box values must be finite.")
        if self.longitude_span_deg >= 360.0:
            raise ValueError("Bounding box longitude span must be greater than 0 and less than 360 degrees.")
        if self.south >= self.north:
            raise ValueError("Bounding box south must be less than north.")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.east, self.south, self.north)

    @property
    def crosses_antimeridian(self) -> bool:
        return self.east <= self.west

    @property
    def longitude_span_deg(self) -> float:
        east = self.east + 360.0 if self.crosses_antimeridian else self.east
        return east - self.west

    def as_unwrapped_tuple(
        self,
        *,
        reference_longitude: float | None = None,
    ) -> tuple[float, float, float, float]:
        west = self.west
        east = self.west + self.longitude_span_deg
        if reference_longitude is not None:
            midpoint = (west + east) / 2.0
            shift_turns = math.floor(((reference_longitude - midpoint) / 360.0) + 0.5)
            west += 360.0 * shift_turns
            east += 360.0 * shift_turns
        return (west, east, self.south, self.north)

    @classmethod
    def from_mapping(cls, value: Any, field_name: str) -> "BoundingBox":
        mapping = _require_mapping(value, field_name)
        try:
            west = float(mapping["west"])
            east = float(mapping["east"])
            south = float(mapping["south"])
            north = float(mapping["north"])
        except KeyError as exc:
            raise ValueError(f"{field_name} is missing required key '{exc.args[0]}'.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} values must be numeric.") from exc

        try:
            return cls(west=west, east=east, south=south, north=north)
        except ValueError as exc:
            raise ValueError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True)
class ScenarioMetadata:
    name: str
    title: str
    subtitle: str


@dataclass(frozen=True)
class MapConfig:
    grid_km: float
    projection: str
    bounding_box: BoundingBox
    land_shapefile: Path


@dataclass(frozen=True)
class RoutingConfig:
    algorithm: str
    knight_moves: bool


@dataclass(frozen=True)
class ModelConfig:
    range_nm: float
    distance_cache: bool
    min_cycle_days: float
    routing: RoutingConfig


@dataclass(frozen=True)
class VesselDefinition:
    id: str
    payload_tons: float
    speed_knots: float
    range_nm: float


@dataclass(frozen=True)
class HubDefinition:
    id: str
    label: str
    lat: float
    lon: float
    vessels: dict[str, int]


@dataclass(frozen=True)
class VisualizationConfig:
    land_color: str
    coastline_color: str
    ocean_color: str
    grid_color: str
    spine_color: str
    tick_color: str
    hub_marker_color: str
    hub_edge_color: str
    hub_label_color: str
    hub_label_background_color: str
    font_family: str
    show_hub_coordinates: bool
    figure_width: float
    figure_height: float
    dpi: int
    range_one_way_fill_color: str
    range_one_way_edge_color: str
    range_one_way_alpha: float
    range_round_trip_fill_color: str
    range_round_trip_edge_color: str
    range_round_trip_alpha: float
    overlap_fill_color: str
    overlap_edge_color: str
    overlap_alpha: float
    throughput_colormap: str
    throughput_heatmap_alpha: float
    throughput_color_percentile: float
    throughput_heatmap_sigma: float
    throughput_contour_color: str
    throughput_contour_linewidth: float


@dataclass(frozen=True)
class OperationalLegendConfig:
    enabled: bool
    unit_type: str
    display_mode: str
    consumption_rate_tons_per_day: float
    unit_label: str
    unit_abbreviation: str


@dataclass(frozen=True)
class OutputConfig:
    id: str
    type: str
    title: str
    subtitle: str
    bounding_box: BoundingBox
    filename: Path
    show_hubs: bool
    color_scheme: str | None = None
    contour_levels: tuple[float, ...] = ()
    operational_legend: OperationalLegendConfig = OperationalLegendConfig(
        enabled=False,
        unit_type="ibct",
        display_mode="translated",
        consumption_rate_tons_per_day=DEFAULT_OPERATIONAL_UNIT_RATES_TPD["ibct"],
        unit_label=DEFAULT_OPERATIONAL_UNIT_LABELS["ibct"],
        unit_abbreviation=DEFAULT_OPERATIONAL_UNIT_ABBREVIATIONS["ibct"],
    )
    vessel: str | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    source_path: Path
    defaults_path: Path | None
    scenario: ScenarioMetadata
    map: MapConfig
    model: ModelConfig
    vessels: dict[str, VesselDefinition]
    hubs: tuple[HubDefinition, ...]
    visualization: VisualizationConfig
    outputs: tuple[OutputConfig, ...]

    def resolve_output_path(self, output: OutputConfig) -> Path:
        if output.filename.is_absolute():
            return output.filename
        repo_root = Path(__file__).resolve().parent
        if output.filename.parts and output.filename.parts[0] == "output":
            return repo_root / output.filename
        return repo_root / "output" / output.filename


def load_config(path: str | Path, defaults_path: str | Path | None = None) -> ScenarioConfig:
    source_path = Path(path).expanduser().resolve()
    raw_scenario = _load_yaml_mapping(source_path)
    defaults_reference = raw_scenario.pop("defaults", None)
    resolved_defaults_path = _resolve_defaults_path(
        source_path=source_path,
        explicit_defaults_path=defaults_path,
        defaults_reference=defaults_reference,
    )
    raw_defaults = _load_yaml_mapping(resolved_defaults_path) if resolved_defaults_path else {}

    raw_defaults = _resolve_known_relative_paths(raw_defaults, resolved_defaults_path.parent if resolved_defaults_path else None)
    raw_scenario = _resolve_known_relative_paths(raw_scenario, source_path.parent)

    merged = deep_merge(raw_defaults, raw_scenario)
    output_defaults = _require_mapping(merged.pop("output_defaults", {}), "output_defaults")
    outputs = merged.get("outputs")
    if outputs is None:
        raise ValueError("Scenario configuration must define an 'outputs' list.")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Scenario configuration must define at least one output.")

    merged["outputs"] = [
        deep_merge(output_defaults, _require_mapping(output_value, f"outputs[{index}]"))
        for index, output_value in enumerate(outputs)
    ]
    return parse_config_mapping(merged, source_path=source_path, defaults_path=resolved_defaults_path)


def parse_config_mapping(
    raw_config: Mapping[str, Any],
    *,
    source_path: Path,
    defaults_path: Path | None = None,
) -> ScenarioConfig:
    raw = _require_mapping(raw_config, "config")
    scenario = _parse_scenario_metadata(raw.get("scenario"))
    map_config = _parse_map_config(raw.get("map"))
    model = _parse_model_config(raw.get("model"))
    vessels = _parse_vessels(raw.get("vessels"))
    hubs = _parse_hubs(raw.get("hubs"))
    visualization = _parse_visualization_config(raw.get("visualization"))
    outputs = _parse_outputs(raw.get("outputs"), scenario=scenario, map_config=map_config)

    for hub in hubs:
        for vessel_id, count in hub.vessels.items():
            if vessel_id not in vessels:
                raise ValueError(f"Hub '{hub.id}' references unknown vessel type '{vessel_id}'.")
            if count <= 0:
                raise ValueError(f"Hub '{hub.id}' has a non-positive count for vessel '{vessel_id}'.")

    for output in outputs:
        if output.vessel is not None:
            if output.vessel not in vessels:
                raise ValueError(f"Output '{output.id}' references unknown vessel type '{output.vessel}'.")
            if not any(output.vessel in hub.vessels for hub in hubs):
                raise ValueError(
                    f"Output '{output.id}' requests vessel '{output.vessel}', but no configured hub carries that vessel."
                )
        if output.type == "throughput_field" and not any(hub.vessels for hub in hubs):
            raise ValueError("Throughput outputs require at least one hub with configured vessels.")

    return ScenarioConfig(
        source_path=source_path,
        defaults_path=defaults_path,
        scenario=scenario,
        map=map_config,
        model=model,
        vessels=vessels,
        hubs=tuple(hubs),
        visualization=visualization,
        outputs=tuple(outputs),
    )


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged: dict[str, Any] = {str(key): copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            existing = merged.get(str(key))
            merged[str(key)] = deep_merge(existing, value) if existing is not None else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    yaml_module = _require_yaml()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml_module.safe_load(handle) or {}
    return _require_mapping(parsed, str(path))


def _resolve_defaults_path(
    *,
    source_path: Path,
    explicit_defaults_path: str | Path | None,
    defaults_reference: Any,
) -> Path | None:
    if explicit_defaults_path is not None:
        return Path(explicit_defaults_path).expanduser().resolve()

    if defaults_reference is not None:
        defaults_path = Path(_require_string(defaults_reference, "defaults")).expanduser()
        if not defaults_path.is_absolute():
            defaults_path = (source_path.parent / defaults_path).resolve()
        return defaults_path

    local_defaults = source_path.parent / "defaults.yaml"
    repo_defaults = Path(__file__).resolve().parent / "defaults.yaml"
    if local_defaults.exists():
        return local_defaults.resolve()
    if repo_defaults.exists():
        return repo_defaults.resolve()
    return None


def _resolve_known_relative_paths(
    raw_config: Mapping[str, Any],
    base_dir: Path | None,
) -> dict[str, Any]:
    resolved = _require_mapping(raw_config, "config")
    map_config = resolved.get("map")
    if base_dir is None or not isinstance(map_config, Mapping):
        return resolved

    land_shapefile = map_config.get("land_shapefile")
    if land_shapefile is None:
        return resolved

    path_value = Path(_require_string(land_shapefile, "map.land_shapefile")).expanduser()
    if not path_value.is_absolute():
        path_value = (base_dir / path_value).resolve()
    resolved["map"] = dict(map_config)
    resolved["map"]["land_shapefile"] = str(path_value)
    return resolved


def _parse_scenario_metadata(value: Any) -> ScenarioMetadata:
    mapping = _require_mapping(value, "scenario")
    name = _require_string(mapping.get("name"), "scenario.name")
    title = _optional_string(mapping.get("title"), default=name.replace("_", " ").title())
    subtitle = _optional_string(mapping.get("subtitle"), default="")
    return ScenarioMetadata(name=name, title=title, subtitle=subtitle)


def _parse_map_config(value: Any) -> MapConfig:
    mapping = _require_mapping(value, "map")
    bounding_box_value = mapping.get("bounding_box", mapping.get("bbox"))
    if bounding_box_value is None:
        raise ValueError("map.bounding_box is required.")
    projection = _normalize_choice(
        mapping.get("projection"),
        "map.projection",
        supported=SUPPORTED_PROJECTIONS,
        default="mercator",
    )
    land_shapefile = Path(_require_string(mapping.get("land_shapefile"), "map.land_shapefile")).expanduser()
    return MapConfig(
        grid_km=_positive_float(mapping.get("grid_km"), "map.grid_km"),
        projection=projection,
        bounding_box=BoundingBox.from_mapping(bounding_box_value, "map.bounding_box"),
        land_shapefile=land_shapefile,
    )


def _parse_model_config(value: Any) -> ModelConfig:
    mapping = _require_mapping(value, "model")
    routing_mapping = _require_mapping(mapping.get("routing"), "model.routing")
    return ModelConfig(
        range_nm=_positive_float(mapping.get("range_nm"), "model.range_nm"),
        distance_cache=_boolean(mapping.get("distance_cache"), "model.distance_cache", default=True),
        min_cycle_days=_positive_float(mapping.get("min_cycle_days"), "model.min_cycle_days"),
        routing=RoutingConfig(
            algorithm=_normalize_choice(
                routing_mapping.get("algorithm"),
                "model.routing.algorithm",
                supported=SUPPORTED_ROUTING_ALGORITHMS,
                default="dijkstra",
            ),
            knight_moves=_boolean(
                routing_mapping.get("knight_moves"),
                "model.routing.knight_moves",
                default=True,
            ),
        ),
    )


def _parse_vessels(value: Any) -> dict[str, VesselDefinition]:
    mapping = _require_mapping(value, "vessels")
    vessels: dict[str, VesselDefinition] = {}
    for vessel_id, vessel_value in mapping.items():
        field_name = f"vessels.{vessel_id}"
        vessel_mapping = _require_mapping(vessel_value, field_name)
        vessels[str(vessel_id)] = VesselDefinition(
            id=str(vessel_id),
            payload_tons=_positive_float(vessel_mapping.get("payload_tons"), f"{field_name}.payload_tons"),
            speed_knots=_positive_float(vessel_mapping.get("speed_knots"), f"{field_name}.speed_knots"),
            range_nm=_positive_float(vessel_mapping.get("range_nm"), f"{field_name}.range_nm"),
        )
    return vessels


def _parse_hubs(value: Any) -> list[HubDefinition]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("hubs must be a list of mappings.")

    hubs: list[HubDefinition] = []
    for index, hub_value in enumerate(value):
        field_name = f"hubs[{index}]"
        hub_mapping = _require_mapping(hub_value, field_name)
        vessels_mapping = _require_mapping(hub_mapping.get("vessels", {}), f"{field_name}.vessels")
        hub_id = _require_string(hub_mapping.get("id"), f"{field_name}.id")
        label = _optional_string(hub_mapping.get("label"), default=hub_id.replace("_", " ").title())
        hubs.append(
            HubDefinition(
                id=hub_id,
                label=label,
                lat=_float(hub_mapping.get("lat"), f"{field_name}.lat"),
                lon=_float(hub_mapping.get("lon"), f"{field_name}.lon"),
                vessels={
                    str(vessel_id): _positive_int(count, f"{field_name}.vessels.{vessel_id}")
                    for vessel_id, count in vessels_mapping.items()
                },
            )
        )
    if not hubs:
        raise ValueError("At least one hub is required.")
    return hubs


def _parse_visualization_config(value: Any) -> VisualizationConfig:
    mapping = _require_mapping(value, "visualization")
    return VisualizationConfig(
        land_color=_optional_string(mapping.get("land_color"), default=DEFAULT_LAND_COLOR),
        coastline_color=_optional_string(mapping.get("coastline_color"), default=DEFAULT_COASTLINE_COLOR),
        ocean_color=_optional_string(mapping.get("ocean_color"), default=DEFAULT_OCEAN_COLOR),
        grid_color=_optional_string(mapping.get("grid_color"), default=DEFAULT_GRID_COLOR),
        spine_color=_optional_string(mapping.get("spine_color"), default=DEFAULT_SPINE_COLOR),
        tick_color=_optional_string(mapping.get("tick_color"), default=DEFAULT_TICK_COLOR),
        hub_marker_color=_optional_string(mapping.get("hub_marker_color"), default=DEFAULT_HUB_MARKER_COLOR),
        hub_edge_color=_optional_string(mapping.get("hub_edge_color"), default=DEFAULT_HUB_EDGE_COLOR),
        hub_label_color=_optional_string(mapping.get("hub_label_color"), default=DEFAULT_HUB_LABEL_COLOR),
        hub_label_background_color=_optional_string(
            mapping.get("hub_label_background_color"),
            default=DEFAULT_HUB_LABEL_BACKGROUND_COLOR,
        ),
        font_family=_optional_string(mapping.get("font_family"), default="DejaVu Sans"),
        show_hub_coordinates=_boolean(
            mapping.get("show_hub_coordinates"),
            "visualization.show_hub_coordinates",
            default=True,
        ),
        figure_width=_positive_float(mapping.get("figure_width"), "visualization.figure_width"),
        figure_height=_positive_float(mapping.get("figure_height"), "visualization.figure_height"),
        dpi=_positive_int(mapping.get("dpi"), "visualization.dpi"),
        range_one_way_fill_color=_optional_string(
            mapping.get("range_one_way_fill_color"),
            default=DEFAULT_RANGE_ONE_WAY_FILL_COLOR,
        ),
        range_one_way_edge_color=_optional_string(
            mapping.get("range_one_way_edge_color"),
            default=DEFAULT_RANGE_ONE_WAY_EDGE_COLOR,
        ),
        range_one_way_alpha=_unit_interval(
            mapping.get("range_one_way_alpha"),
            "visualization.range_one_way_alpha",
            default=0.30,
        ),
        range_round_trip_fill_color=_optional_string(
            mapping.get("range_round_trip_fill_color"),
            default=DEFAULT_RANGE_ROUND_TRIP_FILL_COLOR,
        ),
        range_round_trip_edge_color=_optional_string(
            mapping.get("range_round_trip_edge_color"),
            default=DEFAULT_RANGE_ROUND_TRIP_EDGE_COLOR,
        ),
        range_round_trip_alpha=_unit_interval(
            mapping.get("range_round_trip_alpha"),
            "visualization.range_round_trip_alpha",
            default=0.36,
        ),
        overlap_fill_color=_optional_string(mapping.get("overlap_fill_color"), default=DEFAULT_OVERLAP_FILL_COLOR),
        overlap_edge_color=_optional_string(mapping.get("overlap_edge_color"), default=DEFAULT_OVERLAP_EDGE_COLOR),
        overlap_alpha=_unit_interval(mapping.get("overlap_alpha"), "visualization.overlap_alpha", default=0.42),
        throughput_colormap=_optional_string(mapping.get("throughput_colormap"), default="viridis"),
        throughput_heatmap_alpha=_unit_interval(
            mapping.get("throughput_heatmap_alpha"),
            "visualization.throughput_heatmap_alpha",
            default=0.65,
        ),
        throughput_color_percentile=_percentile(
            mapping.get("throughput_color_percentile"),
            "visualization.throughput_color_percentile",
            default=97.0,
        ),
        throughput_heatmap_sigma=_non_negative_float(
            mapping.get("throughput_heatmap_sigma"),
            "visualization.throughput_heatmap_sigma",
            default=1.0,
        ),
        throughput_contour_color=_optional_string(
            mapping.get("throughput_contour_color"),
            default="#000000",
        ),
        throughput_contour_linewidth=_positive_float(
            mapping.get("throughput_contour_linewidth"),
            "visualization.throughput_contour_linewidth",
        ),
    )


def _parse_outputs(
    value: Any,
    *,
    scenario: ScenarioMetadata,
    map_config: MapConfig,
) -> list[OutputConfig]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("outputs must be a list of mappings.")

    outputs: list[OutputConfig] = []
    for index, output_value in enumerate(value):
        field_name = f"outputs[{index}]"
        output_mapping = _require_mapping(output_value, field_name)
        output_type = _normalize_choice(
            output_mapping.get("type"),
            f"{field_name}.type",
            supported=SUPPORTED_OUTPUT_TYPES,
        )
        bounding_box_value = output_mapping.get("bounding_box", output_mapping.get("bbox"))
        bounding_box = (
            BoundingBox.from_mapping(bounding_box_value, f"{field_name}.bounding_box")
            if bounding_box_value is not None
            else map_config.bounding_box
        )
        contour_levels_value = output_mapping.get("contour_levels", ())
        outputs.append(
            OutputConfig(
                id=_optional_string(
                    output_mapping.get("id"),
                    default=Path(_optional_string(output_mapping.get("filename"), default=f"output_{index + 1}.png")).stem,
                ),
                type=output_type,
                title=_optional_string(output_mapping.get("title"), default=scenario.title),
                subtitle=_optional_string(output_mapping.get("subtitle"), default=scenario.subtitle),
                bounding_box=bounding_box,
                filename=Path(_require_string(output_mapping.get("filename"), f"{field_name}.filename")).expanduser(),
                show_hubs=_boolean(output_mapping.get("show_hubs"), f"{field_name}.show_hubs", default=True),
                color_scheme=_optional_string(output_mapping.get("color_scheme")),
                contour_levels=_float_tuple(contour_levels_value, f"{field_name}.contour_levels"),
                operational_legend=_parse_operational_legend(
                    output_mapping.get("operational_legend"),
                    f"{field_name}.operational_legend",
                ),
                vessel=_optional_string(output_mapping.get("vessel")),
            )
        )
    if not outputs:
        raise ValueError("At least one output is required.")
    return outputs


def _default_operational_legend_config(unit_type: str, *, enabled: bool) -> OperationalLegendConfig:
    return OperationalLegendConfig(
        enabled=enabled,
        unit_type=unit_type,
        display_mode="translated",
        consumption_rate_tons_per_day=DEFAULT_OPERATIONAL_UNIT_RATES_TPD[unit_type],
        unit_label=DEFAULT_OPERATIONAL_UNIT_LABELS[unit_type],
        unit_abbreviation=DEFAULT_OPERATIONAL_UNIT_ABBREVIATIONS[unit_type],
    )


def _parse_operational_legend(value: Any, field_name: str) -> OperationalLegendConfig:
    if value is None:
        return _default_operational_legend_config("ibct", enabled=False)
    if isinstance(value, bool):
        return _default_operational_legend_config("ibct", enabled=value)

    mapping = _require_mapping(value, field_name)
    unit_type = _normalize_choice(
        mapping.get("unit_type"),
        f"{field_name}.unit_type",
        supported=SUPPORTED_OPERATIONAL_UNIT_TYPES,
        default="ibct",
    )
    default_config = _default_operational_legend_config(unit_type, enabled=True)
    raw_rate_value = mapping.get("consumption_rate_tons_per_day", mapping.get("consumption_rate"))
    if unit_type == "custom" and raw_rate_value is None:
        raise ValueError(f"{field_name}.consumption_rate_tons_per_day is required when unit_type is 'custom'.")

    return OperationalLegendConfig(
        enabled=_boolean(mapping.get("enabled"), f"{field_name}.enabled", default=True),
        unit_type=unit_type,
        display_mode=_normalize_choice(
            mapping.get("display_mode"),
            f"{field_name}.display_mode",
            supported=SUPPORTED_OPERATIONAL_DISPLAY_MODES,
            default=default_config.display_mode,
        ),
        consumption_rate_tons_per_day=_positive_float(
            raw_rate_value,
            f"{field_name}.consumption_rate_tons_per_day",
            default=default_config.consumption_rate_tons_per_day,
        ),
        unit_label=_optional_string(mapping.get("unit_label"), default=default_config.unit_label),
        unit_abbreviation=_optional_string(
            mapping.get("unit_abbreviation"),
            default=default_config.unit_abbreviation,
        ),
    )


def _require_yaml():
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for YAML scenario files. Install the repo dependencies from requirements.txt."
        )
    return yaml


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return {str(key): copy.deepcopy(item) for key, item in value.items()}


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("Expected a string value.")
    stripped = value.strip()
    if not stripped:
        return default
    return stripped


def _float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _positive_float(value: Any, field_name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} is required.")
        value = default
    result = _float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return result


def _non_negative_float(value: Any, field_name: str, default: float = 0.0) -> float:
    if value is None:
        value = default
    result = _float(value, field_name)
    if result < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if result <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return result


def _boolean(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true or false.")
    return value


def _unit_interval(value: Any, field_name: str, default: float) -> float:
    if value is None:
        value = default
    result = _float(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return result


def _percentile(value: Any, field_name: str, default: float) -> float:
    if value is None:
        value = default
    result = _float(value, field_name)
    if not 0.0 < result <= 100.0:
        raise ValueError(f"{field_name} must be greater than 0 and at most 100.")
    return result


def _normalize_choice(
    value: Any,
    field_name: str,
    *,
    supported: set[str] | frozenset[str],
    default: str | None = None,
) -> str:
    candidate = _optional_string(value, default=default)
    if candidate is None:
        raise ValueError(f"{field_name} is required.")
    normalized = candidate.lower()
    if normalized not in supported:
        supported_values = ", ".join(sorted(supported))
        raise ValueError(f"{field_name} must be one of: {supported_values}.")
    return normalized


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of numeric values.")
    return tuple(_float(item, field_name) for item in value)
